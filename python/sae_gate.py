"""The gate into stage 2: does a pretrained SAE reproduce its release's metrics on this pipeline?

A pretrained SAE's features only mean something if our activations are the ones it was trained on: same hook point,
same processing convention, same BOS and context handling. This measures, on an atlas's tokens and with our HF
backend, the metrics every release reports, with SAELens's definitions (sae_lens.evals):

- L0: mean number of nonzero features per token, BOS excluded.
- Variance explained: 1 - E||x - x_hat||^2 / (E||x||^2 - ||E x||^2) over tokens, BOS excluded.
- CE loss score: (CE_zero - CE_sae) / (CE_zero - CE_clean), next-token loss over every position, with the
  reconstruction spliced in at every position but BOS and the zero ablation at every position.
- Dead fraction: features that never fire on the tokens seen (not a release metric; depends on the sample size).

Everything is summed per corpus file, so errors come from the same cluster bootstrap over files as the atlas. On a
subset of the same tokens it also recomputes the residual stream, the features and the per-token losses with
TransformerLens's HookedTransformer loaded as SAELens recommends, an independent check of our conventions.

    python -m sae_gate results/atlas --out results/sae-gate              # Bloom's gpt2-small-res-jb, layers 0-11
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from atlas import HFGPT2, bootstrap_scale, default_device, git_commit, row_files

torch.set_grad_enabled(False)


def load_sae(release: str, sae_id: str, device):
    from sae_lens import SAE

    sae = SAE.from_pretrained(release, sae_id, device=device)
    return sae[0] if isinstance(sae, tuple) else sae


def patch_resid_pre(block, fn):
    """Forward pre-hook on a GPT-2 block: fn(x) sees the centered residual stream entering the block (the
    TransformerLens convention) and may return a replacement, also centered, or None to leave it. Each token's mean
    over d_model is added back, which changes nothing downstream: every reader of the stream is a LayerNorm."""

    def pre(module, args, kwargs):
        x = args[0] if args else kwargs["hidden_states"]
        mu = x.mean(-1, keepdim=True)
        new = fn(x - mu)
        if new is None:
            return None
        new = new + mu
        return ((new, *args[1:]), kwargs) if args else (args, {**kwargs, "hidden_states": new})

    return block.register_forward_pre_hook(pre, with_kwargs=True)


def next_token_ce(model: HFGPT2, toks: torch.Tensor) -> torch.Tensor:
    """[B, T] tokens -> [B, T - 1] next-token cross-entropy, through the tied unembedding."""
    h = model.model(toks).last_hidden_state
    logits = h @ model.model.wte.weight.T
    return torch.nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1).float(), toks[:, 1:].flatten(),
                                             reduction="none").view(toks.shape[0], -1)


def reconstruct(sae, x: torch.Tensor) -> torch.Tensor:
    return sae.decode(sae.encode(x))


def gate_batch(model: HFGPT2, sae, layer: int, toks: torch.Tensor) -> dict:
    """Sums for one batch: the SAE on resid_pre of `layer` (positions 1..T-1), and the three next-token losses."""
    block = model.model.h[layer]
    cap = {}
    h = patch_resid_pre(block, lambda x: cap.__setitem__("x", x))
    try:
        ce_clean = next_token_ce(model, toks)
    finally:
        h.remove()
    B, T, d = cap["x"].shape
    x = cap["x"][:, 1:].reshape(-1, d).float()
    feats = sae.encode(x)
    x_hat = sae.decode(feats)
    active = feats != 0

    def splice(xc):  # reconstruction everywhere but BOS, as sae_lens.evals does
        out = xc.clone()
        out[:, 1:] = reconstruct(sae, xc[:, 1:].reshape(-1, d)).view(B, T - 1, d)
        return out

    losses = {"ce_clean": ce_clean}
    for key, fn in (("ce_sae", splice), ("ce_zero", torch.zeros_like)):
        h = patch_resid_pre(block, fn)
        try:
            losses[key] = next_token_ce(model, toks)
        finally:
            h.remove()
    return {
        # Per-token reductions in float32 on the device (MPS has no float64), summed over tokens in float64.
        "n": x.shape[0], "sum_x": x.cpu().double().sum(0), "sum_sq": float((x**2).sum(-1).cpu().double().sum()),
        "sum_sq_resid": float(((x - x_hat) ** 2).sum(-1).cpu().double().sum()), "sum_l0": float(active.sum()),
        "fires": active.sum(0).cpu(), "n_ce": ce_clean.numel(),
        **{k: float(v.cpu().double().sum()) for k, v in losses.items()},
    }


def add(acc: dict | None, b: dict) -> dict:
    return dict(b) if acc is None else {k: acc[k] + b[k] for k in acc}


def metrics(accs: list[dict], weights=None) -> dict:
    """Release metrics from per-file sums, each file counted weights[i] times (1 by default)."""
    w = np.ones(len(accs)) if weights is None else weights
    s = {k: sum(wi * a[k] for wi, a in zip(w, accs)) for k in ("n", "sum_x", "sum_sq", "sum_sq_resid", "sum_l0", "n_ce",
                                                               "ce_clean", "ce_sae", "ce_zero")}
    total_var = s["sum_sq"] / s["n"] - float((s["sum_x"] / s["n"]).pow(2).sum())
    ce = {k: s[k] / s["n_ce"] for k in ("ce_clean", "ce_sae", "ce_zero")}
    return {"l0": s["sum_l0"] / s["n"], "var_explained": 1 - (s["sum_sq_resid"] / s["n"]) / total_var, **ce,
            "ce_score": (ce["ce_zero"] - ce["ce_sae"]) / (ce["ce_zero"] - ce["ce_clean"])}


def tl_check(sae, layer: int, toks: torch.Tensor, ours: HFGPT2) -> dict:
    """The same quantities from TransformerLens's HookedTransformer, loaded as SAELens recommends, on the same tokens:
    worst differences against our HF pipeline. Loaded once per process (cached on the function)."""
    from transformer_lens import HookedTransformer

    if not hasattr(tl_check, "model"):
        tl_check.model = HookedTransformer.from_pretrained_no_processing(
            "gpt2", device=toks.device, **sae.cfg.metadata.model_from_pretrained_kwargs)
    tl = tl_check.model
    name = f"blocks.{layer}.hook_resid_pre"
    _, cache = tl.run_with_cache(toks, names_filter=name)
    x_tl = cache[name]
    cap = {}
    h = patch_resid_pre(ours.model.h[layer], lambda x: cap.__setitem__("x", x))
    try:
        ce_ours = next_token_ce(ours, toks)
    finally:
        h.remove()
    x_ours = cap["x"]
    f_tl, f_ours = sae.encode(x_tl[:, 1:]), sae.encode(x_ours[:, 1:])
    ce_tl = tl(toks, return_type="loss", loss_per_token=True)
    scale = x_tl[:, 1:].abs().max()
    return {"tl_resid_max_rel_diff": float((x_ours[:, 1:] - x_tl[:, 1:]).abs().max() / scale),
            "tl_l0_ours": float((f_ours != 0).sum(-1).float().mean()), "tl_l0_tl": float((f_tl != 0).sum(-1).float().mean()),
            "tl_ce_max_abs_diff": float((ce_ours - ce_tl).abs().max()), "tl_ce_mean_ours": float(ce_ours.mean()),
            "tl_ce_mean_tl": float(ce_tl.mean())}


def main(argv=None):
    p = argparse.ArgumentParser(prog="sae_gate", description=__doc__.split("\n\n")[0])
    p.add_argument("atlas", type=Path, help="an atlas directory: its tokens.npy and docs.npy are used")
    p.add_argument("--out", type=Path, default=Path("results/sae-gate"))
    p.add_argument("--release", default="gpt2-small-res-jb")
    p.add_argument("--layers", default="0-11", help="resid_pre layers, e.g. 0-11 or 0,6,10")
    p.add_argument("--rows-per-file", type=int, default=250)
    p.add_argument("--check-rows", type=int, default=32, help="rows for the TransformerLens cross-check, per layer")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--device", default=default_device())
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

    a, _, b = args.layers.partition("-")
    layers = list(range(int(a), int(b) + 1)) if b else [int(x) for x in args.layers.split(",")]
    release = get_pretrained_saes_directory()[args.release]
    tokens = np.load(args.atlas / "tokens.npy", mmap_mode="r")
    files = row_files(args.atlas)
    rows_by_file = {int(f): np.flatnonzero(files == f)[: args.rows_per_file] for f in np.unique(files)}
    model = HFGPT2(args.device)
    rng = np.random.default_rng(0)
    out = []
    for layer in layers:
        t0 = time.time()
        sae_id = f"blocks.{layer}.hook_resid_pre"
        sae = load_sae(args.release, sae_id, args.device)
        meta = sae.cfg.metadata
        per_file = []
        for f, rows in rows_by_file.items():
            acc = None
            for r0 in range(0, len(rows), args.batch_size):
                toks = torch.from_numpy(tokens[rows[r0 : r0 + args.batch_size]].astype(np.int64)).to(args.device)
                acc = add(acc, gate_batch(model, sae, layer, toks))
            per_file.append(acc)
        m = metrics(per_file)
        F = len(per_file)
        w = rng.multinomial(F, np.full(F, 1 / F), size=args.n_boot)
        reps = [metrics(per_file, wi) for wi in w]
        scale = bootstrap_scale(np.arange(F), True)
        fires = sum(a["fires"] for a in per_file)
        check = tl_check(sae, layer, torch.from_numpy(tokens[rows_by_file[0][: args.check_rows]].astype(np.int64)).to(args.device), model)
        row = {"layer": layer, "hook": sae_id, "release": args.release, "d_sae": sae.cfg.d_sae,
               "n_tokens": int(sum(a["n"] for a in per_file)), **m,
               **{f"{k}_err": float(np.std([r[k] for r in reps]) * scale) for k in m},
               "dead_frac": float((fires == 0).float().mean()),
               "l0_expected": release.expected_l0.get(sae_id), "var_explained_expected": release.expected_var_explained.get(sae_id),
               "sae_hook": meta.hook_name, "sae_dataset": meta.dataset_path, "sae_context_size": meta.context_size,
               "sae_prepend_bos": meta.prepend_bos, "sae_processing": json.dumps(meta.model_from_pretrained_kwargs), **check}
        out.append(row)
        print(f"layer {layer:2d}: L0 {m['l0']:6.1f} ± {row['l0_err']:.1f} (expected {row['l0_expected']}), "
              f"var. explained {m['var_explained']:.3f} ± {row['var_explained_err']:.3f} (expected {row['var_explained_expected']}), "
              f"CE score {m['ce_score']:.3f} ± {row['ce_score_err']:.3f}, dead {row['dead_frac']:.1%}; "
              f"TL check: resid {check['tl_resid_max_rel_diff']:.1e}, CE {check['tl_ce_max_abs_diff']:.1e}  ({time.time() - t0:.0f}s)")
        del sae
    df = pd.DataFrame(out)
    df.to_parquet(args.out / "gate.parquet", index=False)
    (args.out / "manifest.json").write_text(json.dumps({
        "atlas": str(args.atlas), "release": args.release, "layers": layers, "rows_per_file": args.rows_per_file,
        "check_rows": args.check_rows, "git_commit": git_commit(), "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2) + "\n")
    print(f"-> {args.out / 'gate.parquet'}")


if __name__ == "__main__":
    main()
