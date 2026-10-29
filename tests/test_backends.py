"""HFGPT2 must reproduce the TransformerLens bridge (compatibility mode) at every hook point it offers.

Downloads GPT-2 small on first use and runs on CPU, about 10 s.
"""

import numpy as np
import pytest
import torch

from atlas import BOS, HFGPT2, TLGPT2, _Stop, default_hooks, run

HOOKS = [
    f"blocks.{l}.{h}"
    for l in (0, 5, 11)
    for h in ("hook_resid_pre", "hook_resid_mid", "hook_resid_post", "hook_attn_out", "hook_mlp_out",
              "mlp.hook_pre", "mlp.hook_post")
]
TEXT = ("When Mary and John went to the store, John gave a drink to Mary. The store was closed on Sunday, "
        "so they walked to the park instead and talked about the weather for an hour.")


@pytest.fixture(scope="module")
def models():
    return HFGPT2("cpu"), TLGPT2("cpu")


@pytest.fixture(scope="module")
def tokens(models):
    from transformers import AutoTokenizer

    text = AutoTokenizer.from_pretrained("gpt2")(TEXT)["input_ids"][:32]
    rand = torch.randint(0, 50257, (3, 31), generator=torch.Generator().manual_seed(0)).tolist()
    return torch.tensor([[BOS] + text[:31]] + [[BOS] + r for r in rand])  # [4, 32]: one real, three random


def capture(model, tokens, names):
    acts = {}
    model.run(tokens, {n: (lambda act, n=n: acts.__setitem__(n, act.clone())) for n in names})
    return acts


def test_cfg_matches(models):
    hf, tl = models
    assert (hf.cfg.n_layers, hf.cfg.d_model, hf.cfg.d_mlp) == (tl.cfg.n_layers, tl.cfg.d_model, tl.cfg.d_mlp)
    assert default_hooks(hf.cfg) == default_hooks(tl.cfg)


@pytest.mark.parametrize("name", HOOKS)
def test_every_hook_matches_bridge(models, tokens, name):
    hf, tl = models
    a, b = capture(hf, tokens, [name])[name], capture(tl, tokens, [name])[name]
    assert a.shape == b.shape
    # Same math, different op order: agreement to float32 rounding relative to the activation scale.
    # BOS included: its outlier norm is the hardest case.
    assert (a - b).abs().max() <= 2e-5 * b.abs().max() + 1e-5, (a - b).abs().max()


def test_residual_needs_centering(models, tokens):
    """Guards the one convention difference: raw HF residuals differ from the bridge by their mean."""
    hf, tl = models
    name = "blocks.5.hook_resid_pre"
    raw = {}
    h = hf.model.h[5].register_forward_pre_hook(lambda m, args, kw: raw.__setitem__("x", args[0] if args else kw["hidden_states"]), with_kwargs=True)
    try:
        hf.model(tokens)
    finally:
        h.remove()
    ref = capture(tl, tokens, [name])[name]
    assert (raw["x"] - ref).abs().max() > 1e-2
    torch.testing.assert_close(raw["x"] - raw["x"].mean(-1, keepdim=True), ref, rtol=1e-4, atol=1e-4)


def test_stop_skips_later_hooks(models, tokens):
    hf, _ = models
    seen = []

    def stop(act):
        seen.append("mlp0")
        raise _Stop

    hf.run(tokens, {"blocks.0.mlp.hook_post": stop, "blocks.1.hook_resid_pre": lambda act: seen.append("resid1")})
    assert seen == ["mlp0"]


def test_unknown_hook_raises(models, tokens):
    with pytest.raises(KeyError):
        models[0].run(tokens, {"blocks.0.attn.hook_z": lambda act: None})


def test_run_gives_same_atlas(models, tokens):
    hf, tl = models
    toks = tokens.numpy().astype(np.uint16)
    a, b = run(hf, toks, 0, len(toks), batch_size=2, k=5, progress=False), run(tl, toks, 0, len(toks), batch_size=2, k=5, progress=False)
    for name in a.acc:
        ma, mb = a.acc[name]["moments"], b.acc[name]["moments"]
        assert ma.n == mb.n == 4 * 31
        scale = mb.max.abs().max()
        assert (ma.mean - mb.mean).abs().max() <= 2e-5 * scale
        assert (ma.var.sqrt() - mb.var.sqrt()).abs().max() <= 2e-5 * scale
        for kind in ("topk", "bottomk"):
            if kind in a.acc[name]:
                va, vb = a.acc[name][kind].values, b.acc[name][kind].values
                torch.testing.assert_close(va, vb, rtol=1e-4, atol=2e-5 * max(scale.item(), vb.abs().max().item()))
        # Values within rounding of a bin edge may land in the neighbouring bin.
        ca, cb = a.acc[name]["hist"].counts, b.acc[name]["hist"].counts
        assert (ca - cb).abs().sum() <= 2e-3 * cb.sum()


def test_run_on_a_subset_stops_after_the_last_hook_reached(models, tokens):
    # resid_pre of layer 2 is listed last but reached before layer 2's MLP: stopping there would skip the MLP hook.
    hf, _ = models
    toks = tokens.numpy().astype(np.uint16)
    hooks = {"blocks.2.mlp.hook_pre": hf.cfg.d_mlp, "blocks.2.hook_resid_pre": hf.cfg.d_model}
    a = run(hf, toks, 0, len(toks), batch_size=2, k=5, progress=False, hooks=hooks)
    assert set(a.acc) == set(hooks)
    assert all(a.acc[name]["moments"].n == 4 * 31 for name in hooks)


def test_raw_backend_differs_only_by_the_residual_mean(models, tokens):
    """hf-raw, for the dependence on centering the residual stream: MLP activations unchanged, residual uncentered."""
    hf, _ = models
    raw = HFGPT2("cpu", center=False)
    names = ["blocks.5.hook_resid_pre", "blocks.11.hook_resid_post", "blocks.5.mlp.hook_pre", "blocks.5.mlp.hook_post"]
    a, b = capture(hf, tokens, names), capture(raw, tokens, names)
    for name in names[2:]:
        torch.testing.assert_close(b[name], a[name])
    for name in names[:2]:
        assert (b[name] - a[name]).abs().max() > 1e-2
        torch.testing.assert_close(b[name] - b[name].mean(-1, keepdim=True), a[name], rtol=1e-5, atol=1e-4)


def test_resid_pre_patch_identity_and_zero_ablation(models, tokens):
    """The splicing hook behind the SAE gate's CE loss score: an identity replacement leaves the loss unchanged, the
    replacement sees the centered stream (the TransformerLens convention), and a zero ablation changes the loss."""
    from sae_gate import next_token_ce, patch_resid_pre

    hf, tl = models
    clean = next_token_ce(hf, tokens)
    seen = {}
    h = patch_resid_pre(hf.model.h[5], lambda x: seen.setdefault("x", x) * 1.0)
    try:
        same = next_token_ce(hf, tokens)
    finally:
        h.remove()
    torch.testing.assert_close(same, clean, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(seen["x"], capture(tl, tokens, ["blocks.5.hook_resid_pre"])["blocks.5.hook_resid_pre"], rtol=1e-4, atol=1e-4)
    h = patch_resid_pre(hf.model.h[5], torch.zeros_like)
    try:
        zero = next_token_ce(hf, tokens)
    finally:
        h.remove()
    assert (zero.mean() - clean.mean()).abs() > 0.5


def test_run_with_sae_features_and_events(models, tokens):
    """Stage 2: an SAE on resid_pre turns its features into atlas units, and events record every nonzero activation."""
    from types import SimpleNamespace

    hf, _ = models
    g = torch.Generator().manual_seed(0)
    W = torch.randn(hf.cfg.d_model, 64, generator=g) / 10
    sae = SimpleNamespace(cfg=SimpleNamespace(d_sae=64), encode=lambda x: torch.relu(x @ W - 1.0))
    toks = tokens.numpy().astype(np.uint16)
    hooks = {"blocks.2.hook_resid_pre": hf.cfg.d_model}
    a = run(hf, toks, 0, len(toks), batch_size=2, k=3, progress=False, hooks=hooks, saes={2: sae}, events=(2,))
    name = "blocks.2.hook_resid_pre.sae"
    assert set(a.acc) == set(hooks) | {name} and a.acc[name]["moments"].n == 4 * 31
    # Events are exactly the nonzero activations, at the right (row, pos).
    acts = capture(hf, tokens, ["blocks.2.hook_resid_pre"])["blocks.2.hook_resid_pre"]
    feats = sae.encode(acts)[:, 1:]
    idx = np.concatenate([c[0] for c in a.events[2]])
    feat = np.concatenate([c[1] for c in a.events[2]])
    val = np.concatenate([c[2] for c in a.events[2]])
    assert len(idx) == int((feats != 0).sum()) == int(a.acc[name]["moments"].n_pos.sum())
    T = toks.shape[1]
    np.testing.assert_allclose(val, feats[idx // T, idx % T - 1, feat].numpy(), rtol=1e-4, atol=1e-5)


def test_heads_read_and_ablated_at_c_proj_match_processed_transformerlens(models, tokens):
    """IOI analysis (python/ioi.py): a head's output read at attn.c_proj, with the folded value bias removed, is
    processed TransformerLens's hook_z, and zero-ablating it there changes the losses exactly as zeroing hook_z does.
    (Processed HookedTransformer, the convention of ioi-hydra's head-level results.)"""
    from transformer_lens import HookedTransformer

    from ioi import ablate_heads, capture_z
    from sae_gate import next_token_ce

    hf, _ = models
    ht = HookedTransformer.from_pretrained("gpt2", device="cpu")
    _, cache = ht.run_with_cache(tokens, names_filter="blocks.9.attn.hook_z")
    torch.testing.assert_close(capture_z(hf, tokens, [9])[9], cache["blocks.9.attn.hook_z"].flatten(-2), rtol=1e-4, atol=1e-4)

    def zero_9_6(x, hook):
        x[:, :, 6] = 0.0
        return x

    with ablate_heads(hf, [(9, 6)]):
        ours = next_token_ce(hf, tokens)
    theirs = ht.run_with_hooks(tokens, return_type="loss", loss_per_token=True, fwd_hooks=[("blocks.9.attn.hook_z", zero_9_6)])
    assert (ours - next_token_ce(hf, tokens)).abs().max() > 1e-3  # the ablation does something
    torch.testing.assert_close(ours, theirs, rtol=1e-4, atol=1e-4)


def test_resample_ablation_with_the_clean_prompts_changes_nothing(models):
    """python/ioi.py: resample-ablating heads with references taken from the clean prompts themselves must leave the
    logit difference unchanged (the reference, the value-bias convention and the write-back all round-trip)."""
    from dataclasses import replace

    import ioi
    from atlas import load_tokenizer

    hf, _ = models
    ds = ioi.make_ioi_dataset(load_tokenizer(), n_per_template=2)
    same = replace(ds, corrupt_tokens=ds.clean_tokens)
    clean = ioi.run(hf, same, resid_layers=(12,))
    with ioi.ablate_heads(hf, ioi.NAME_MOVERS, ioi.ablation_reference(hf, same, "resample")):
        ablated = ioi.run(hf, same, resid_layers=(12,))
    torch.testing.assert_close(ablated["ld"], clean["ld"], rtol=1e-4, atol=1e-4)
    with ioi.ablate_heads(hf, ioi.NAME_MOVERS, ioi.ablation_reference(hf, same, "zero")):
        zeroed = ioi.run(hf, same, resid_layers=(12,))
    assert (zeroed["ld"] - clean["ld"]).abs().max() > 0.1  # a real ablation does something


def test_ioi_reproduces_ioi_hydra(models):
    """python/ioi.py on our HF backend must reproduce ioi-hydra's head-level result on its 128 prompts: clean LD 3.38;
    with the name movers mean-ablated, LD 3.53 and 4.57 logits of direct effect lost (a 103% repair)."""
    import ioi
    from atlas import load_tokenizer

    hf, _ = models
    ds = ioi.make_ioi_dataset(load_tokenizer())
    clean = ioi.run(hf, ds, resid_layers=(12,))
    with ioi.ablate_heads(hf, ioi.NAME_MOVERS, ioi.ablation_reference(hf, ds, "mean")):
        ablated = ioi.run(hf, ds, resid_layers=(12,))
    lost = sum(clean["de"][:, l, h].mean() - ablated["de"][:, l, h].mean() for l, h in ioi.NAME_MOVERS)
    assert round(clean["ld"].mean().item(), 2) == 3.38
    assert round(ablated["ld"].mean().item(), 2) == 3.53
    assert round(lost.item(), 2) == 4.57
