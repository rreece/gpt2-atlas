"""Stage 2, step 2: the IOI circuit and its Hydra repair, in the SAE feature basis.

The dataset, logit difference and mean ablation are ported from ioi-hydra
(https://github.com/rreece/ioi-hydra, python/ioi_hydra.py): the same names, templates and seeds give the same 128
prompts, so its head-level results (clean LD 3.38; name movers mean-ablated: LD 3.53, direct effect lost 4.57, repair
103%) are the reference. Everything here runs on our HF backend, without TransformerLens:

- A head's output before W_O is the input to its layer's attn.c_proj, head h in columns h*d_head:(h+1)*d_head, so
  heads are read and ablated with a forward pre-hook on c_proj. TransformerLens's processed weights first fold
  LayerNorm 1 into W_V, which makes its bias part of the value bias, then fold that value bias into the output bias:
  its hook_z is the c_proj input minus (ln_1.bias W_V + b_V). Heads here are read and written in that convention,
  which makes zero ablation and per-head direct effects match ioi-hydra's. (Mean ablation is the same either way.)
- The final LayerNorm is (x - mean) / scale * gamma + beta, so with the scale of a given run the logit difference
  LD = logit(IO) - logit(S) is linear in the centered final residual stream x: LD = x . d / scale + beta . u, with
  u = W_U[IO] - W_U[S] and d = gamma * u. That splits LD exactly into per-head direct effects, and, with an SAE on the
  final residual stream (x = W_dec a + b_dec + error), into per-feature contributions plus the decoder bias and the
  SAE's error ("dark matter").

References: Wang et al. 2022 (IOI), https://arxiv.org/abs/2211.00593; McGrath et al. 2023 (the Hydra effect),
https://arxiv.org/abs/2307.15771.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from atlas import BOS, HFGPT2
from sae_gate import patch_resid_pre, resid_module

torch.set_grad_enabled(False)

# From ioi-hydra. Candidate names, filtered to those that are a single GPT-2 token with a leading space.
NAMES = [
    "Mary", "John", "Alice", "Bob", "Tom", "James", "Dan", "Sarah", "Michael", "David",
    "Paul", "Mark", "Lisa", "Anna", "Emma", "Kevin", "Jessica", "Andrew", "Richard", "Laura",
    "Steven", "Brian", "Rachel", "Jennifer", "Robert", "William", "Thomas", "Daniel", "Chris",
    "Matthew", "Joseph", "Charles", "George", "Edward", "Peter", "Eric", "Adam", "Amy", "Kate",
    "Sam", "Ben", "Jack", "Joe", "Mike", "Ryan", "Jason", "Justin", "Brandon", "Anthony",
    "Jane", "Ann", "Helen", "Linda", "Karen", "Susan", "Nancy", "Betty", "Martin", "Frank",
]

# [A] and [B] are the first two names, [S] is the repeated subject. The answer (IO) is omitted.
TEMPLATES = [
    "When [A] and [B] went to the store, [S] gave a drink to",
    "When [A] and [B] went to the park, [S] gave a ball to",
    "Then, [A] and [B] went to the office. [S] gave a book to",
    "After [A] and [B] went to the restaurant, [S] gave a ring to",
    "While [A] and [B] were working at the school, [S] gave a computer to",
    "Friends [A] and [B] found a bone at the garden. [S] gave it to",
    "Then, [A] and [B] had a long argument, and afterwards [S] said to",
    "The local big house where [A] and [B] lived was quiet. [S] handed a key to",
]

NAME_MOVERS = [(9, 9), (9, 6), (10, 0)]


@dataclass
class IOIData:
    prompts: list[str]
    corrupt_prompts: list[str]
    clean_tokens: torch.Tensor  # [N, T]: BOS + prompt, right-padded
    corrupt_tokens: torch.Tensor  # [N, T], the matched ABC prompts (same template, so same length)
    io: torch.Tensor  # [N] token of the indirect object (the answer)
    s: torch.Tensor  # [N] token of the subject
    end: torch.Tensor  # [N] index of the final prompt token
    template_id: torch.Tensor  # [N]
    order: list[str]  # "ABBA" or "BABA"

    def __len__(self) -> int:
        return len(self.prompts)


def make_ioi_dataset(tokenizer, n_per_template: int = 16, templates=TEMPLATES, seed: int = 0) -> IOIData:
    """Balanced ABBA / BABA IOI prompts plus matched ABC corruptions, as in ioi-hydra (same names, order and seed)."""
    rng = random.Random(seed)
    names = [n for n in NAMES if len(tokenizer(" " + n)["input_ids"]) == 1]
    fill = lambda t, a, b, s: t.replace("[A]", a).replace("[B]", b).replace("[S]", s)
    prompts, corrupt, io_names, s_names, tids, orders = [], [], [], [], [], []
    for tid, tmpl in enumerate(templates):
        for i in range(n_per_template):
            io, s, c1, c2, c3 = rng.sample(names, 5)
            prompts.append(fill(tmpl, io, s, s) if i % 2 == 0 else fill(tmpl, s, io, s))
            orders.append("ABBA" if i % 2 == 0 else "BABA")
            corrupt.append(fill(tmpl, c1, c2, c3))
            io_names.append(io)
            s_names.append(s)
            tids.append(tid)

    def tokens(texts):
        ids = [[BOS] + tokenizer(t)["input_ids"] for t in texts]
        T = max(map(len, ids))
        return torch.tensor([x + [BOS] * (T - len(x)) for x in ids]), torch.tensor([len(x) - 1 for x in ids])

    clean, end = tokens(prompts)
    corr, end_c = tokens(corrupt)
    assert torch.equal(end, end_c), "a corrupt prompt differs in length from its clean prompt"
    tok1 = lambda n: tokenizer(" " + n)["input_ids"][0]
    return IOIData(prompts, corrupt, clean, corr, torch.tensor([tok1(n) for n in io_names]),
                   torch.tensor([tok1(n) for n in s_names]), end, torch.tensor(tids), orders)


# ---------------------------------------------------------------------------------------------
# Heads: read and ablate at attn.c_proj
# ---------------------------------------------------------------------------------------------


def _head_slice(model: HFGPT2, h: int) -> slice:
    d_head = model.cfg.d_model // model.model.config.n_head
    return slice(h * d_head, (h + 1) * d_head)


def value_bias(model: HFGPT2, layer: int) -> torch.Tensor:
    """The value bias TransformerLens's processing folds out of every head of a layer, concatenated [d_model]:
    ln_1.bias W_V + b_V, with W_V and b_V the last thirds of attn.c_attn's weight and bias."""
    d = model.cfg.d_model
    block = model.model.h[layer]
    return block.ln_1.bias @ block.attn.c_attn.weight[:, 2 * d : 3 * d] + block.attn.c_attn.bias[2 * d : 3 * d]


def capture_z(model: HFGPT2, toks: torch.Tensor, layers) -> dict[int, torch.Tensor]:
    """Every head's output before W_O in TransformerLens's convention (hook_z, value bias folded out), heads
    concatenated: {layer: [N, T, d_model]}."""
    out, handles = {}, []
    for l in layers:
        handles.append(model.model.h[l].attn.c_proj.register_forward_pre_hook(
            lambda m, args, l=l: out.__setitem__(l, args[0].detach() - value_bias(model, l))))
    try:
        model.model(toks)
    finally:
        for h in handles:
            h.remove()
    return out


def mean_reference(z: dict[int, torch.Tensor], template_id: torch.Tensor) -> dict[int, torch.Tensor]:
    """ioi-hydra's mean ablation: each head's output replaced, per position, by its mean over the corrupt prompts of the
    same template. z: capture_z on the corrupt prompts."""
    ref = {}
    for l, zl in z.items():
        m = torch.zeros_like(zl)
        for tid in template_id.unique():
            mask = (template_id == tid).to(zl.device)
            m[mask] = zl[mask].mean(0, keepdim=True)
        ref[l] = m
    return ref


@contextmanager
def ablate_heads(model: HFGPT2, heads, ref: dict[int, torch.Tensor] | None = None):
    """Inside the block, `heads` [(layer, head), ...] output ref[layer]'s columns, in capture_z's convention (None:
    zero ablation, hook_z = 0)."""
    handles = []
    by_layer: dict[int, list[int]] = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)
    for l, hs in by_layer.items():
        def pre(m, args, l=l, hs=hs):
            x = args[0].clone()
            b = value_bias(model, l)
            for h in hs:
                sl = _head_slice(model, h)
                x[..., sl] = b[sl] + (0.0 if ref is None else ref[l][..., sl])
            return (x, *args[1:])
        handles.append(model.model.h[l].attn.c_proj.register_forward_pre_hook(pre))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------------------------
# A run: logit difference, its split into heads, and the residual stream at the final token
# ---------------------------------------------------------------------------------------------


def run(model: HFGPT2, ds: IOIData, resid_layers=(10, 11, 12)) -> dict:
    """One forward pass of the clean prompts (under any ablation the caller has set up). Returns, at each prompt's
    final token: the logit difference ld [N]; the centered residual stream before each of resid_layers [N, d_model]
    (12: the final residual stream); the final LayerNorm scale [N]; each head's direct effect de [N, L, H]; and the
    LD direction d [N, d_model] and constant beta . u [N], so that ld = x_final . d / scale + const exactly."""
    dev = model.model.device
    n, end, io, s = torch.arange(len(ds), device=dev), ds.end.to(dev), ds.io.to(dev), ds.s.to(dev)
    caps, handles = {}, []
    for l in resid_layers:
        handles.append(patch_resid_pre(resid_module(model, l), lambda x, l=l: caps.__setitem__(l, x[n, end].clone())))
    z = {}
    for l in range(model.cfg.n_layers):
        handles.append(model.model.h[l].attn.c_proj.register_forward_pre_hook(
            lambda m, args, l=l: z.__setitem__(l, args[0][n, end] - value_bias(model, l))))
    try:
        h = model.model(ds.clean_tokens.to(model.model.device)).last_hidden_state[n, end]
    finally:
        for hd in handles:
            hd.remove()
    wte, ln = model.model.wte.weight, model.model.ln_f
    logits_io, logits_s = (h * wte[io]).sum(-1), (h * wte[s]).sum(-1)
    u = wte[io] - wte[s]  # [N, d_model]
    d = ln.weight * u
    const = (ln.bias * u).sum(-1)
    x = caps[12] if 12 in caps else None
    scale = (x.pow(2).mean(-1) + ln.eps).sqrt() if x is not None else None
    de = None
    if scale is not None:
        H = model.model.config.n_head
        de = torch.zeros(len(ds), model.cfg.n_layers, H)
        for l, zl in z.items():
            W = model.model.h[l].attn.c_proj.weight  # Conv1D: [d_in, d_out], input columns per head
            for hh in range(H):
                sl = _head_slice(model, hh)
                out = zl[:, sl] @ W[sl]
                out = out - out.mean(-1, keepdim=True)
                de[:, l, hh] = ((out * d).sum(-1) / scale).cpu()
    return {"ld": (logits_io - logits_s).cpu(), "resid": {l: v.cpu() for l, v in caps.items()},
            "scale": None if scale is None else scale.cpu(), "de": de, "d": d.cpu(), "const": const.cpu()}


def feature_attribution(sae, x: torch.Tensor, d: torch.Tensor, scale: torch.Tensor) -> dict:
    """Split x . d / scale (the logit difference less its constant) into the SAE's features, its decoder bias and its
    error: x = W_dec a + b_dec + error. Returns acts [N, d_sae], per-feature contributions [N, d_sae], bias [N] and
    error [N]; contributions + bias + error = x . d / scale exactly."""
    a = sae.encode(x)
    x_hat = sae.decode(a)
    per_feature = a * (d @ sae.W_dec.T) / scale[:, None]  # a_f (W_dec[f] . d) / scale
    bias = (sae.b_dec * d).sum(-1) / scale
    error = ((x - x_hat) * d).sum(-1) / scale
    return {"acts": a, "contrib": per_feature, "bias": bias, "error": error}


def bootstrap_mean(x, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% bootstrap interval over prompts, as in ioi-hydra."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
