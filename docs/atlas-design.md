# GPT-2 Activation Atlas — Design Doc

*Ryan Reece, 2026-10-22, updated 2026-10-28.*

## Summary

We are building a queryable atlas of what every unit in GPT-2 small does over a large text corpus: per-unit statistics, top-activating contexts and residual-stream geometry, first in the neuron basis and then in pretrained SAE feature bases. Stage 1 is finished: 1M OpenWebText tokens on an M4 laptop, in 16 shards merged exactly, with errors that include the file-to-file term and its corpus, position and centering systematics measured. Stage 2 is under way: pretrained SAEs passed a gate on this pipeline, their features are atlas units at layers 6 and 9, and the IOI circuit's self-repair has been studied in three dictionaries.

The atlas grows out of [ioi-hydra](https://github.com/rreece/ioi-hydra), a replication of the IOI circuit and its Hydra-effect self-repair. That project cached every activation for 128 prompts. The atlas asks the same kind of questions over a text corpus, where caching everything is impossible, so the whole design is built around streaming reductions instead. It is deliberately laptop-sized: 1M tokens from ~8k documents, with bootstrap errors, in about 5 minutes on an M4.

## Goals and non-goals

The atlas should answer "what does this unit respond to, how often, and how strongly?" for any unit in seconds, with numbers that are reproducible and carry known uncertainties.

**Goals**

- Per-unit summaries for every MLP neuron (12 × 3072) and residual dimension (13 × 768): count, mean, std, max, firing rate, quantiles, histogram.
- Top-k exemplars per unit (and bottom-k for residual dimensions, which are signed), stored as pointers into the token stream and shown with decoded context.
- Residual-stream mean and covariance per layer, for PCA, whitening and outlier dimensions.
- The same outputs in SAE feature bases (stage 2), so neurons and features can be compared side by side.
- Horizontal scaling: shards run on any number of machines and merge exactly.
- Every number traceable to a model revision, corpus shard, code commit and processing convention.

**Non-goals**

- Live production monitoring. That is a separate design: in-graph probes inside the serving engine (see the vLLM notes from 2026-10-21).
- Storing raw dense activations. If something isn't in the atlas, recompute it from tokens.
- Models other than GPT-2 small, for now. Nothing should hard-code GPT-2 beyond the default hook list.
- Duplicating [Neuronpedia](https://www.neuronpedia.org/gpt2-small). Where it already hosts dashboards for a dictionary, we use its outputs and spend our effort on what's missing: systematics and comparisons across dictionaries. For example, it shows which of Bloom's residual-stream SAE features (`res-jb`, the dictionary the atlas uses at layers 6, 9 and the final residual) fire on each token of an IOI prompt, in [this search](https://www.neuronpedia.org/gpt2-small?sourceSet=res-jb&selectedLayers=%5B%5D&sortIndexes=%5B%5D&ignoreBos=true&q=Then%2C%20John%20and%20Mike%20went%20to%20the%20office.%20John%20gave%20a%20book%20to), and gives each feature a dashboard, such as [12-res-jb/9680](https://www.neuronpedia.org/gpt2-small/12-res-jb/9680), a "say John" feature at the final residual whose top logits are all "John". OpenAI's 32k and 128k residual SAEs are there too (`res_post_32k-oai`, `res_post_128k-oai`); the atlas checks single features against its dashboards ([results](results-stage2.md#2-the-atlas-in-the-sae-basis)).

## Design principles

Activations are a pure function of (weights, tokens), so we keep tokens plus reductions and recompute everything else. The rest of the design follows from that.

1. **Store reductions and pointers, not tensors.** Dense activations for 1B tokens are about 20 TB for the residual stream alone and roughly 300 PB for every hook point. An exemplar is a `(row, pos)` pointer into `tokens.npy` (uint16, 2 bytes per token); rerunning one row costs milliseconds.
2. **Reduce on the device.** Hooks reduce each batch on the GPU; only small summaries cross to the CPU. The host transfer, not FLOPs, is the bottleneck to avoid.
3. **Every accumulator merges exactly.** Moments and covariance use Chan et al.'s parallel formulas in float64; histograms use fixed edges; top-k merges by re-selecting. Shard A plus shard B equals one pass over A + B, so scaling out is map-reduce.
4. **Pin conventions and record provenance.** Activations follow TransformerLens's processed convention (folded LayerNorm, centered writing weights), matching HookedTransformer and therefore the pretrained SAEs, whichever backend computes them. Each state file records its row range, git commit, torch version and model.
5. **Exclude known artefacts explicitly.** Position 0 (BOS) is an attention sink whose residual norm is orders of magnitude above the rest, so it is dropped from all statistics rather than left to dominate them.
6. **Query with plain files.** Outputs are Parquet and npz, queried with DuckDB or pandas. There is no database server to run.

## Architecture

The pipeline is three commands over plain files: `tokenize` once, `run` per shard anywhere, `export` to merge. Two tools sit beside it and share its model backend and SAE loader: `sae_gate.py`, which checks a pretrained SAE against its release on this pipeline, and `ioi.py`, which runs the IOI prompts with head ablations and splits the logit difference into SAE features. The notebook reads all of their outputs and writes the results' figures.

![Atlas pipeline: tokenize, sharded run, export, query](figures/atlas_architecture.svg)

**Data model.** A *row* is one sequence the model processes independently: BOS plus 127 consecutive tokens from one document (511 with `--seq-len 512`); by default one randomly chosen window per document, from eight files of the corpus (`--dataset`: OpenWebText by default, FineWeb or Python code), with documents sampled across each whole file (random row groups) because the files are not shuffled. A *pointer* is `(row, pos)`, stored as the integer `row × seq_len + pos`; positions 1 onward are real tokens. Shards are row ranges, so a shard's state is self-describing and any set of non-overlapping shards merges into one atlas.

**Hook points (stage 1).** MLP `hook_post` in all 12 layers (3072 units each) and the residual stream at 13 points: `hook_resid_pre` for layers 0–11 plus the final `hook_resid_post`. The forward pass stops after the last hook, so no layer past it runs; `--hooks` selects a subset for deep scans. In stage 2 (`--sae`) an SAE encoder runs inside the same hooks: its features become atlas units at the chosen layers (`--sae-layers`), and `--sae-events` writes their nonzero activations as sparse `(row, pos, feature, value)` events, one Parquet file per layer and shard. `--positions` also keeps every statistic per position bucket.

**Model backends.** The default backend, `HFGPT2`, is plain Hugging Face `GPT2Model` (no LM head, so no unembed) with PyTorch forward hooks, exposed under TransformerLens hook names. Compatibility mode changes only one thing the atlas reads: everything written to the residual stream has its per-token mean over d_model removed, because centering the writing weights is linear. `HFGPT2` applies that one subtraction. `TLGPT2`, the TransformerLens bridge, is kept as the reference, and `tests/test_backends.py` holds the two equal at every hook point. TransformerLens stays the tool for circuit work (patching, per-head outputs, direct logit attribution), where it saves real effort.

## Stage 1 status

Stage 1 is built and validated on 1M tokens, with both backends. The code lives in `python/atlas.py` (CLI `python -m atlas`, `--backend hf|tl`), `scripts/run-atlas`, `tests/test_atlas.py` and `tests/test_backends.py`.

**Validation**

| Check | Result |
| --- | --- |
| Merge exactness (unit tests: moments, covariance, top-k, histogram, state round trip, tokenizer) | pass; split vs. whole agree to float32 rounding (~1e-7 relative). Now part of a 52-test suite ([validation](results-stage2.md#5-validation)) |
| HF backend vs. TransformerLens bridge, per hook (7 hook types × layers 0, 5, 11, BOS included) | 21 of 21 agree within 2e-5 of the activation scale; raw HF residuals fail without centering, as they should |
| Full 1M-token atlas, HF vs. TL backend | means, stds, quantiles agree to ~1e-5 relative; largest difference in any unit's max is 2e-4; 0.6% of top-k pointers swap among exact ties (identical left contexts) |
| Pointers reproduce: recompute stored top-k entries from `(row, pos)` | 10 of 10 match to 5 decimals, layers 0, 9 and 11 |
| Bottom-k for residual dimensions (added 2026-10-26) | top side of a rerun identical to the run without it; 12 of 12 sampled bottom-k entries recompute to 1e-6 relative; no change in speed |
| Sharded run, with bootstrap over shards and files, merged by `export` | 7,816 rows from 7,816 documents in 16 shards (2 in each of 8 OpenWebText files); total errors on unit means ~0.002 of their spread (~250k effective samples), effective dimension to ±0.5 |
| Residual stream behaves like known GPT-2 | variance grows ~5000× from layer 0 to the final residual; dim 480 carries 7.5% of the variance at layer 1, falling to 2% by layer 7; from layer 7 to 11 no basis dimension carries more than ~2.5%; at the final residual dims 496, 430 and 36 carry 11%, 8% and 8%; effective dimension peaks at ~170 in layers 7–9 and collapses to ~7 at the final residual |
| Neurons behave like known GPT-2 | rarely-firing neurons (firing rate < 1e-4) concentrate in layers 0–2: 62, 87, 21 |

**Measured cost (M4, MPS, 1M tokens)**

| Quantity | TL bridge | Plain HF (default) |
| --- | --- | --- |
| Forward pass with hooks on all 25 points | ~5,300 tok/s | ~9,600 tok/s |
| Full run, all accumulators | ~3,200 tok/s (before tuning) | ~5,150 tok/s after tuning (fused moments, int32 histograms, histograms on every 4th token) |
| Wall time for 1M tokens (run only) | ~5 min | ~3.2 min |
| State file per shard | 171 MB (histograms 97 MB, covariance 61 MB), independent of token count | same |

Tokenizing takes about 1 min and export 9 s with either backend.

**Known limitations**

- The accumulators still cost about as much as the forward pass (full run ~5,150 vs ~9,700 tok/s forward-only on an M4): top-k and histograms each read every MLP activation. Histograms use every 4th token, which shifts quantiles by a median 0.02 std at p99 (0.13 std for the worst 1% of units). Further speed belongs on CUDA.
- Exemplars now span ~8k documents (one window each), but in the 1M-token atlases rare units (firing rate < 1e-4) still have fewer than ~100 examples. A deep scan (`run --hooks`) fixes this per hook: layers 0–2 over 50M tokens gives the rare neurons there a median of 526 examples and bounds 89 silent neurons below 6×10⁻⁸ ([results](results-stage1.md#7-deep-scan-of-layers-02-rare-and-never-firing-neurons)). Later layers and other corpora are not scanned yet.
- The default atlas covers positions 1–127. A 512-token atlas with position buckets covers positions out to 511 ([results](results-stage1.md#8-position-and-context-length)); longer context is not seen.
- Rows start mid-document after a BOS, which GPT-2 never saw in training (its windows started mid-document without BOS).

## Stage 2 status

Stage 2 runs pretrained SAEs inside the same pipeline. The code is `--sae`, `--sae-layers` and `--sae-events` in `python/atlas.py`, the gate in `python/sae_gate.py` (`run-sae-gate`), and the IOI analysis in `python/ioi.py`; results are in [results-stage2.md](results-stage2.md).

- **Gate.** Bloom's `gpt2-small-res-jb` reproduces its release's L0, alive features, variance explained and CE loss at every layer; OpenAI's 32k and 128k TopK SAEs pass within their 64-token training context ([results](results-stage2.md#1-the-gate-into-stage-2)).
- **Feature atlas.** Bloom's features are atlas units at layers 6 and 9, with sparse per-token events at layer 9; single features match Neuronpedia's dashboards, and the decoder bias is the residual stream's mean ([results](results-stage2.md#2-the-atlas-in-the-sae-basis)).
- **IOI repair.** With the name movers ablated, the repair restores the clean run's LD-carrying features (cosine 0.86–0.94 in three dictionaries), mostly "say this name" features. The SAE error carries about half the logit difference with 24k and 32k dictionaries and a fifth to a third with 128k. The off-dictionary part is written by the circuit's own heads and is a small-norm, structured remainder (3% of the final residual's squared norm, a quarter to two fifths of the LD), missed because the logit direction carries little of the stream's norm, and spread over hundreds of inactive features rather than held by a few. These claims hold across ablation methods, prompt sets and templates ([results](results-stage2.md#3-the-ioi-repair-in-the-sae-basis)).
- **Still to do.** Layers 10–11 in the feature atlas, the corpus and position systematics in the feature basis, and feature matching across dictionaries.

## Roadmap

Stage 1 is finished at laptop scale: every stage 1 systematic in the table below is measured ([results](results-stage1.md)). The gate into stage 2 has passed, and stage 2 is in progress (see [Stage 2 status](#stage-2-status)). The gate into stage 3 asks for a question pretrained dictionaries cannot answer. The logit difference they miss in the IOI task is the first: it sits in a minor direction of the stream that a reconstruction loss under-weights, and it is spread over hundreds of the dictionary's directions rather than held by a few inactive features, so a better encoder would not recover it but a dictionary with the logit difference in its objective could ([results](results-stage2.md#3-the-ioi-repair-in-the-sae-basis)).

![Atlas roadmap: 4 stages, 3 gates](figures/atlas_roadmap.svg)

Each stage reuses the same tokenize → run → export pipeline and only adds accumulators or a change of basis inside the hooks. The first research payoff came in stage 2: under name-mover ablation, the backups and the released brake restore the clean run's SAE features at the final residual rather than build new ones.

## Systematics and validation plan

Atlas numbers are measurements conditional on a corpus, a convention and a dictionary; each of those is a systematic, reported as a spread next to the statistical error, as in ioi-hydra. The sizes measured so far are summarized in [results: systematic uncertainties](results-stage1.md#10-systematic-uncertainties).

| Source | How we vary it | Applies from |
| --- | --- | --- |
| Statistical (finite tokens) | bootstrap over shards, resampled within each file, corrected for the few shards per file (done: 16 shards in 8 files) | stage 1 |
| File to file, within a corpus | cluster bootstrap over files, the total error `*_err` (done: 8 OpenWebText files). The 50M-token deep scan bounds the term at about 0.0003 of a unit's spread, adding 2–6% to the error; an earlier two-file estimate overstated it 5–10× ([results](results-stage1.md#5-errors-statistical-and-file-to-file)) | stage 1 |
| Corpus | OpenWebText vs FineWeb vs Python code (done): typical per-unit shifts 12–15× the file-to-file floor for FineWeb and 77–117× for code; offset dimensions and rarely-firing neurons are the same on all three ([results](results-stage1.md#6-corpus-systematic-openwebtext-vs-fineweb-vs-python-code)) | stage 1 |
| Context length and position | 512-token windows with statistics per position bucket (done: `run --positions`). Position matters most in the first ~16 positions; later positions and long documents differ little ([results](results-stage1.md#8-position-and-context-length)). Covariance within each position is still open | stage 1 |
| Centering the residual stream | centered (TransformerLens processed convention) vs. raw HF residuals, the only part of weight processing the atlas sees (done: `--backend hf-raw`). Negligible for GPT-2 small: the all-ones direction holds at most 0.1% of the variance, and MLP activations are identical ([results](results-stage1.md#9-dependence-on-centering-the-residual-stream)) | stage 1 |
| SAE dictionary | Bloom 24k vs. OpenAI 32k / 128k on the same layer (done for the IOI question: the SAE error holds 46% / 46% / 26% of the logit difference, so what the dictionary misses depends on its size; with 128k features the features' LD is restored exactly and a smaller overshoot lands off-dictionary; robust across ablation methods and prompt sets, [results](results-stage2.md#3-the-ioi-repair-in-the-sae-basis)). Feature splitting and matching across dictionaries still to do | stage 2 |
| SAE reconstruction error | report the unexplained variance ("dark matter") beside every feature-level claim (done for the IOI question: the error's share of the logit difference, split into shrinkage, off-span and the components that write it, [results](results-stage2.md#3-the-ioi-repair-in-the-sae-basis)) | stage 2 |
| SAE seed | two trainings differing only in seed | stage 3 |

**Gates before trusting a pretrained SAE.** On our own pipeline, its variance explained, CE loss recovered, L0 and dead-feature fraction must match the release's reported numbers. A mismatch means a convention differs (hook name, LayerNorm folding, BOS handling), and the SAE's outputs would be quietly meaningless. Done for `gpt2-small-res-jb` (`python/sae_gate.py`, `run-sae-gate`): at every layer L0, alive features, variance explained and the CE increase match the release post's table, and our activations match TransformerLens's to 10⁻⁶ ([results](results-stage2.md#1-the-gate-into-stage-2)). OpenAI's `gpt2-small-resid-post-v5-32k` and `-128k` pass within their 64-token training context and fail beyond it. Compare with the release's own publication: SAELens's built-in reference values for this release are rough, and the post computes variance explained about the origin rather than the mean.

**Regression checks on every run.** Pointers must reproduce on recompute for a random sample of top-k entries, and residual outlier dimensions must stay stable across shards.

## Open questions

- [ ] **Who is the atlas for?** A personal research tool favours notebooks and DuckDB; a shared one needs a viewer (static HTML pages per unit, or uploads to Neuronpedia).
- [x] **Compute.** Decided: stay laptop-sized (1–2M tokens per run) and scale only per unit, with deep scans of single hooks. A CUDA machine is needed only if stage 2 covers many SAE layers.
- [ ] **Row layout.** Keep per-document windows with BOS, or switch to GPT-2's training layout (contiguous 1024-token windows without BOS) for stage 3 onward?
- [ ] **Sequence length.** 128 keeps attention cheap and matches IOI-length prompts; 512 or 1024 sees long-context behaviour at 4–8× the attention cost.
- [ ] **Which hook points next.** Attention outputs per head (`hook_z`) would tie the atlas directly to the head-level IOI analysis. `ioi.py` already reads each head's output at `attn.c_proj` for its 128 prompts; the atlas does not yet accumulate it over the corpus.
- [ ] **Naming.** "row" is the array axis; "seq" may read better in code and Parquet columns.
