# GPT-2 Activation Atlas — Initial Results

*Ryan Reece, 2026-10-26. Stage 1 (neuron basis) at laptop scale. Design: [atlas-design.md](atlas-design.md).*

## Summary

A first look at what GPT-2 small does over 1M OpenWebText tokens, from the stage 1 atlas: statistics for all 36,864
MLP neurons and 9,984 residual-stream dimensions (13 hook points × 768), with bootstrap errors over 8 shards.

- **The residual stream grows and then collapses.** Its total variance grows about 4800× from the embedding to the
  final residual. Its effective dimension rises from 14 to about 170 in layers 7–9, then collapses to about 7 at the
  final residual, where the top principal component alone carries 34% of the variance.
- **A few residual dimensions are constant offsets.** Dimensions 64, 266, 326, 373 and 447 sit many standard
  deviations away from zero on almost every token, from layer 1 onward. The covariance view subtracts the mean, so it
  does not see them.
- **Rarely-firing MLP neurons live in layers 0–2** (170 of the 183 neurons that fire on fewer than 1 in 10⁴ tokens).
  Layer 1's median neuron fires on 29% of tokens, at least twice as often as any other layer's.
- **Errors are small and the files agree.** A unit's mean is known to about 0.002 of its spread. The two OpenWebText
  files differ no more than two halves of one file do, so the corpus systematic still needs a different corpus.
- **Every stored exemplar reproduces.** Recomputing sampled top- and bottom-k pointers returns the stored values to
  1e-6 relative.

These are initial results: one corpus, 128-token windows and the neuron basis only. The caveats are listed at the end.

## Setup

| | |
| --- | --- |
| Model | GPT-2 small, TransformerLens processed convention (folded LayerNorm, centered writing weights) |
| Data | 7,814 rows from 7,814 documents (one random 127-token window each), OpenWebText files 0 and 1 |
| Tokens | 992,378 per unit; position 0 (BOS) excluded from all statistics |
| Hooks | MLP `hook_post` in layers 0–11; residual stream before each layer and after the last ("layer 12") |
| Errors | bootstrap over 8 shards (4 per file), resampled within each file |
| Compute | about 3.5 min on an M4 (MPS) for the run, 9 s to export |

To reproduce: `source setup.sh`, then `run-atlas` (statistics into `results/atlas/`) and `run-notebook`, which runs
`notebooks/atlas.ipynb` headless and writes every figure here into `docs/figures/`.

## 1. Residual-stream geometry

![Total variance, effective dimension and cumulative variance by layer](figures/results_resid_geometry.png)

| Layer | Total variance | Effective dimension | PCs for 90% of variance | Largest basis dimension (share) |
| --- | --- | --- | --- | --- |
| 0 | 11.6 ± 0.0 | 14.3 ± 0.0 | 410 | 479 (2%) |
| 1 | 1,032 ± 1 | 31.1 ± 0.0 | 314 | 480 (7.5%) |
| 4 | 2,155 ± 1 | 103.8 ± 0.2 | 424 | 480 (4%) |
| 8 | 5,858 ± 5 | 170.5 ± 0.3 | 500 | 447 (2%) |
| 9 | 8,106 ± 6 | 171.9 ± 0.3 | 507 | 447 (2%) |
| 11 | 19,060 ± 16 | 103.9 ± 0.4 | 509 | 496 (2.5%) |
| 12 (final) | 56,020 ± 174 | 6.8 ± 0.0 | 347 | 496 (11%) |

The effective dimension is the participation ratio (Σλ)²/Σλ² of the covariance eigenvalues. Errors are bootstrap
over shards. The notebook prints every layer.

- **Variance.** The first block multiplies the variance by 89 (layer 0 is the embedding alone). After that it grows
  steadily, by 54× from layer 1 to the final residual, with the last block alone contributing 2.9×.
- **Effective dimension.** It rises through the middle layers, where the variance is spread most evenly (the top PC
  carries only 4% at layer 8), then collapses in the last two blocks. At the final residual a handful of directions
  dominate, which is what `ln_final` and the unembedding then read.
- **Basis dimensions.** Dimension 480 stands out early (7.5% of the variance at layer 1, down to 2% by layer 7). From
  layer 7 to 11 no basis dimension carries more than 2.5%, so which one is largest there means little. At the final
  residual, dimensions 496, 430 and 36 carry 11%, 8% and 8%.

## 2. Offset dimensions in the residual stream

The covariance in §1 is taken about the mean, so it is blind to dimensions that are large but nearly constant. Ranking
residual dimensions by |mean| / std picks out a small, stable set:

| Dimension | Layer 1 | Layer 5 | Layer 9 | Layer 11 | Layer 12 (final) |
| --- | --- | --- | --- | --- | --- |
| 64 | −23.7 ± 2.0 | −26.8 ± 2.2 | −27.7 ± 2.1 | −28.1 ± 2.2 | −118.9 ± 35.0 |
| 373 | −16.2 ± 4.0 | −20.1 ± 4.1 | −34.5 ± 8.3 | −93.9 ± 17.9 | −143.1 ± 38.9 |
| 447 | 17.4 ± 4.4 | 29.0 ± 7.0 | 49.3 ± 12.0 | 89.4 ± 17.6 | 127.5 ± 38.6 |
| 266 | 11.3 ± 2.5 | 11.6 ± 2.4 | 11.8 ± 2.5 | 11.9 ± 2.7 | — |
| 326 | 9.3 ± 1.7 | 10.4 ± 2.3 | 11.1 ± 3.2 | — | — |
| 481 | — | — | 37.1 ± 11.5 | 92.8 ± 18.8 | 133.3 ± 38.5 |

Entries are mean ± standard deviation over tokens (not errors on the mean), and "—" means |mean| < 3 std at that
layer. In all, 79 (layer, dimension) pairs pass |mean| > 3 std: 15 small ones at layer 0, these few dimensions from
layer 1 on, and dimensions 87 and 480 at the final residual only.

Dimensions 64, 266 and 326 are switched on by the first block and then barely change. Dimensions 373, 447 and 481 grow
steadily with depth. This resembles the outlier ("massive activation") dimensions reported for GPT-2 and other
transformers. Whether these offsets act as biases that later LayerNorms divide out, or carry information in their
small variations, is a question for stage 2.

## 3. MLP neurons

![Firing rate, peak activation and rarely-firing neurons by layer](figures/results_mlp_neurons.png)

A neuron "fires" when its post-GELU activation is above 0.

- **Firing rates.** Layer 1 is the outlier: its median neuron fires on 29% of tokens, against 6–14% in every other
  layer. From layer 4 on the median rises slowly, from about 9–10% to 14%.
- **Peak activations** climb through the last layers. The five highest-peak neurons are all in layers 10–11, led by
  L11/2378 (max 58, next highest 24).
- **Rarely-firing neurons** (rate < 1e-4, fewer than about 100 of 1M tokens) number 62, 87 and 21 in layers 0–2, and
  13 across all later layers. 116 neurons never fire at all in this sample. With 1M tokens, "never" only bounds the rate
  below a few in 10⁶; deep scans of single units are the planned way to tell rare from dead.

## 4. Single units

The atlas keeps each unit's histogram and its top-20 exemplars as `(row, pos)` pointers, decoded here with the token
marked «like this». Residual dimensions are signed, so they also keep bottom-20.

**L11/2378, the highest-peak neuron.** It fires on 5.8% of tokens. Every top-10 context is the first newline of a
blank line «⏎»⏎ at the end of a list item, a heading or a paragraph (street lists, transcripts, quotes). Above zero the
histogram has two bumps, a broad one near 10 and a sharper one near 35–40.

![Histogram of L11 neuron 2378](figures/results_neuron_l11_peak.png)

**L9/1000, a typical mid-depth neuron.** It fires on 7.8% of tokens and peaks at 3.8. Its top contexts mix ordinals
and positions ("3«rd» R", "«second» from left") with age phrases ("now in «their» 50s", "in «his» 20s"). Ten exemplars
are not enough to name what it does. Its histogram is in
[figures/results_neuron_l9_1000.png](figures/results_neuron_l9_1000.png).

**Residual dimension 447 at layer 9, both extremes.** This is one of the offset dimensions from §2 (mean 49, std 12),
positive on 99.9% of tokens.

![Histogram of residual dimension 447 at layer 9](figures/results_resid_l9_447.png)

| Side | Value | Context |
| --- | --- | --- |
| top | 96.9 | …you could argue that «SOM»A is actually more… |
| top | 96.5 | Statistically, the «Canucks» have the second-… |
| top | 95.6 | In short, «Shark»Linux will automatically update… |
| top | 93.9 | What is «Loki»P? |
| bottom | −23.4 | …Nov 2010 (14) Oct 2010 (12«)» Sep 2010 (10… |
| bottom | −23.1 | …Oct 2010 (12) Sep 2010 (10«)» Aug 2010 (9… |
| bottom | −21.8 | …Read or Share this story: http://tspne.ws«/»2nnEo |
| bottom | −20.5 | …who asked not to «be» identified because… |

The largest values fall on the first token of a capitalized name or a new, unfamiliar word, often one split into
several BPE pieces. The most negative fall inside long, repetitive archive lists such as "Month Year (count)" and URL
paths. The bottom side only exists because of the bottom-k added in this round: top-k alone shows half the picture for
a signed unit.

## 5. Statistical errors and the corpus check

| Median bootstrap error | MLP neurons | Residual dimensions |
| --- | --- | --- |
| on the mean, in units of std | 0.0019 | 0.0022 |
| on the std, relative | 0.0047 | 0.0010 |
| on the firing rate, relative | 0.0060 | — |

An error of 0.002 std on a mean corresponds to roughly 250k effective samples out of 1M tokens: tokens within one
document, and documents within one shard, are correlated.

![Cross-file vs within-file differences](figures/results_corpus_check.png)

The plots compare each unit's difference between OpenWebText files 0 and 1 (solid) with its difference between two
halves of the same file (dashed, scaled by 1/√2 to match the data size). The curves lie on top of each other. The ratio
of medians is 0.95–1.00 and the ratio of 99th percentiles is about 0.9. The effective dimension of each file agrees
with the other's to within about 1 at every layer. A separate check found no shared documents between the files and
per-unit z-scores consistent with pure noise. The two files are statistical replicates (different documents, same
distribution), so a real corpus systematic needs FineWeb or code.

## 6. Validation

| Check | Result |
| --- | --- |
| Pointers reproduce: 12 random top- and bottom-k entries recomputed from `(row, pos)` | all match, worst relative difference 9e-7 |
| Exact merging: unit tests for every accumulator, split vs. whole, including bottom-k | 39 of 39 pass |
| Plain HF backend vs. TransformerLens bridge | agree within 2e-5 of the activation scale at every hook (see design doc) |

## Caveats

- **One corpus.** Everything is conditional on OpenWebText, and the two files are replicates of each other.
- **Short context.** Only positions 1–127 of 128-token windows are covered. Rows start mid-document after a BOS, which
  GPT-2 never saw in training.
- **The neuron basis.** Single neurons are often polysemantic, so a unit's top-10 contexts (as for L9/1000) may not
  name what it does. SAE bases are stage 2.
- **Thin tails.** Rare units have fewer than about 100 examples in 1M tokens. Histograms (and so quantiles) use every
  4th token.

## Next

From the [roadmap](atlas-design.md#roadmap):

- **Deep scans of single units.** Tell rare neurons from dead ones, and get more exemplars for the units above.
- **A real corpus systematic:** FineWeb or code against OpenWebText.
- **Then stage 2:** pretrained SAE bases, behind the gate that each SAE reproduces its release's reported metrics on
  this pipeline.
