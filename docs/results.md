# GPT-2 Activation Atlas — Initial Results

*Ryan Reece, 2026-10-26, updated 2026-10-27. Stage 1 (neuron basis) at laptop scale. Design:
[atlas-design.md](atlas-design.md).*

## Summary

A first look at what GPT-2 small does over 1M OpenWebText tokens, from the stage 1 atlas: statistics for all 36,864
MLP neurons and 9,984 residual-stream dimensions (13 hook points × 768), with bootstrap errors over 8 shards. The same
atlas on 1M tokens each of FineWeb and Python code measures how much of it depends on the corpus.

- **The residual stream grows and then collapses.** Its total variance grows about 4900× from the embedding to the
  final residual. Its effective dimension rises from 14 to about 170 in layers 7–9, then collapses to about 7 at the
  final residual, where the top principal component alone carries 35% of the variance.
- **A few residual dimensions are constant offsets.** Dimensions 64, 266, 326, 373, 447 and 481 sit many standard
  deviations away from zero on almost every token. The covariance view subtracts the mean, so it does not see them.
  They are the same dimensions on all three corpora.
- **Rarely-firing MLP neurons live in layers 0–2** (169 of the 182 neurons that fire on fewer than 1 in 10⁴ tokens),
  and they are mostly the same neurons on every corpus. 109 neurons never fire on any of the 3M tokens. Layer 1's
  median neuron fires on 29% of tokens, at least twice as often as any other layer's.
- **The corpus matters, and code matters most.** Against the difference between two OpenWebText files, FineWeb moves
  a typical unit 12–15× further and Python code 77–117× further. On code the residual stream's effective dimension
  stays near 25 at every layer, against up to 170 on web text.
- **Errors are small, but files differ slightly.** A unit's mean is known to about 0.002 of its spread. Two files of
  the same corpus differ a little more than statistical noise, so the bootstrap over shards understates the error
  from one file to the next.
- **Every stored exemplar reproduces.** Recomputing sampled top- and bottom-k pointers returns the stored values to
  1e-6 relative.

These are initial results: 128-token windows, three corpora and the neuron basis only. The caveats are listed at the
end.

## Setup

| | |
| --- | --- |
| Model | GPT-2 small, TransformerLens processed convention (folded LayerNorm, centered writing weights) |
| Data | 7,814 rows from 7,814 documents (one random 127-token window each), from two files of the corpus, with documents sampled across each whole file |
| Corpora | OpenWebText (default; files 0 and 1), FineWeb sample-10BT (files 0 and 1), codeparrot-clean Python (files 0 and 1); revisions pinned in each `manifest.json` |
| Tokens | 992,378 per unit; position 0 (BOS) excluded from all statistics |
| Hooks | MLP `hook_post` in layers 0–11; residual stream before each layer and after the last ("layer 12") |
| Errors | bootstrap over 8 shards (4 per file), resampled within each file |
| Compute | about 3.5 min per corpus on an M4 (MPS) for the run, 1–3 min to tokenize, 10 s to export |

Sections 1–5 are OpenWebText; section 6 compares the corpora.

To reproduce: `source setup.sh`, then `run-atlas` (OpenWebText into `results/atlas/`),
`run-atlas results/atlas-fineweb 1e6 fineweb`, `run-atlas results/atlas-code 1e6 code`, and `run-notebook`, which
runs `notebooks/atlas.ipynb` headless, in place, and writes every figure here into `docs/figures/`.

## 1. Residual-stream geometry

![Total variance, effective dimension and cumulative variance by layer](figures/results_resid_geometry.png)

| Layer | Total variance | Effective dimension | PCs for 90% of variance | Largest basis dimension (share) |
| --- | --- | --- | --- | --- |
| 0 | 11.6 ± 0.0 | 14.3 ± 0.0 | 411 | 479 (2%) |
| 1 | 1,033 ± 1 | 31.1 ± 0.0 | 314 | 480 (7.5%) |
| 4 | 2,156 ± 1 | 103.9 ± 0.1 | 424 | 480 (4%) |
| 8 | 5,866 ± 2 | 169.8 ± 0.3 | 500 | 447 (2%) |
| 9 | 8,114 ± 2 | 170.8 ± 0.2 | 507 | 447 (2%) |
| 11 | 19,089 ± 12 | 103.0 ± 0.3 | 510 | 496 (2.5%) |
| 12 (final) | 56,611 ± 165 | 6.7 ± 0.0 | 345 | 496 (10.5%) |

The effective dimension is the participation ratio (Σλ)²/Σλ² of the covariance eigenvalues. Errors are bootstrap
over shards. The notebook prints every layer.

- **Variance.** The first block multiplies the variance by 89 (layer 0 is the embedding alone). After that it grows
  steadily, by 55× from layer 1 to the final residual, with the last block alone contributing 3.0×.
- **Effective dimension.** It rises through the middle layers, where the variance is spread most evenly (the top PC
  carries only 4% at layer 8), then collapses in the last two blocks. At the final residual a handful of directions
  dominate, which is what `ln_final` and the unembedding then read.
- **Basis dimensions.** Dimension 480 stands out early (7.5% of the variance at layer 1, down to 2% by layer 7). From
  layer 7 to 11 no basis dimension carries more than 2.5%, so which one is largest there means little. At the final
  residual, dimensions 496, 430 and 36 carry 10.5%, 8% and 8%.

## 2. Offset dimensions in the residual stream

The covariance in §1 is taken about the mean, so it is blind to dimensions that are large but nearly constant. Ranking
residual dimensions by |mean| / std picks out a small, stable set:

| Dimension | Layer 1 | Layer 6 | Layer 11 | Layer 12 (final) |
| --- | --- | --- | --- | --- |
| 64 | −23.7 ± 2.0 | −27.2 ± 2.2 | −28.1 ± 2.2 | −118.7 ± 35.4 |
| 373 | −16.2 ± 4.0 | −21.3 ± 4.6 | −93.7 ± 18.0 | −142.9 ± 39.4 |
| 447 | 17.3 ± 4.4 | 32.0 ± 8.0 | 89.3 ± 17.8 | 127.4 ± 39.1 |
| 266 | 11.3 ± 2.5 | 11.7 ± 2.4 | 11.9 ± 2.7 | — |
| 326 | 9.3 ± 1.7 | 10.5 ± 2.4 | — | — |
| 481 | — | — | 92.7 ± 18.9 | 133.2 ± 39.0 |

Entries are mean ± standard deviation over tokens (not errors on the mean), and "—" means |mean| < 3 std at that
layer. In all, 78 (layer, dimension) pairs pass |mean| > 3 std: 15 small ones at layer 0, the six dimensions in the table
(481 only from the later layers), dimension 101 at layer 1 only, and dimensions 87 and 480 at the final residual only.

Dimensions 64, 266 and 326 are switched on by the first block and then barely change. Dimensions 373, 447 and 481 grow
steadily with depth. This resembles the outlier ("massive activation") dimensions reported for GPT-2 and other
transformers. §6 shows they are the same dimensions on FineWeb and code, so they belong to the model, not the data.
Whether they act as biases that later LayerNorms divide out, or carry information in their small variations, is a
question for stage 2.

## 3. MLP neurons

![Firing rate, peak activation and rarely-firing neurons by layer](figures/results_mlp_neurons.png)

A neuron "fires" when its post-GELU activation is above 0.

- **Firing rates.** Layer 1 is the outlier: its median neuron fires on 29% of tokens, against 6–14% in every other
  layer. From layer 4 on the median rises slowly, from about 9–10% to 14%.
- **Peak activations** climb through the last layers. The five highest-peak neurons are all in layers 10–11, led by
  L11/2378 (max 60, next highest 23).
- **Rarely-firing neurons** (rate < 1e-4, fewer than about 100 of 1M tokens) number 61, 87 and 21 in layers 0–2, and
  13 across all later layers. 124 neurons never fire at all in this sample, and 109 never fire on any of the three
  corpora (§6). With 3M tokens, "never" only bounds the rate below about 1 in 10⁶; deep scans of single units are the
  planned way to tell rare from dead.

## 4. Single units

The atlas keeps each unit's histogram and its top-20 exemplars as `(row, pos)` pointers, decoded here with the token
marked «like this». Residual dimensions are signed, so they also keep bottom-20.

**L11/2378, the highest-peak neuron.** It fires on 6.0% of tokens. Every top-10 context is the first newline of a
blank line «⏎»⏎ at the end of a paragraph or list item, often closing a quote. Above zero the histogram has two bumps,
a broad one near 10 and a sharper one near 35–40.

![Histogram of L11 neuron 2378](figures/results_neuron_l11_peak.png)

**L9/1000, a typical mid-depth neuron.** It fires on 7.9% of tokens and peaks at 2.9. Its top contexts mix
determiners and pronouns in set phrases ("of «my» choosing", "one of «a» number", "one-of-a«-»kind") with ages and
ordinals ("a woman in «her» mid thirties", "2«nd»"). Ten exemplars are not enough to name what it does. Its histogram
is in [figures/results_neuron_l9_1000.png](figures/results_neuron_l9_1000.png).

**Residual dimension 447 at layer 9, both extremes.** This is one of the offset dimensions from §2 (mean 49, std 12),
positive on 99.9% of tokens.

![Histogram of residual dimension 447 at layer 9](figures/results_resid_l9_447.png)

| Side | Value | Context |
| --- | --- | --- |
| top | 101.6 | …The FBI stated that «Polic»arpio became enraged… |
| top | 97.5 | …official policies or procedures for the use of «Theft»Track… |
| top | 97.2 | …Although «Chat»Secure's Facebook and… |
| top | 95.7 | What if «Spotify» made a Podcast App |
| bottom | −30.0 | …December 2016 November 2016 October «2016» September 2016 August 2016… |
| bottom | −28.9 | …November 2016 October 2016 September «2016» August 2016 July 2016… |
| bottom | −27.0 | …December 2017 November 2017 October 2017 September «2017» August 2017… |

The largest values fall on the first token of a capitalized name or an unfamiliar product word, often the first of
several BPE pieces. All ten most negative fall on the year in long "Month Year" archive lists. The bottom side only
exists because of the bottom-k added in this round: top-k alone shows half the picture for a signed unit.

## 5. Statistical errors and a within-corpus check

| Median bootstrap error | MLP neurons | Residual dimensions |
| --- | --- | --- |
| on the mean, in units of std | 0.0017 | 0.0019 |
| on the std, relative | 0.0044 | 0.0010 |
| on the firing rate, relative | 0.0053 | — |

An error of 0.002 std on a mean corresponds to roughly 250k effective samples out of 1M tokens: tokens within one
document, and documents within one shard, are correlated.

![Cross-file vs within-file differences](figures/results_corpus_check.png)

The plots compare each unit's difference between OpenWebText files 0 and 1 (solid) with its difference between two
halves of the same file (dashed, scaled by 1/√2 to match the data size). The files differ slightly *more* than the
halves do: the ratio is 1.12–1.14 in the median and 1.10–1.12 at the 99th percentile. A per-unit z-test on the shard
means agrees. The file-vs-file z-scores have variance 1.8–2.0, against 1.3–1.4 for a control split that mixes the two
files and 1.5 expected from noise alone. The files share no documents, and the difference is small (about 0.0015 of a
unit's spread, a twentieth of the FineWeb shift in §6), but it is real. The two files' effective dimensions agree
within 0.5 at every layer.

Two consequences:

- **Errors.** The bootstrap resamples shards within each file, so it covers statistical noise but not the variation
  from one file of the corpus to the next. For claims at the 0.002-std level, that file-to-file term should be added.
- **An earlier conclusion was wrong.** On the first version of this atlas, which read the first few thousand documents
  of each file, the files looked like pure statistical replicates. Most likely, consecutive documents are correlated,
  which inflated the shard-to-shard scatter and hid the difference. Documents are now sampled across each whole file (random row
  groups), and the difference shows.

## 6. Corpus systematic: OpenWebText vs FineWeb vs Python code

The same pipeline, sampling and 1M-token budget, run on two more corpora: FineWeb (sample-10BT; filtered Common Crawl,
web text like OpenWebText but a different selection) and Python code (codeparrot-clean, from GitHub).

![Per-unit shifts of each corpus against OpenWebText](figures/results_corpus_units.png)

Each curve is the distribution, over units, of a unit's shift from OpenWebText: |Δ mean| in units of its spread on
OpenWebText, or for firing rates the |log₁₀ ratio|. The dashed curve is the within-OpenWebText floor from §5 (file 0
vs file 1, scaled to full-size samples).

| Median shift | FineWeb | Python code | Within-OpenWebText floor |
| --- | --- | --- | --- |
| MLP neurons: mean, in units of std | 0.028 (12×) | 0.172 (77×) | 0.0022 |
| Residual dims: mean, in units of std | 0.037 (14×) | 0.263 (102×) | 0.0026 |
| MLP neurons: \|log₁₀ firing-rate ratio\| | 0.044 (15×) | 0.335 (117×) | 0.0029 |

![Per-layer summaries by corpus](figures/results_corpus_layers.png)

| Layer | Effective dimension: OpenWebText | FineWeb | Python code |
| --- | --- | --- | --- |
| 1 | 31.1 | 31.4 | 14.2 |
| 4 | 103.9 | 105.7 | 19.1 |
| 9 | 170.8 | 183.4 | 27.1 |
| 11 | 103.0 | 120.3 | 24.6 |
| 12 (final) | 6.7 | 12.1 | 2.5 |

Bootstrap errors are at most ±0.3.

- **FineWeb** follows OpenWebText closely in shape but sits clearly above the floor. Its effective dimension is higher
  from about layer 6 on, and its total variance at the final residual is 24% lower. Firing rates per layer are almost
  identical.
- **Code** differs from the first block on:
  - Its effective dimension stays near 25 at every layer, and at the final residual the top PC carries 61% of the
    variance (35% for OpenWebText).
  - Its neurons fire less: the median is 5–7% after layer 1, and 21% in layer 1.
  - Its total variance at the final residual is 42% higher.

  A likely cause is code's token statistics. 37% of its tokens are pure whitespace (indentation), against 2–4% for web
  text, and its ten most common token types cover half of all tokens, against about a quarter. That would let a few
  directions carry much of the variance, but the atlas alone does not prove it.
- **What does not depend on the corpus.**
  - The same residual dimensions are offsets on all three corpora, with similar means through layer 11: FineWeb
    within 4% of OpenWebText and code within 4–17%, except dimension 326 (up to 14% and 70% larger).
  - The rarely-firing neurons are mostly the same neurons: 168 are rare on every corpus, and 109 never fire on any.
  - Layer 1 is the high-firing layer on all three.
- **Code-specific neurons.** A few early neurons are nearly silent on web text and fire on code. L0/2469 fires on
  0.04% of OpenWebText tokens and 30% of code tokens, with its top code context on indentation; L1/1323 goes from 0.06%
  to 13%, with its top context on a run of escaped tabs (`"\t\t…"`) in a string. In the other direction, L4/961 fires on 9.4% of OpenWebText tokens and 0.01% of code tokens.

## 7. Validation

| Check | Result |
| --- | --- |
| Pointers reproduce: 12 random top- and bottom-k entries recomputed from `(row, pos)` | all match, worst relative difference 1.4e-6 |
| Exact merging and sampling: unit tests for every accumulator (split vs. whole, including bottom-k) and for sampling across a file | 40 of 40 pass |
| Plain HF backend vs. TransformerLens bridge | agree within 2e-5 of the activation scale at every hook (see design doc) |

## Caveats

- **Two files per corpus.** Files of one corpus differ slightly (§5), and with two files per corpus the size of that
  file-to-file term is only roughly known.
- **Short context.** Only positions 1–127 of 128-token windows are covered. Rows start mid-document after a BOS, which
  GPT-2 never saw in training.
- **The neuron basis.** Single neurons are often polysemantic, so a unit's top-10 contexts (as for L9/1000) may not
  name what it does. SAE bases are stage 2.
- **Thin tails.** Rare units have fewer than about 100 examples in 1M tokens. Histograms (and so quantiles) use every
  4th token.

## Next

From the [roadmap](atlas-design.md#roadmap):

- **Deep scans of single units.** Confirm whether the 109 neurons that never fire are dead, and get more exemplars for
  rare and code-specific units.
- **The file-to-file term.** More files per corpus, to size it and fold it into the errors.
- **Then stage 2:** pretrained SAE bases, behind the gate that each SAE reproduces its release's reported metrics on
  this pipeline.
