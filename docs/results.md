# GPT-2 Activation Atlas — Initial Results

*Ryan Reece, 2026-10-26, updated 2026-10-27. Stage 1 (neuron basis) at laptop scale. Design:
[atlas-design.md](atlas-design.md).*

## Contents

- [Summary](#summary)
- [Setup](#setup)
- [1. Residual-stream geometry](#1-residual-stream-geometry)
- [2. Offset dimensions in the residual stream](#2-offset-dimensions-in-the-residual-stream)
- [3. MLP neurons](#3-mlp-neurons)
- [4. Single units](#4-single-units)
- [5. Errors: statistical and file to file](#5-errors-statistical-and-file-to-file)
- [6. Corpus systematic: OpenWebText vs FineWeb vs Python code](#6-corpus-systematic-openwebtext-vs-fineweb-vs-python-code)
- [7. Deep scan of layers 0–2: rare and never-firing neurons](#7-deep-scan-of-layers-02-rare-and-never-firing-neurons)
- [8. Position and context length](#8-position-and-context-length)
- [9. Dependence on centering the residual stream](#9-dependence-on-centering-the-residual-stream)
- [10. Systematic uncertainties](#10-systematic-uncertainties)
- [11. The gate into stage 2](#11-the-gate-into-stage-2)
- [12. Stage 2: the atlas in the SAE basis](#12-stage-2-the-atlas-in-the-sae-basis)
- [13. Stage 2: the IOI repair in the SAE basis](#13-stage-2-the-ioi-repair-in-the-sae-basis)
- [14. Validation](#14-validation)
- [Caveats](#caveats)
- [Next](#next)

## Summary

A first look at what GPT-2 small does over 1M OpenWebText tokens, from the stage 1 atlas: statistics for all 36,864
MLP neurons and 9,984 residual-stream dimensions (13 hook points × 768), with bootstrap errors over 16 shards in 8
files. The same atlas on 1M tokens each of FineWeb and Python code measures how much of it depends on the corpus, a
50M-token deep scan of layers 0–2 separates rare neurons from silent ones, and a 512-token atlas measures how the
statistics depend on position in the context. Running the same tokens without centering the residual stream checks
the last processing convention.

- **The residual stream grows and then collapses.** Its total variance grows about 4800× from the embedding to the
  final residual. Its effective dimension rises from 14 to about 170 in layers 7–9, then collapses to about 7 at the
  final residual, where the top principal component alone carries 34% of the variance.
- **A few residual dimensions are constant offsets.** Dimensions 64, 266, 326, 373, 447 and 481 sit many standard
  deviations away from zero on almost every token. The covariance view subtracts the mean, so it does not see them.
  They are the same dimensions on all three corpora.
- **Rarely-firing MLP neurons live in layers 0–2** (170 of the 183 neurons that fire on fewer than 1 in 10⁴ tokens),
  and they are mostly the same neurons on every corpus. Layer 1's median neuron fires on 29% of tokens, at least twice
  as often as any other layer's.
- **89 neurons in layers 0–2 never fire in 50M tokens.** The deep scan checked the 100 there that never fired on
  3 × 1M tokens: 11 are merely rare, and 89 stay silent (rate below 6×10⁻⁸). About a third of those stay well below
  threshold, the closest thing to dead neurons a sample can show; another third come within 0.5 of firing.
- **The corpus matters, and code matters most.** Against the difference between two halves of OpenWebText, FineWeb
  moves a typical unit 14–18× further and Python code 85–133× further. On code the residual stream's effective
  dimension stays near 25 at every layer, against up to 170 on web text.
- **Errors are small, and files of one corpus are near replicates.** A unit's mean is known to about 0.002 of its
  spread. The errors now include the variation from one file of the corpus to the next, which the deep scan bounds at
  about 0.0003 of a unit's spread; it adds only 2–6% to the error.
- **Position matters most at the start of the context.** Layer 1's high firing rate builds up over the first ~16
  positions (5% at position 1, 29% from 16 on), and the offset dimensions are largest at position 1 and decay with
  position. Beyond about position 16, statistics change little out to 511, and long documents look like the rest.
- **Centering the residual stream changes almost nothing.** Raw (uncentered) residuals differ from the TransformerLens
  convention only along the all-ones direction, which carries at most 0.1% of the variance; MLP neurons are identical.
- **The gate into stage 2 passes.** Bloom's pretrained residual SAEs reproduce their release's L0, alive features,
  variance explained and CE loss on this pipeline, and our activations match TransformerLens's to 10⁻⁶.
- **Stage 2 has begun.** At layers 6 and 9 the SAE features are now atlas units, with sparse per-token events at layer 9.
  Single features match Neuronpedia's dashboards, and the SAE's decoder bias is the residual stream's mean.
- **The IOI repair restores the clean features.** With the name movers ablated, the logit difference at the final
  residual is carried by the same SAE features as in the clean run (cosine 0.86–0.94 in three dictionaries). With
  the 24k and 32k dictionaries about half of it sits in the SAE's error and shifts there under the repair; with a 128k
  dictionary only a quarter does, and the restoration is complete. The answer is carried mostly by "say this name"
  features, about four per name, and the off-dictionary rest is written by the circuit's own heads (§13).
- **Every stored exemplar reproduces.** Recomputing sampled top- and bottom-k pointers returns the stored values to
  2e-6 relative.

These are initial results: up to 512-token windows, three corpora and the neuron basis only. The caveats are listed at the
end.

## Setup

| | |
| --- | --- |
| Model | GPT-2 small, TransformerLens processed convention (folded LayerNorm, centered writing weights) |
| Data | 1M tokens per corpus, one random 127-token window per document, with documents sampled across each whole file |
| Corpora | OpenWebText (default): 7,816 rows, 977 from each of files 0–7. FineWeb sample-10BT and codeparrot-clean (Python): 7,814 rows from files 0 and 1. Revisions pinned in each `manifest.json` |
| Tokens | 992,632 per unit for OpenWebText; position 0 (BOS) excluded from all statistics |
| Hooks | MLP `hook_post` in layers 0–11; residual stream before each layer and after the last ("layer 12") |
| Errors | bootstrap over 16 shards in 8 files: resampling files (total error) or shards within files (statistical), see §5 |
| Compute | about 4 min per corpus on an M4 (MPS) for the run, 1–3 min to tokenize, about 3.5 min to export (two bootstraps over 16 shards) |

Sections 1–5 are OpenWebText; section 6 compares the corpora; section 7 is a 50M-token deep scan of layers 0–2;
section 8 is position and context length, from a 512-token atlas; section 9 is the dependence on centering the
residual stream.

To reproduce: `source setup.sh`, then `run-atlas` (OpenWebText into `results/atlas/`),
`run-atlas results/atlas-fineweb 1e6 fineweb`, `run-atlas results/atlas-code 1e6 code`, the runs in §7–§9, and
`run-notebook`, which runs `notebooks/atlas.ipynb` headless, in place, and writes every figure here into
`docs/figures/`. The FineWeb and code atlases here were built with 2 files, before the default became 8.

## 1. Residual-stream geometry

![Total variance, effective dimension and cumulative variance by layer](figures/results_resid_geometry.png)

| Layer | Total variance | Effective dimension | PCs for 90% of variance | Largest basis dimension (share) |
| --- | --- | --- | --- | --- |
| 0 | 11.6 ± 0.0 | 14.3 ± 0.0 | 411 | 479 (2%) |
| 1 | 1,033 ± 1 | 31.2 ± 0.0 | 314 | 480 (7.5%) |
| 4 | 2,157 ± 1 | 104.0 ± 0.1 | 424 | 480 (4%) |
| 8 | 5,861 ± 4 | 170.7 ± 0.3 | 500 | 447 (2%) |
| 9 | 8,109 ± 7 | 171.8 ± 0.3 | 507 | 447 (2%) |
| 11 | 19,070 ± 21 | 104.1 ± 0.5 | 510 | 496 (2.5%) |
| 12 (final) | 56,097 ± 199 | 6.8 ± 0.0 | 348 | 496 (10.5%) |

The effective dimension is the participation ratio (Σλ)²/Σλ² of the covariance eigenvalues. Errors are the total
bootstrap error (§5). The notebook prints every layer.

- **Variance.** The first block multiplies the variance by 89 (layer 0 is the embedding alone). After that it grows
  steadily, by 54× from layer 1 to the final residual, with the last block alone contributing 2.9×.
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
| 64 | −23.7 ± 2.0 | −27.2 ± 2.2 | −28.1 ± 2.2 | −118.9 ± 35.2 |
| 373 | −16.2 ± 4.0 | −21.3 ± 4.6 | −93.8 ± 17.9 | −143.0 ± 39.1 |
| 447 | 17.3 ± 4.4 | 32.0 ± 8.0 | 89.4 ± 17.6 | 127.5 ± 38.8 |
| 266 | 11.3 ± 2.5 | 11.7 ± 2.4 | 11.9 ± 2.7 | — |
| 326 | 9.3 ± 1.7 | 10.5 ± 2.4 | — | — |
| 481 | — | — | 92.8 ± 18.8 | 133.3 ± 38.7 |

Entries are mean ± standard deviation over tokens (not errors on the mean), and "—" means |mean| < 3 std at that
layer. In all, 79 (layer, dimension) pairs pass |mean| > 3 std: 16 small ones at layer 0, the six dimensions in the
table (481 only from the later layers), dimension 101 at layer 1 only, and dimensions 87 and 480 at the final residual
only.

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
  L11/2378 (max 57, next highest 24).
- **Rarely-firing neurons** (rate < 1e-4, fewer than about 100 of 1M tokens) number 62, 87 and 21 in layers 0–2, and
  13 across all later layers. 122 neurons never fire at all in this sample, and 110 never fire on any of the three
  corpora (§6). With 3M tokens, "never" only bounds the rate below about 1 in 10⁶; §7 scans layers 0–2 over 50M
  tokens to tell rare from silent.

## 4. Single units

The atlas keeps each unit's histogram and its top-20 exemplars as `(row, pos)` pointers, decoded here with the token
marked «like this». Residual dimensions are signed, so they also keep bottom-20.

**L11/2378, the highest-peak neuron.** It fires on 5.8% of tokens. Every top-10 context is the first newline of a
blank line «⏎»⏎ at the end of a paragraph or list item, usually closing a quote. Above zero the histogram has two
bumps, a broad one near 10 and a sharper one near 35–40.

![Histogram of L11 neuron 2378](figures/results_neuron_l11_peak.png)

**L9/1000, a typical mid-depth neuron.** It fires on 7.9% of tokens and peaks at 3.2. Its top contexts are mostly
determiners and pronouns in set phrases: "one of «a» number", "the first of «a» string of attacks", "of «their»
choice", "on «the» recommendation". Ten exemplars are not enough to name what it does. Its histogram is in
[figures/results_neuron_l9_1000.png](figures/results_neuron_l9_1000.png).

**Residual dimension 447 at layer 9, both extremes.** This is one of the offset dimensions from §2 (mean 49, std 12),
positive on 99.9% of tokens.

![Histogram of residual dimension 447 at layer 9](figures/results_resid_l9_447.png)

| Side | Value | Context |
| --- | --- | --- |
| top | 104.2 | …In nutshell, «Su»ave complements Azure… |
| top | 100.5 | …By default, «Retro»fit does the JSON… |
| top | 99.2 | …This probably means «Hus»k will combine its… |
| top | 98.7 | …returning to nostalgic locations. «Cron»o snickered… |
| bottom | −29.8 | …get three actions in your «inbox» every week. |
| bottom | −22.7 | …Absorb 20 dragon souls⏎⏎21.1%⏎⏎«Rare» 30.43% |
| bottom | −21.1 | …pic.twitter.com«/»w3tp4… |
| bottom | −19.1 | …You must select a newsletter to subscribe to. «Sign» Up… |

The largest values fall on the first token of a capitalized name or an unfamiliar product word, often the first of
several BPE pieces. The most negative fall in boilerplate and list text: newsletter sign-up lines, URL paths, a stats
table. The bottom side only exists because of the bottom-k added in this round: top-k alone shows half the picture for
a signed unit.

## 5. Errors: statistical and file to file

`export` gives two bootstrap errors on every unit's statistics and on each layer's total variance and effective
dimension:

- **`*_err_stat`**, the statistical error, resamples the 16 shards within each of the 8 files.
- **`*_err`**, the total error, resamples whole files instead (a cluster bootstrap), so it also covers the variation
  from one file of the corpus to the next.

Both are corrected for the few resampling units. A bootstrap over G units underestimates the variance by (G − 1)/G,
which with 2 shards per file would make the statistical error √2 too small.

| Median error | MLP neurons: total / stat | Residual dims: total / stat |
| --- | --- | --- |
| on the mean, in units of std | 0.0021 / 0.0021 | 0.0025 / 0.0024 |
| on the std, relative | 0.0054 / 0.0053 | 0.0012 / 0.0011 |
| on the firing rate, relative | 0.0068 / 0.0065 | — |

An error of 0.002 std on a mean corresponds to roughly 250k effective samples out of 1M tokens: tokens within one
document, and documents within one shard, are correlated.

![File-to-file variance vs noise, and total vs statistical error](figures/results_corpus_check.png)

**The file-to-file term is real but small.** For each unit, the variance of its 8 file means can be compared with the
variance expected from noise alone, estimated from the scatter of each file's 2 shards; with no file effect their
ratio follows F(7, 8), median 0.99.

- **In the 1M atlas** the median ratio is 1.06–1.08, but each file holds under 1k documents, so a file mean carries
  noise of about 0.005 of a unit's spread and the term is poorly resolved. All 36k units also share the same 8 files,
  so they are not independent tests.
- **The 50M-token deep scan (§7)** reads about half of the documents in the same 8 files, so its file means are close
  to the files' true means. There the file-to-file std is about 0.0003 of a unit's spread for means (median ratio
  1.12) and 0.0001 for firing rates (1.03).
- **Effect on the errors.** The total error exceeds the statistical error by 2–6% at both 1M and 50M tokens. At a
  much larger scale per file the file term would grow in relative weight, which `*_err` would then capture.

**Two corrections to earlier versions of these results.**

- **Errors were about 13% too small.** Earlier errors used 4 shards per file without the small-sample correction.
- **The file difference was overstated.** An earlier version reported that OpenWebText files 0 and 1 differ by about
  0.0015 of a unit's spread. That came from a single pair of files, one realization shared by every unit, and the
  deep scan shows the true file-to-file term is 5–10× smaller. The files are statistical replicates to this
  precision.

## 6. Corpus systematic: OpenWebText vs FineWeb vs Python code

The same pipeline, sampling and 1M-token budget, run on two more corpora: FineWeb (sample-10BT; filtered Common Crawl,
web text like OpenWebText but a different selection) and Python code (codeparrot-clean, from GitHub).

![Per-unit shifts of each corpus against OpenWebText](figures/results_corpus_units.png)

Each curve is the distribution, over units, of a unit's shift from OpenWebText: |Δ mean| in units of its spread on
OpenWebText, or for firing rates the |log₁₀ ratio|. The dashed curve is a within-OpenWebText floor: the same shift
between files 0–3 and files 4–7, scaled by 1/√2 to full-size samples.

| Median shift | FineWeb | Python code | Within-OpenWebText floor |
| --- | --- | --- | --- |
| MLP neurons: mean, in units of std | 0.028 (14×) | 0.172 (85×) | 0.0020 |
| Residual dims: mean, in units of std | 0.037 (16×) | 0.263 (113×) | 0.0023 |
| MLP neurons: \|log₁₀ firing-rate ratio\| | 0.045 (18×) | 0.335 (133×) | 0.0025 |

![Per-layer summaries by corpus](figures/results_corpus_layers.png)

| Layer | Effective dimension: OpenWebText | FineWeb | Python code |
| --- | --- | --- | --- |
| 1 | 31.2 | 31.4 | 14.2 |
| 4 | 104.0 | 105.7 | 19.1 |
| 9 | 171.8 | 183.4 | 27.1 |
| 11 | 104.1 | 120.3 | 24.6 |
| 12 (final) | 6.8 | 12.1 | 2.5 |

Bootstrap errors are at most ±0.5.

- **FineWeb** follows OpenWebText closely in shape but sits clearly above the floor. Its effective dimension is higher
  from about layer 6 on, and its total variance at the final residual is 23% lower. Firing rates per layer are almost
  identical.
- **Code** differs from the first block on:
  - Its effective dimension stays near 25 at every layer, and at the final residual the top PC carries 61% of the
    variance (34% for OpenWebText).
  - Its neurons fire less: the median is 5–7% after layer 1, and 21% in layer 1.
  - Its total variance at the final residual is 43% higher.

  A likely cause is code's token statistics. 37% of its tokens are pure whitespace (indentation), against 2–4% for web
  text, and its ten most common token types cover half of all tokens, against about a quarter. That would let a few
  directions carry much of the variance, but the atlas alone does not prove it.
- **What does not depend on the corpus.**
  - The same residual dimensions are offsets on all three corpora, with similar means through layer 11: FineWeb
    within 4% of OpenWebText and code within 4–17%, except dimension 326 (up to 14% and 69% larger).
  - The rarely-firing neurons are mostly the same neurons: 168 are rare on every corpus, and 110 never fire on any.
  - Layer 1 is the high-firing layer on all three.
- **Code-specific neurons.** A few early neurons are nearly silent on web text and fire on code. L0/2469 fires on
  0.04% of OpenWebText tokens and 30% of code tokens, with its top code context on indentation; L1/1323 goes from 0.06%
  to 13%, with its top context on a run of escaped tabs (`"\t\t…"`) in a string. In the other direction, L4/961 fires
  on 9.6% of OpenWebText tokens and 0.01% of code tokens.

## 7. Deep scan of layers 0–2: rare and never-firing neurons

The atlases see 1M tokens per corpus, too few to tell a neuron that fires once in 10⁷ tokens from one that never
fires. The deep scan runs only layers 0–2, which hold 100 of the 110 neurons that never fired on any corpus, over
49.6M OpenWebText tokens: 390,632 documents from files 0–7, one window each. The forward pass stops after layer 2,
so it runs at about 22k tokens/s (38 min on an M4, plus 7 min to tokenize).

It records the MLP pre-activation (`mlp.hook_pre`) rather than the post-GELU value. GELU keeps the sign, so the
firing rate is the same, but the pre-activation also says how far a silent neuron stays from firing: its maximum over
the scan, and the top-k contexts that come closest.

    python -m atlas tokenize results/deepscan-owt --n-tokens 5e7 --files 0,1,2,3,4,5,6,7
    python -m atlas run results/deepscan-owt --shards 16 --hooks 'blocks.[0-2].mlp.hook_pre'
    python -m atlas export results/deepscan-owt

![Firing rates in the atlas vs the deep scan, and the candidates' maximum pre-activation](figures/results_deepscan.png)

| Layer | Candidates (never fired on 3 × 1M tokens) | Fire in 50M tokens | Still never fire |
| --- | --- | --- | --- |
| 0 | 29 | 4 | 25 |
| 1 | 58 | 4 | 54 |
| 2 | 13 | 3 | 10 |
| **Total** | **100** | **11** | **89** |

- **89 never fire**, which bounds their rate below 6×10⁻⁸ (3 in 50M, 95% CL). The other 11 are rare rather than
  silent, firing 2–39 times each.
- **How far from firing.** Over 50M tokens, 31 of the 89 never come within 1 unit of threshold. L1/242 has mean −9.6,
  std 1.6 and max −2.6; it is as close to dead as a sample can show. Another 32 come within 0.5: L0/1382 reaches
  −0.03, so for these "never" may only mean "rarer than 1 in 50M". For all 89, zero lies 2.9–21 standard deviations
  above the mean pre-activation (median 6.2).
- **Closest approaches look like features.** The contexts that come nearest to firing are specific, much like live
  neurons' top contexts. L0/1382 peaks on "ref…" word pieces («ref»rained, «ref»old), and L2/2980 on the newline that
  closes a quoted paragraph. Of the 11 that do fire, L0/2206 (39 times) fires on «hub» and «Mac» in place names and
  product names ("tourist «hub»", "Vegan Cheddar «Mac» & Cheese"), and L1/1966 (27 times) on the «MORE» link that
  follows an inline headline in political news ("…emergency declaration «MORE» on Tuesday…").
- **Rare neurons get enough examples.** For the 58 rare neurons in layers 0–2, the median number of firing tokens goes
  from 10 in the 1M atlas to 512.
- **Consistency.** Where both measure a rate well (above 0.1%), the deep scan's firing rates match the 1M atlas's:
  median ratio 1.000, 1st–99th percentile 0.97–1.03. The deep scan also measures the file-to-file term (§5).
- **Silent on OpenWebText is not silent everywhere.** Two more layer 1 neurons, 1822 and 2400, never fire in the 50M
  OpenWebText tokens, with maxima just below 0 (−0.02 and −0.01). Yet they fired on 1M tokens of code, 33 and 2 times.

## 8. Position and context length

Every statistic above pools positions 1–127 of 128-token windows. A 512-token atlas measures how the statistics depend
on position, out to 511: 2M OpenWebText tokens from 3,912 documents of at least 511 tokens (files 0–7), with
`run --positions`. That adds moments per position bucket for every unit (`positions.parquet`; buckets 1, 2–3, 4–7, …,
256–511) and a residual covariance per coarse bucket (`resid_positions.npz`; 1–7, 8–31, 32–127, 128–511).

    python -m atlas tokenize results/atlas-512 --n-tokens 2e6 --seq-len 512
    python -m atlas run results/atlas-512 --shards 16 --positions
    python -m atlas export results/atlas-512

GPT-2's attention is causal, so positions 1–127 of a 512-token window compute exactly what a 128-token window of the
same text would. "Context length" is therefore position: the longer windows add positions 128–511. The only other
difference is which documents qualify, since a 512-token window needs a document of at least 511 tokens.

![Firing rate, total variance and effective dimension by position](figures/results_position_layers.png)

**Firing rates.** Median MLP neuron firing rate by position:

| Layer | 1 | 2–3 | 4–7 | 8–15 | 16–31 | 32–63 | 256–511 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 4.8% | 7.2% | 6.7% | 6.5% | 6.2% | 6.0% | 5.8% |
| 1 | 5.0% | 14.8% | 21.3% | 26.8% | 28.6% | 28.8% | 28.8% |
| 6 | 5.5% | 10.0% | 10.1% | 10.2% | 10.2% | 10.4% | 10.2% |
| 11 | 10.7% | 12.8% | 13.2% | 13.7% | 13.9% | 14.0% | 13.5% |

Layer 1's high firing rate (§3) is a property
of position: its median neuron fires on 5% of tokens at position 1 and builds up over the first ~16 positions to 29%.
Every layer fires least at position 1, and from about position 16 on firing rates are nearly flat out to 511.

**Offset dimensions.** The offset dimensions of §2 are largest at the start of the context and decay with position
(mean over tokens):

| Dimension, layer | 1 | 4–7 | 16–31 | 64–127 | 256–511 |
| --- | --- | --- | --- | --- | --- |
| 447, layer 11 | 124.8 | 108.5 | 93.4 | 84.1 | 80.4 |
| 481, layer 11 | 131.5 | 113.3 | 96.8 | 87.4 | 81.8 |
| 373, layer 1 | −26.7 | −25.2 | −18.9 | −14.0 | −12.9 |
| 64, layer 1 | −14.7 | −21.0 | −24.7 | −23.3 | −22.2 |

Most lose a third to a half of their size between position 1 and 256–511, with the steepest change in the first ~30
positions. Dimension 64 is the exception: it starts smaller and settles by position 16.

![Per-unit shifts between positions, and long-document selection](figures/results_position_units.png)

**Per-unit position dependence.** For every unit, the shift of its mean between position buckets, in units of its
spread, against a noise floor from the bootstrap errors of the two buckets:

| Median \|Δ mean\| / std | MLP neurons (floor) | Residual dims (floor) |
| --- | --- | --- |
| positions 1–7 vs 32–127 | 0.077 (0.011) | 0.17 (0.012) |
| positions 256–511 vs 32–127 | 0.018 (0.005) | 0.058 (0.006) |
| positions 1–127: long documents vs the default atlas | 0.009 (0.005) | 0.011 (0.006) |

- **Early positions** differ the most, 7–14× the floor.
- **Late positions** differ by much less, 3.5–10× the floor: real but small, mostly in the residual stream.
- **Long documents** (at least 511 tokens) differ from the default atlas's documents (at least 127) by under 2× the
  floor, a third of the FineWeb shift (§6). The 128-token results carry over to long documents.

**Caution: covariance by position mixes in the position embedding.** A bucket's covariance includes the variation
between the positions inside it as well as between tokens at the same position. At layer 0, the position embedding
alone is 51% of the variance in positions 128–511 (9–24% in the narrower buckets), and it lies almost entirely in about
2 directions. That is why the widest bucket's effective dimension collapses to 7 at layer 0. The effective-dimension
and total-variance panels above therefore measure position spread as well as token content, and only the qualitative
picture is safe: at the final residual the effective dimension falls with position, from 9.7 (positions 1–7) to 5.5
(128–511). Separating the two needs a covariance within each position, the next step for this section.

## 9. Dependence on centering the residual stream

Everything above follows TransformerLens's processed convention, as the pretrained SAEs do. For what the atlas reads,
that convention does one thing: it removes each token's mean over d_model from the residual stream. Centering the
weights that write to the stream is linear, so it equals subtracting that mean. (Folding LayerNorm leaves MLP
activations unchanged.) The `hf-raw` backend skips the centering, on the same tokens as the default atlas:

    python -m atlas run results/atlas-raw --shards 16 --backend hf-raw   # tokens.npy copied from results/atlas

The raw residual is the centered one plus m·**1**, where m is the token's mean over the 768 dimensions. The all-ones
direction is orthogonal to everything in the centered stream, so the raw total variance is the centered one plus the
variance along **1**, exactly, and any change in effective dimension comes from that single direction.

![Residual total variance and effective dimension, centered vs raw](figures/results_convention.png)

| Layer | Total variance: centered | Raw | Along the all-ones direction (share) | Mean of m | Std of m | Effective dimension: centered / raw |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 11.6 | 11.7 | 0.0 (0.0%) | −0.002 | 0.003 | 14.3 / 14.3 |
| 4 | 2,156.7 | 2,157.1 | 0.4 (0.0%) | −0.018 | 0.024 | 104.0 / 103.9 |
| 8 | 5,861.2 | 5,862.2 | 1.0 (0.0%) | 0.025 | 0.035 | 170.7 / 170.6 |
| 11 | 19,070.0 | 19,072.6 | 2.6 (0.0%) | 0.173 | 0.058 | 104.1 / 104.1 |
| 12 (final) | 56,096.7 | 56,138.3 | 41.5 (0.1%) | −0.094 | 0.233 | 6.8 / 6.8 |

- **MLP neurons are identical,** to the last bit (maximum difference 0), as expected: LayerNorm removes the mean
  before every MLP.
- **The all-ones direction is nearly empty.** A token's mean over d_model is on average −0.09 to 0.17, with a spread
  of 0.003–0.23 across tokens, against unit spreads that grow from about 0.1 to over 8 with depth. It carries at most
  0.1% of the raw variance. Raw minus centered total variance equals the variance along **1** at every layer, as it
  must.
- **Nothing downstream changes.** Effective dimensions agree within errors at every layer. The offset dimensions are
  the same: 79 (layer, dimension) pairs either way, 78 in common, and the other one sits at the 3-std threshold. Their
  means move by at most 0.2.

LayerNorm removes the mean before every block reads the stream, so the model gets almost no training signal along the
all-ones direction, which is consistent with it staying nearly empty. For GPT-2 small this systematic is negligible.
It would matter for a model or tool where that direction is not nearly empty.

## 10. Systematic uncertainties

Every systematic measured so far, ranked by size. Sizes are median shifts of a unit's mean over all units, in units of
that unit's spread (std), unless noted.

| Source | How measured | Typical size | Relative to the statistical error | Verdict |
| --- | --- | --- | --- | --- |
| [Corpus: Python code](#6-corpus-systematic-openwebtext-vs-fineweb-vs-python-code) | code vs OpenWebText, 1M tokens each | 0.17 (MLP), 0.26 (residual); firing rates change by a factor of ~2.2 | ~100× | Dominant: a different regime, with effective dimension ~25 against ~170 |
| [Position: the first ~7 tokens](#8-position-and-context-length) | positions 1–7 vs 32–127, 512-token atlas | 0.08 (MLP), 0.17 (residual) | 7–14× its floor | Large: layer 1 fires on 5% of tokens at position 1 and 29% from 16 on; offsets are largest at position 1 |
| [Corpus: FineWeb](#6-corpus-systematic-openwebtext-vs-fineweb-vs-python-code) | FineWeb vs OpenWebText | 0.028–0.045 | 14–18× | Real but modest: web text against web text |
| [Position: late (256–511)](#8-position-and-context-length) | positions 256–511 vs 32–127 | 0.018 (MLP), 0.058 (residual) | 3.5–10× its floor | Small, mostly in the residual stream |
| [SAE dictionary, for the IOI logit difference](#13-stage-2-the-ioi-repair-in-the-sae-basis) | Bloom 24k vs OpenAI 32k vs OpenAI 128k | share of the LD in the SAE error: 46% / 46% / 26% | — | Large: what the dictionary misses depends on its size, not its architecture |
| [SAE context window](#11-the-gate-into-stage-2) | OpenAI's SAEs on positions 1–63 vs 64–127 | variance explained 0.84 vs −0.27 | — | Decisive: these SAEs are only valid within their 64-token training context; a BOS hardly matters |
| [Long-document selection](#8-position-and-context-length) | documents of ≥511 vs ≥127 tokens, positions 1–127 | ~0.01 | under 2× | Near noise: the 128-token results carry over |
| [Statistical](#5-errors-statistical-and-file-to-file) | bootstrap over shards | **0.002** on a mean; 0.005 relative on a std; 0.007 relative on a firing rate | — | The reference scale |
| [File to file, within a corpus](#5-errors-statistical-and-file-to-file) | cluster bootstrap over files; the 50M-token deep scan | ≤ 0.0003 per file | adds 2–6% to the error | Negligible at this scale, and included in `*_err` |
| [Centering the residual stream](#9-dependence-on-centering-the-residual-stream) | raw vs TransformerLens-centered residuals | MLP: 0 exactly; residual: ≤ 0.1% of the variance, means move by ≤ 0.2 | — | Negligible for GPT-2 small |
| Backend: HF vs TransformerLens bridge ([design doc](atlas-design.md#stage-1-status)) | every hook; the full atlas | 2×10⁻⁵ of the activation scale; atlas statistics agree to ~10⁻⁵ relative | — | Negligible; 0.6% of top-k pointers swap among exact ties |
| [SAE pipeline conventions](#11-the-gate-into-stage-2) | our HF pipeline vs TransformerLens's `HookedTransformer` | 10⁻⁶ (residual stream), 10⁻⁴ (per-token loss) | — | Negligible |
| Histogram subsampling ([design doc](atlas-design.md#stage-1-status)) | every 4th token vs every token | p99 quantiles move by a median 0.02 std (0.13 for the worst 1% of units) | — | Affects quantiles only |

The position rows compare one position bucket with another, so their floor is the error of a bucket (0.005–0.012 of
a unit's spread), not of the whole atlas (0.002).

**In short.** The atlas is precise: a unit's mean is known to about 0.2% of its spread, and every implementation choice
(backend, residual centering, merge arithmetic, which files of a corpus are read) sits well below that. What the
numbers mean depends on the data: the corpus (3–4% of a unit's spread for other web text, about 20% for code) and the
first ~16 positions of context move statistics 10–100× more than the statistical error. So a statement such as "neuron
X has mean m" should carry its statistical error and its conditions, "on OpenWebText, positions 1–127". Past position
~16, and for longer documents, those conditions matter much less.

**Not yet measured.**

- **Context beyond 511 tokens.** GPT-2 goes to 1024.
- **The row layout.** Rows put a BOS before a window that starts mid-document, unlike GPT-2's training windows; that
  may be part of why the first positions look special. One consequence is now seen: features that fire on document
  boundaries (a mid-prompt `<|endoftext|>`) never fire here (§12). For OpenAI's SAEs a BOS hardly matters on positions within their context (§11).
- **Covariance within each position.** Effective dimension by position (§8) is only qualitative so far.
- **SAE-specific systematics beyond §13.** The dictionary systematic is measured for the IOI question only (§13);
  for the feature atlas itself (§12) it is not yet.

## 11. The gate into stage 2

A pretrained SAE's features only mean something if our activations are the ones it was trained on: same hook point,
same processing convention, same BOS and context handling. The gate checks this before stage 2 uses any features. It
runs Joseph Bloom's `gpt2-small-res-jb` SAEs (24,576 features at each residual `hook_resid_pre`, layers 0–11) on
254,000 tokens per layer of the default atlas (250 rows from each of its 8 files), through our HF backend, and
compares the metrics with the
[release post](https://www.lesswrong.com/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream):

    run-sae-gate   # = python -m sae_gate results/atlas --out results/sae-gate; about 40 min on an M4

The SAEs' own metadata already matches this pipeline: trained on OpenWebText with 128-token contexts, a prepended BOS,
and centered writing weights, the convention of §9. The metrics follow SAELens's definitions (`sae_lens.evals`):

- **L0**: nonzero features per token, without BOS.
- **Variance explained**, centered: 1 − E‖x − x̂‖² / (E‖x‖² − ‖E x‖²), SAELens's current definition; and about the
  origin, 1 − E‖x − x̂‖² / E‖x‖², as the release post computed it.
- **CE loss**: next-token cross-entropy with the reconstruction spliced in everywhere but BOS. The CE loss score is
  (CE_zero − CE_sae) / (CE_zero − CE_clean), with the zero ablation at every position.
- **Alive features**: features that fire at least once on these tokens.

![The gate: L0, variance explained and CE loss against the release post](figures/results_sae_gate.png)

| Layer | L0 (post) | Alive (post) | Variance explained, centered | About the origin (post) | CE increase (post) | CE loss score |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 12.4 (12.24) | 79.3% (80.0%) | 97.8% | 98.9% (99.15%) | 0.020 (0.02) | 0.999 |
| 1 | 14.7 (14.68) | 83.1% (83.4%) | 95.0% | 98.1% (98.37%) | 0.018 (0.03) | 0.996 |
| 2 | 18.7 (18.80) | 79.6% (80.0%) | 94.3% | 97.8% (98.07%) | 0.064 (0.07) | 0.987 |
| 3 | 26.0 (25.75) | 86.3% (86.3%) | 92.0% | 96.6% (96.97%) | 0.155 (0.18) | 0.985 |
| 4 | 33.3 (33.14) | 97.7% (97.7%) | 90.1% | 95.3% (95.77%) | 0.122 (0.14) | 0.992 |
| 5 | 44.1 (43.61) | 99.7% (99.7%) | 88.8% | 94.3% (94.90%) | 0.138 (0.15) | 0.989 |
| 6 | 50.6 (49.68) | 99.9% (100%) | 87.5% | 93.3% (93.90%) | 0.138 (0.14) | 0.987 |
| 7 | 58.3 (57.29) | 100% (100%) | 86.4% | 92.5% (93.08%) | 0.147 (0.15) | 0.985 |
| 8 | 66.5 (65.47) | 100% (100%) | 85.5% | 92.0% (92.57%) | 0.150 (0.15) | 0.981 |
| 9 | 72.1 (71.10) | 100% (100%) | 84.5% | 91.5% (92.05%) | 0.156 (0.15) | 0.989 |
| 10 | 55.1 (53.79) | 100% (100%) | 81.3% | 90.5% (91.12%) | 0.202 (0.22) | 0.976 |
| 11 | 60.3 (59.16) | 99.8% (100%) | 81.3% | 93.0% (93.30%) | 0.234 (0.27) | 0.965 |
| 12 (final residual) | 70.1 (—) | 100% (—) | 89.0% | 97.1% (—) | 0.329 (—) | 0.928 |

Row 12 is the SAE on the final residual stream (`blocks.11.hook_resid_post`), gated for §13; the release post has no
row for it, and SAELens's rough `expected_l0` is 70. The CE increase is CE_sae − CE_clean in nats per token; the post's is its CE with the SAE minus a clean CE it gives
only as "3.3", so it is uncertain by about ±0.05. Bootstrap errors over files are at most 0.13 on L0, 0.05 points on
variance explained and 0.0003 on the CE loss score.

**The gate passes.**

- **Conventions agree exactly.** On 32 rows per layer, our HF pipeline reproduces the residual stream of
  TransformerLens's `HookedTransformer`, loaded as SAELens recommends, to 3×10⁻⁶ relative, and its per-token losses
  to 10⁻⁴.
- **Every release metric matches.** L0 agrees with the post within 2.4% at every layer, alive features within 0.7
  points, variance explained about the origin within 0.7 points (always slightly below, by 0.2–0.6), and the CE increase within the rounding of the
  post's clean CE.
- **SAELens's built-in reference values for this release are rough.** Its `expected_l0` (10, 10, 18, 23, …) and
  `expected_var_explained` (0.999, 0.9, 0.77 in three tiers) are hand-entered approximations; the post's table is the
  reference to use.

Two differences, both understood:

- **Centered variance explained is 1–12 points lower than the post's.** The post divides by E‖x‖², which includes the
  residual stream's large constant offsets (§2). The SAE reconstructs those almost for free with its decoder bias, so
  the gap grows with depth as the offsets do. The centered value is the more honest measure of how much of the
  stream's variation the SAE captures: 98% at layer 0, falling to 81% at layers 10–11.
- **Our clean CE is 3.50, against the post's 3.3.** The absolute loss depends on the evaluation text: our rows are a
  BOS plus a window starting mid-document, so early positions have little context. The increase from splicing in the
  SAE, which is what the gate tests, agrees.

With the gate passed, stage 2 can use these SAEs on this pipeline: feature atlases in the SAE basis, with the same
accumulators.

**OpenAI's SAEs.** The dictionary systematic of §13 also uses OpenAI's TopK SAEs for GPT-2 small (Gao et al. 2024;
32k and 128k features, k = 32, on `resid_post`), at the three points the IOI analysis needs. Their conventions differ
from Bloom's: each token is normalized to unit std before encoding (undone on decoding), and they were trained on
64-token contexts without a BOS. The normalization removes each token's mean, so centering does not matter (§9), but
the context does:

| Rows, OpenAI 32k SAE | Centered variance explained: before layer 10 / before layer 11 / final residual |
| --- | --- |
| BOS + 127 tokens (our rows) | 0.83 / 0.82 / 0.90 |
| No BOS, 64 tokens (their training format), all positions | 0.99 / 0.98 / 0.91 |
| BOS + 63 tokens | 0.84 / 0.83 / 0.90 |

The second row looks best but is misleading: without a BOS, position 0 is a real token that becomes the attention
sink, with about 20× the norm of the others, and it dominates the pooled variance while being easy to reconstruct
(0.6% error, against 26% for positions 1–63). Measured directly before layer 10, positions 1–63 are reconstructed
equally well with or without a BOS (0.84 either way), while positions 64–127, outside the training context, are
reconstructed worse than by the mean (−0.27). So these SAEs are only valid within 64 tokens, where a BOS makes little
difference, and the gate that matters for the IOI prompts (15–18 tokens with a BOS) is BOS + 63:

| BOS + 63 tokens | L0 | Centered variance explained | About the origin | CE loss score |
| --- | --- | --- | --- | --- |
| 32k, before layer 10 / 11 / final residual | 32 | 0.84 / 0.83 / 0.90 | 0.92 / 0.94 / 0.97 | 0.98 / 0.97 / 0.94 |
| 128k, before layer 10 / 11 / final residual | 32 | 0.87 / 0.86 / 0.91 | 0.94 / 0.95 / 0.98 | 0.99 / 0.98 / 0.96 |

Our activations match TransformerLens's for these SAEs too (residual stream within 10⁻⁵, per-token loss within
10⁻⁴), and variance explained about the origin is consistent with SAELens's rough reference of 0.9; there is no
per-layer published table to compare against.

## 12. Stage 2: the atlas in the SAE basis

With the gate passed, the gate-validated SAEs encode the residual stream inside the hooks, and their 24,576 features at
layers 6 and 9 become ordinary atlas units (`kind = "sae"`): firing rate, moments, histogram and top-20 contexts, on
the same 1M tokens as the default atlas, so features and neurons can be compared token for token. At layer 9 every
nonzero activation is also kept as a sparse event (`events-L9-*.parquet`: row, pos, feature, value), 71.5M of them
(454 MB), for per-token queries such as the IOI analysis to come.

    mkdir -p results/atlas-sae && cp results/atlas/{tokens.npy,docs.npy,manifest.json} results/atlas-sae/
    python -m atlas run results/atlas-sae --shards 16 --batch-size 32 --hooks 'blocks.6.hook_resid_pre,blocks.9.hook_resid_pre' \
        --sae gpt2-small-res-jb --sae-layers 6,9 --sae-events 9
    python -m atlas export results/atlas-sae

The run stops after layer 9, so it takes about 5 minutes on an M4 (3,600 tokens/s), plus the export.

![Feature firing rates against neurons, and single features against Neuronpedia](figures/results_sae_features.png)

| Layer | Features | Never fire (1M tokens) | Firing rate: 10% / median / 90% | Fire on > 1% of tokens | L0 | Median MLP neuron firing rate |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 24,576 | 13 | 0.018% / 0.055% / 0.55% | 1,043 | 50.7 | 10.2% |
| 9 | 24,576 | 3 | 0.035% / 0.14% / 0.75% | 1,494 | 72.1 | 10.5% |

- **Features are sparse, neurons are not.** The median feature fires about 75× (layer 9) to 185× (layer 6) less often
  than the median MLP neuron at the same layer, and almost no feature is dead over 1M tokens.
- **Single features match Neuronpedia.** Neuronpedia hosts dashboards for these SAEs, built from OpenWebText with
  128-token prompts. For 50 live features per layer, evenly spaced through our firing-rate distribution (records
  fetched once from its public API and cached under `results/neuronpedia/`):
  - firing rates agree to a median ratio of 1.02 (layer 6) and 1.00 (layer 9), the middle two-thirds within about ±10%;
  - maximum activations to a median ratio of 0.97, slightly lower here because we see 1M tokens against its 3.1M;
  - its top-10 activating tokens are among our top-20 for a median of 100% of them, and 88–94% of features share at
    least one.
- **One outlier is a document-boundary feature.** L6/6819 fires on 5 of our 1M tokens (max 0.7) but on 0.13% of
  Neuronpedia's (max 60), always on an `<|endoftext|>` in the middle of a prompt, where one document ends and the next
  begins. Neuronpedia's prompts concatenate documents; our rows hold one document window after a BOS, which is
  excluded, so such features have nothing to fire on here. This is a first measured consequence of the row-layout
  systematic (§10): features tied to document boundaries are invisible in this atlas, 1 of the 100 sampled. The other
  outlier, L9/21059, is about equally rare in both (rates of 10⁻⁶ and 3×10⁻⁷).
- **The decoder bias is the residual stream's mean.** cos(b_dec, mean) is 0.993 (layer 6) and 0.994 (layer 9), and
  the offset dimensions of §2 agree within a few percent (dimension 447 at layer 9: mean 49.3, b_dec 51.3). The SAE
  stores the constant offsets in its bias, which is why its variance explained about the origin runs so far above the
  centered value (§11).
- **The examples read as features.** L9/11770 fires on the token "Fun" ("Func", "Funtenna", "Fun fact") and L9/23599
  on the unit in "over the last five years / three decades", matching Neuronpedia's explanations. The densest feature,
  L9/7155 (11% of tokens), fires on conjunctions and punctuation in our top contexts, which Neuronpedia's label
  ("dates and months") does not obviously describe.

## 13. Stage 2: the IOI repair in the SAE basis

[ioi-hydra](https://github.com/rreece/ioi-hydra) found that mean-ablating the three name-mover heads {9.9, 9.6, 10.0}
removes 4.57 logits of direct effect, yet the logit difference LD = logit(IO) − logit(S) does not drop: the network
repairs 103% of the damage, mostly through the copy-suppression head 10.7 releasing its brake and backup heads in
layers 10–11 stepping up. Head-level accounting cannot say whether the repair rebuilds the same *representation* or
reaches the same LD another way. SAE features can.

`python/ioi.py` ports ioi-hydra's 128 prompts and its mean ablation to our HF backend. At each prompt's final token it
encodes the residual stream with the gate-validated SAEs at three points, clean against ablated: **before layer 10**
(after 9.9 and 9.6 are ablated), **before layer 11** (after layer 10's brake and backups respond) and **the final
residual stream**, which the unembedding reads. The final LayerNorm is linear given a run's scale, so at the final
residual LD splits exactly into the SAE's features, its decoder bias and its error:

LD = Σ_f a_f (W_dec[f] · d) / scale + (b_dec · d) / scale + (error · d) / scale + const, with d = γ ⊙ (W_U[IO] − W_U[S]).

At the two earlier points the same projection, with each point's own scale, is a logit lens: an approximation, used
only qualitatively. A **control** ablation (heads 9.10, 9.1 and 10.8, the three in layers 9–10 with the smallest direct
effects and no role in the circuit) shows how much features move under an ablation that should not matter. Errors
are 95% bootstrap intervals over prompts.

**The head-level result reproduces exactly**, on our HF backend without TransformerLens:

| | This pipeline | ioi-hydra |
| --- | --- | --- |
| Clean LD | 3.38 [3.14, 3.61] | 3.38 |
| LD with the name movers mean-ablated | 3.53 [3.25, 3.82] | 3.53 |
| Direct effect lost | 4.57 [4.29, 4.85] | 4.57 |
| Repair fraction | 1.034 | 1.03 |
| 10.7's direct effect, clean → ablated | −2.06 → 0.07 | −2.06 → +0.07 |
| Final LayerNorm scale, clean → ablated | 17.8 → 16.6 | 17.8 → 16.6 |

![The IOI repair in the SAE basis](figures/results_ioi_features.png)

**Features, clean against ablated:**

| Point | Run | L0 | Active features shared (Jaccard) | Per-feature LD contributions (cosine) | Ablated feature LD from clean-active features |
| --- | --- | --- | --- | --- | --- |
| Before layer 10 | name movers ablated | 91 → 55 | 0.47 [0.45, 0.48] | 0.39 [0.35, 0.43] | — |
| Before layer 10 | control | 91 → 90 | 0.98 [0.97, 0.98] | 1.00 | 1.00 |
| Before layer 11 | name movers ablated | 68 → 62 | 0.55 [0.54, 0.57] | 0.87 [0.85, 0.89] | 0.94 |
| Before layer 11 | control | 68 → 68 | 0.97 [0.97, 0.98] | 1.00 | 1.00 |
| Final residual | name movers ablated | 73 → 61 | 0.57 [0.56, 0.58] | 0.92 [0.90, 0.94] | 0.97 |
| Final residual | control | 73 → 73 | 0.97 [0.97, 0.98] | 1.00 | 1.00 |

Before layer 10 the ablated run's feature LD is near zero (0.27), so its share is not meaningful.

**Where the LD lives:**

| Point | Run | LD (logit lens; exact at the final residual) | SAE features | SAE error |
| --- | --- | --- | --- | --- |
| Before layer 10 | clean | 13.98 [13.15, 14.81] | 7.29 | 6.75 |
| Before layer 10 | ablated | 1.86 [1.35, 2.37] | 0.27 | 1.66 |
| Before layer 11 | clean | 8.98 [8.36, 9.58] | 5.21 | 3.81 |
| Before layer 11 | ablated | 8.09 [7.55, 8.63] | 3.95 | 4.19 |
| Final residual | clean | 3.35 [3.06, 3.66] | 1.82 [1.49, 2.12] | 1.56 [1.16, 1.97] |
| Final residual | ablated | 3.51 [3.19, 3.83] | 1.56 [1.28, 1.83] | 1.97 [1.61, 2.32] |

The decoder bias carries −0.02 to −0.07 throughout. At the final residual, features + bias + error + const reproduce
the LD to 2×10⁻⁵. The change in LD under the ablation, +0.15 [−0.02, 0.33], splits into features −0.26 [−0.40, −0.10]
and SAE error +0.41 [0.24, 0.58].

- **The control moves nothing.** 97–98% of active features are shared and the per-feature LD contributions are
  unchanged, so the changes under the name-mover ablation are real.
- **Before layer 10, the damage is visible in the features.** With 9.9 and 9.6 ablated, only 47% of active features
  are shared with the clean run, and the logit-lens LD falls from 14.0 to 1.9.
- **After the repair, the LD is carried by the same features.** Before layer 11 and at the final residual, the
  per-feature LD contributions correlate with the clean run's at cosine 0.87 and 0.92, and 94–97% of the ablated run's
  feature LD comes from features active in the clean run. Many features change (only 55–57% are shared), but not the
  ones carrying the answer: the backups and the released brake restore the clean representation rather than build a
  new one.
- **With Bloom's SAE, about half the LD is outside the dictionary.** At the final residual the SAE's error carries
  1.56 of the clean LD's 3.35. It is not mostly shrinkage: the error's component along the reconstruction (L1-trained
  SAEs under-activate) carries only 0.29 of it, and the rest, 96% of the error's norm, is off-dictionary. The ablation
  moves about 0.4 logits into it (features −0.26, error +0.41). The dictionary systematic below shows this is mostly a
  limit of the dictionary's size.
- **The LD-carrying features are prompt-specific.** Each of the ten that carry the most LD is active on only 5–18% of
  prompts, so the answer is spread over many features, plausibly tied to which names appear, rather than one
  "indirect object" feature. Neuronpedia's explanations for them are generic ("mentions of specific individuals and
  organizations"). The feature that gains the most under the ablation (L12/8884) becomes active on 84% of prompts,
  up from 63%.

**The dictionary systematic.** The same analysis with OpenAI's TopK SAEs (§11), 32k and 128k features, on the same
three points:

![The IOI split across three dictionaries](figures/results_ioi_dictionaries.png)

| Final residual | L0 | Error's share of the clean LD | Change under the ablation: features | SAE error | LD contributions vs clean (cosine) | Ablated feature LD from clean-active features |
| --- | --- | --- | --- | --- | --- | --- |
| Bloom 24k | 73 | 46% | −0.26 [−0.40, −0.10] | +0.41 [+0.24, +0.58] | 0.92 | 0.97 |
| OpenAI 32k | 32 | 46% | −0.18 [−0.31, −0.05] | +0.33 [+0.15, +0.51] | 0.86 | 0.88 |
| OpenAI 128k | 32 | 26% | +0.01 [−0.19, +0.20] | +0.15 [−0.01, +0.30] | 0.94 | 0.88 |

Before layer 11 the contrast is starker: the error carries 3.8–4.0 logits of the logit-lens LD for the two smaller
dictionaries and 0.9 for the 128k one.

- **The dark matter shrinks with a better dictionary, for two different reasons.** Bloom's 24k and OpenAI's 32k SAEs
  leave the same 46% of the LD in their error, but differently (below): Bloom's 73 active features span enough but its
  L1-trained coefficients are off, while 32k's 32 active features span too little. The 128k SAE, with 32 active
  features from a larger dictionary, fixes both and leaves 26%. Its overall reconstruction is only 1–3 points better
  than 32k's (§11).
- **With the largest dictionary the repair is a full restoration.** The features carry the same LD after the repair as
  before (+0.01), and the error's gain (+0.15) is consistent with zero. The shift of LD into the dark matter seen with
  the smaller dictionaries is largely an artifact of their size.
- **The restoration is robust.** In all three dictionaries the ablated run's per-feature LD contributions match the
  clean run's (cosine 0.86–0.94 at the final residual), and 88–97% of its feature LD comes from features active in
  the clean run.

**What carries the answer: name features.** On a name-balanced set (every single-token name as the indirect object
once per template, 472 prompts), call a feature a *dedicated name feature* if more than half of its activation falls on
one IO name and its decoder direction, read through the unembedding, boosts that name's logit more than any other
name's. At the final residual:

| Dictionary | Dedicated name features | Names covered (of 59) | Per name (median) | Clean LD carried by them | By other features | By the SAE error |
| --- | --- | --- | --- | --- | --- | --- |
| Bloom 24k | 270 | 59 | 4 | 1.02 | 0.80 | 1.56 |
| OpenAI 32k | 124 | 54 | 2 | 0.44 | 1.39 | 1.54 |
| OpenAI 128k | 262 | 58 | 4 | 1.35 | 1.13 | 0.88 |

- **The IOI answer is carried mostly by "say this name" features,** about four per name in Bloom's 24k and OpenAI's
  128k dictionaries. Among the ten features carrying the most LD, a "Ben" feature (Bloom 15719) fires for only 8 of
  the 59 names and boosts "Ben" most.
- **The 128k advantage is not coverage.** It has about as many name features as Bloom's 24k and covers about as many
  names, but they carry more LD (1.35 against 1.02), and so do its other features. OpenAI's 32k has half as many name
  features and carries more of the answer in generic ones.

**Who writes the off-dictionary part.** The SAE's error is not linear in what writes the residual stream, but its part
outside the span of the active features' decoder directions (and the decoder bias) is: per prompt, projecting onto
that span's orthogonal complement splits it exactly (to 3×10⁻⁶) into each head's, each MLP's and the embeddings' and
biases' writes. The rest of the error lies inside the span, from coefficients that are off.

![Each head's direct effect, and the part outside the active features' span](figures/results_ioi_offdict.png)

| Final residual, clean | SAE error LD | Outside the active features' span | Inside it (coefficients off) |
| --- | --- | --- | --- |
| Bloom 24k (ReLU, L1; 73 active) | 1.56 | 0.92 | 0.63 |
| OpenAI 32k (TopK; 32 active) | 1.54 | 1.40 | 0.14 |
| OpenAI 128k (TopK; 32 active) | 0.88 | 0.86 | 0.01 |

| Head | Direct effect, clean → ablated | Outside the span, OpenAI 128k: clean → ablated | Share of the clean direct effect (Bloom 24k / 128k) |
| --- | --- | --- | --- |
| 9.9 (name mover) | +2.89 → −0.03 | +1.14 → −0.04 | 41% / 39% |
| 9.6 (name mover) | +1.12 → +0.01 | +0.41 → +0.00 | 44% / 37% |
| 10.0 (name mover) | +0.53 → −0.01 | +0.19 → −0.02 | 38% / 35% |
| 10.7 (brake) | −2.06 → +0.07 | −0.89 → +0.12 | 42% / 43% |
| 10.10 (backup) | +0.55 → +1.32 | +0.17 → +0.41 | 27% / 31% |
| 10.6 (backup) | +0.39 → +0.80 | +0.12 → +0.28 | 40% / 30% |
| 11.10 (late brake) | −0.92 → −1.19 | −0.42 → −0.54 | 43% / 46% |

- **The off-dictionary LD is written by the circuit's own heads.** The name movers, the brake 10.7 and the late brake
  11.10 each put 35–46% of their direct effect outside the active features' span, in both dictionaries; the backups
  27–40%. Apart from head 8.10 (+0.17), no other component writes much of it.
- **Under the ablation the balance shifts, not the mechanism.** The name movers' off-span part vanishes with them, the
  released brake's −0.89 goes with it, and the backups add theirs, for a net change of +0.35 (Bloom) or +0.15 (128k)
  logits outside the span.
- **What the off-span direction is remains open.** Each head writes a large, consistent share of its IO-name signal
  outside even the 128k dictionary's active features; why the dictionaries do not capture it is the next question.

## 14. Validation

Checks that the atlas computes what it claims, grouped by what they check against. The notebook reruns the first
group's checks and the calibration test with every run (notebook §12); the unit tests are in `tests/`.

**Exactness and internal consistency**

| Check | Result |
| --- | --- |
| Pointers reproduce: 12 random top- and bottom-k entries recomputed from `(row, pos)` | all match, worst relative difference 2.1e-6 |
| Position buckets pool back to the whole: each unit's buckets in the 512-token atlas (§8), merged, against its overall moments | counts equal; means agree to 1e-7 of a unit's spread, stds to 7e-7 relative, firing rates exactly |
| Deep scan vs the 1M atlas (§7): firing rates on different data, same code | median ratio 1.000, 1st–99th percentile 0.97–1.03 |
| Centering identity (§9): raw minus centered total variance against the variance along the all-ones direction | equal at every layer, as the decomposition requires |
| SAE feature events vs feature moments (§12): events per feature against firing counts, accumulated independently | equal for every layer-9 feature; 72.05 events per token, equal to L0 |
| Linear final-LayerNorm split of the logit difference (§13): features + bias + error + const against the logits | equal to 2e-5 |
| Component split of the off-dictionary remainder (§13): heads + MLPs + rest against the remainder, and the components against the residual stream | equal to 3e-6 and 8e-6 |
| Unit tests: exact merging for every accumulator (split vs whole, including bottom-k and the position buckets), sampling across a file, hook selection, the file-level bootstrap and its small-sample correction, the uncentered backend, the SAE gate's splicing hook, SAE features and events in a run, IOI heads against processed TransformerLens, the name-balanced prompts and the off-dictionary split | 50 of 50 pass |

**Cross-implementation**

| Check | Result |
| --- | --- |
| Plain HF backend vs TransformerLens bridge, 7 hook types × layers 0, 5, 11 (`tests/test_backends.py`) | agree within 2e-5 of the activation scale; a run over a subset of hooks stops only after the last one reached |
| Full atlas, HF vs TransformerLens backend ([design doc](atlas-design.md#stage-1-status)) | statistics agree to ~1e-5 relative; 0.6% of top-k pointers swap among exact ties |
| SAE pipeline vs TransformerLens's `HookedTransformer` (§11), every layer including the final residual, Bloom's and OpenAI's SAEs | residual stream within 1e-5 relative, per-token loss within 1e-4 |
| Heads read and zero-ablated at `attn.c_proj` vs processed TransformerLens's `hook_z` (§13; `tests/test_backends.py`) | head outputs within 1e-5, losses under a zero ablation within 1e-4 |

**External: published numbers**

| Check | Result |
| --- | --- |
| Bloom's `gpt2-small-res-jb` SAEs vs the release post (§11) | L0 within 2.4%, alive features within 0.7 points, variance explained about the origin within 0.7 points, CE increase within the rounding of the post's clean CE |
| ioi-hydra's head-level results on our HF backend (§13) | clean and ablated LD, direct effect lost, repair fraction and per-head responses reproduce to the reported precision |
| Single SAE features vs Neuronpedia's dashboards (§12), 50 per layer | firing rates within a median ratio of 1.00–1.02, maxima 0.97, top tokens shared for a median 100%; the one outlier is a document-boundary feature |
| Known features of GPT-2 small's residual stream (§1, §2) | outlier dimensions (480 early, 447 in the middle layers, 496 at the final residual) and large constant offsets, consistent with the outlier ("massive activation") dimensions reported for GPT-2 |

**Statistical calibration**

| Check | Result |
| --- | --- |
| The file-to-file test (§5) under its null: shards shuffled across files, 20 permutations | median ratio 1.007 (range 0.88–1.14), against 0.990 for F(7, 8). The actual files give 1.061, inside that range: the 1M atlas alone cannot resolve the file-to-file term, which is why §5 measures it with the deep scan |
| Small-sample correction of the bootstrap (unit test) | the corrected cluster bootstrap recovers s / √G for a statistic that differs only between files |

**Data integrity**

| Check | Result |
| --- | --- |
| No document used twice, default atlas | 7,816 rows from 7,816 distinct (file, document) pairs, files 0–7 |
| Identical 127-token windows across documents | one window appears twice: newsletter boilerplate ("Sign up for Take Action Now and get three actions in your inbox every week") |
| No documents shared between OpenWebText files 0 and 1 (a one-off check, hashing all 100,173 documents of each) | none shared |

## Caveats

- **Few files for the other corpora.** The FineWeb and code atlases use 2 files each, so their errors are statistical
  only. On OpenWebText the file term adds only a few percent, but FineWeb (crawl dumps) and code (repositories) may
  vary more from file to file.
- **Context.** The main atlases cover positions 1–127 and the position study 1–511, not GPT-2's full 1024. Rows start
  mid-document after a BOS, which GPT-2 never saw in training, and that may be part of what makes the first positions
  special (§8).
- **Covariance by position mixes in the position embedding.** A position bucket's covariance includes the variation
  between the positions inside it, so the effective dimension by position (§8) is only qualitative.
- **One ablation for §13.** The IOI analysis uses mean ablation only, and before the final residual its LD
  attribution is a logit lens. OpenAI's SAEs are valid only within their 64-token training context (§11).
- **The neuron basis.** Single neurons are often polysemantic, so a unit's top-10 contexts (as for L9/1000) may not
  name what it does. SAE bases are stage 2.
- **Thin tails.** In the 1M-token atlases, rare units have fewer than about 100 examples; the deep scan (§7) covers
  this for layers 0–2 only. Histograms (and so quantiles) use every 4th token.
- **Silent is not dead.** "Never fires" is always relative to a corpus and a sample size: two neurons silent on 50M
  OpenWebText tokens fire on code (§7).

## Next

Every stage 1 systematic is now measured, the gate into stage 2 passes, and the first stage 2 question has an answer. From the [roadmap](atlas-design.md#roadmap):

- **The off-span IO-name signal (§13):** why every circuit head writes 30–45% of its direct effect outside the span of
  even the 128k dictionary's active features, and whether larger or task-trained dictionaries capture it.
- **Systematics for §13:** zero and resample ablation, other templates and seeds, as in ioi-hydra.
- **More of the feature atlas:** layers 10–11 for the IOI question, and the corpus and position systematics in the
  feature basis.
- **Follow-ups for stage 1:** a covariance within each position, to separate token content from position spread
  (§8); deep scans of the 10 silent candidates in layers 3–6, and of code for the code-specific neurons.
