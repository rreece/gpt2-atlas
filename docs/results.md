# GPT-2 Activation Atlas — Results

*Ryan Reece, 2026-10-26, updated 2026-10-31. Design: [atlas-design.md](atlas-design.md).*

A one-page summary: the most important conclusions so far, and what the code base can do. The full results, with
their errors, systematics and validation, are in two documents, one per stage of the [roadmap](atlas-design.md#roadmap):
[stage 1, the neuron atlas](results-stage1.md), and [stage 2, SAE features](results-stage2.md).

## Conclusions

**The neuron atlas (stage 1)**, statistics for every MLP neuron and residual-stream dimension of GPT-2 small over 1M
tokens:

- **The measurements are precise and reproducible.** A unit's mean is known to about 0.2% of its spread, including the
  variation between files of the corpus. Every stored exemplar recomputes to 10⁻⁶, and our plain Hugging Face pipeline
  matches TransformerLens to 10⁻⁵.
- **What moves the atlas is the data, not the implementation.** Changing the corpus moves a typical unit 14–18 times
  further than the noise for FineWeb, and 85–133 times for Python code. Within a context, only the first ~16 positions
  differ much from the rest. Processing conventions (centering the residual stream) change almost nothing.
- **The residual stream expands, then collapses.** Its effective dimension rises from 14 at the embedding to about 170
  in layers 7–9, then falls to about 7 at the final residual. Its largest direction in layers 0–5 is position; at the
  final residual it is about the next token (newline against "."). A few dimensions are near-constant offsets.
- **Rare and silent neurons are in the first layers.** Almost all neurons that fire on fewer than 1 in 10⁴ tokens are
  in layers 0–2, and 89 there never fire in 50M tokens.

**SAE features and the IOI circuit (stage 2)**, the residual stream in pretrained sparse autoencoder (SAE) bases, and
how GPT-2 small's indirect-object-identification (IOI) circuit repairs itself when its name-mover heads are removed:

- **Pretrained SAEs work on this pipeline, within their training conditions.** Bloom's (every layer), OpenAI's (32k and
  128k) and Apollo's (four at layer 10) reproduce their published metrics here, and single features match Neuronpedia's dashboards. OpenAI's are valid only
  within 64 tokens, and Apollo's end-to-end SAE only without a BOS.
- **The circuit's self-repair restores the same representation.** With the name movers ablated, the answer is carried
  by the same SAE features as in the clean run (cosine 0.86–0.94, in three dictionaries), mostly "say this name"
  features, about four per name. It holds across ablation methods, prompt sets and templates.
- **The SAEs miss a large part of the answer, and for a structural reason.** Their error carries about half of the
  logit difference with 24k and 32k dictionaries, and a fifth to a third with 128k. The missed part is small in norm
  (about 3% of the stream's squared norm) but lies along the logit direction, which carries little of the stream's
  variance, so a reconstruction loss neglects it. It is spread over hundreds of the dictionary's directions, so a
  better encoder would not recover it.
- **Training on the model's output helps, but does not close the gap.** Spliced in before layer 10, an SAE trained
  end-to-end on the model's output keeps 68–69% of the logit difference, against 50% for one trained to reconstruct
  its layer, at the same sparsity. Whether an objective weighted to the task recovers the rest is the open question
  for stage 3.

## What the code base can do

- **Build an atlas** of any GPT-2 small hook points over OpenWebText, FineWeb or Python code, sharded across processes
  or machines and merged exactly, on a laptop: per-unit statistics, top- and bottom-k exemplars, residual covariance,
  statistics per position, bootstrap errors including the file-to-file term, and deep scans of chosen layers
  (`run-atlas`; `python -m atlas tokenize|run|export`).
- **Put SAE features in the atlas** as units, with every nonzero activation kept as a sparse event (`run --sae`).
- **Query it** with DuckDB or pandas: plain Parquet and npz files, no server ([README: Query](../README.md#query)).
- **Make principal components units** of the residual stream, with exemplars (`run --pcs N --pcs-from DIR`).
- **Inspect and recompute any unit**: `python -m atlas show DIR KIND LAYER UNIT` prints a unit's statistics and reruns
  its top contexts from the stored tokens; `atlas.recompute` returns any hook's activations on any rows.
- **Gate a pretrained SAE** against its published metrics on this pipeline, in its own context format (`run-sae-gate`).
- **Study a circuit in the feature basis** (`python/ioi.py`): the IOI prompts; mean, zero and resample ablation of any
  heads; the logit difference split exactly into heads, MLPs, SAE features, bias and error; and any SAE spliced into
  the residual stream.
- **Reproduce every result and figure** by rerunning `notebooks/atlas.ipynb` (`run-notebook`), and the 57 unit tests
  (`pytest tests`).

**The demo** (`notebooks/demo.ipynb`, 23 s) shows these working together in five steps:

1. **One neuron:** layer-6 neuron 1282 fires hardest on " John" after a title (Chief Justice John Roberts,
   Representative John Lewis), but also on unrelated words: one neuron mixes several things.
2. **One token in the SAE basis:** on "Chief Justice« John»", the layer-9 SAE separates it into a feature for the name
   John (firing on 0.02% of tokens, against the neuron's 5%) and features for courts and justices.
3. **Neuronpedia:** its explanations and top logits for those features agree ("the name John"; Scalia, justices).
4. **The IOI repair on one prompt:** in "Then, John and Sarah went to the office. Sarah gave a book to", one "say John"
   feature carries the answer, and the backup heads write the same feature when the name movers are ablated.
5. **What the SAEs miss:** spliced into the stream, a local SAE loses half of the answer on average, an end-to-end
   SAE a third.

## Limits

Laptop scale and one model: GPT-2 small, 1M tokens per corpus (50M for the deep scan), contexts up to 512 tokens, SAE
units at layers 6 and 9, and one task (IOI), with pretrained dictionaries only. Each document lists its own caveats.
