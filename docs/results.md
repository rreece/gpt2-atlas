# GPT-2 Activation Atlas — Results

*Ryan Reece, 2026-10-26, updated 2026-10-29. Design: [atlas-design.md](atlas-design.md).*

The results are in two documents, one per stage of the [roadmap](atlas-design.md#roadmap).

- **[Stage 1: the neuron atlas](results-stage1.md).** Statistics for every MLP neuron and residual-stream dimension of
  GPT-2 small over 1M OpenWebText tokens, with errors that include the file-to-file term; corpus (FineWeb, code),
  position (out to 511 tokens) and residual-centering systematics; a 50M-token deep scan of layers 0–2; and the
  validation of the pipeline. Headline: the atlas is precise to about 0.2% of a unit's spread, and what moves it is the
  data (the corpus and the first ~16 positions), not the implementation.
- **[Stage 2: SAE features](results-stage2.md).** The gate that checks pretrained SAEs against their releases on this
  pipeline; the atlas in Bloom's SAE basis at layers 6 and 9; and the IOI repair in the feature basis with three
  dictionaries. Headline: when the name movers are ablated, the repair restores the clean run's "say this name"
  features, robustly across ablation methods and prompt sets, and what the dictionary misses depends on its size.
