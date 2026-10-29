# GPT-2 small activation atlas

A queryable atlas of what every MLP neuron and residual-stream dimension in GPT-2 small does over a text corpus:
per-unit statistics, top-activating contexts with decoded text, and residual-stream geometry, with bootstrap
errors. Built around streaming, exactly mergeable reductions, so it never stores raw activations, and kept
laptop-sized: 1M OpenWebText tokens from ~8k documents in about 5 minutes on an M4.

**Design: [docs/atlas-design.md](docs/atlas-design.md)** · **Results: [stage 1](docs/results-stage1.md) · [stage 2](docs/results-stage2.md)**


![MLP neuron activation statistics in GPT2 with OpenWebText data.](docs/figures/results_mlp_neurons.png)

## Setup

    source setup.sh

The first time, this creates `.venv`, installs `requirements.txt` and registers the `gpt2-atlas` Jupyter kernel.
After that it just activates the venv and sets `PATH` (`scripts/`) and `PYTHONPATH` (`python/`).


## Run

    run-atlas                                # 1M tokens from 8 files -> results/atlas/*.parquet, ~10 min on an M4
    run-atlas results/atlas-fineweb 1e6 fineweb   # same for FineWeb; `code` for Python (codeparrot-clean)
    python -m atlas run DIR --positions      # also statistics per position bucket (positions.parquet)
    jupyter lab notebooks/atlas.ipynb        # explore: residual geometry, neuron stats, errors, top contexts
    run-notebook                             # same notebook headless, in place; figures -> docs/figures/results_*.png
    run-sae-gate                             # stage 2 gate: pretrained SAEs vs their release's metrics, ~40 min
    python -m atlas run DIR --sae gpt2-small-res-jb --sae-layers 6,9 --sae-events 9   # stage 2: SAE features as units
    python -c "import duckdb; duckdb.sql(\"select value, context from 'results/atlas/topk.parquet' where kind='post' and layer=9 and unit=1000 order by rank\").show()"
    pytest tests                             # exact merging; HF backend == TransformerLens bridge (~10 s)

The steps can also run separately, e.g. shards on different machines: `python -m atlas {tokenize,run,export} --help`.


## Layout

    python/atlas.py             tokenize, model backends, mergeable accumulators, bootstrap, export (CLI: python -m atlas)
    notebooks/atlas.ipynb       exploration: residual geometry, neuron statistics, errors, single neurons
    scripts/run-atlas           pipeline end to end
    scripts/run-notebook        headless notebook execution, in place
    python/sae_gate.py          the gate into stage 2: SAELens SAEs on our pipeline (CLI: python -m sae_gate)
    scripts/run-sae-gate        the gate end to end
    tests/                      exact merging and bootstrap; HF backend matches TransformerLens
    docs/atlas-design.md        design doc; figures in docs/figures/
    docs/results-stage1.md      stage 1 results: the neuron atlas and its systematics; figures written by the notebook
    docs/results-stage2.md      stage 2 results: the SAE gate, the feature atlas and the IOI repair
    docs/results.md             an index of the two
    docs/vllm-monitoring-ideas-2026-10-21.txt   notes on activation monitoring in live vLLM serving (out of scope here)


## See also

This project grew out of [ioi-hydra](https://github.com/rreece/ioi-hydra), a one-day study of self-repair in GPT-2 small's IOI circuit.

Read more about my [research here](http://rreece.github.io/research/).


Ryan Reece      
2026/10/26

