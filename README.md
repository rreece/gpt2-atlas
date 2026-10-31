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


## Query

The outputs are plain files in the run's directory (`results/` is not in git: run the pipeline first; `results/atlas-sae`
comes from the `--sae` line above):

    neurons.parquet      one row per unit: hook, kind (post = MLP neuron, resid, sae), layer, unit, n, mean, std, max,
                         frac_pos (firing rate), p50 ... p999, and bootstrap errors (*_err, *_err_stat)
    topk.parquet         one row per exemplar: hook, kind, layer, unit, side (top; bottom for resid), rank, value,
                         row, pos, token, context (decoded, the token marked « »)
    events-L9-*.parquet  SAE runs only: every nonzero feature activation at layer 9: row, pos, feature, value
    resid.npz            per layer 0-12: mean [13, 768], cov [13, 768, 768], total variance and effective dimension
    tokens.npy           the token ids [rows, seq_len] that (row, pos) point into

Query the files with DuckDB, which reads only the columns and rows a query needs (the events are 71M rows), then
continue in pandas with `.df()`:

```python
import duckdb
q = lambda s: duckdb.sql(s).show(max_width=160)

# One unit's top contexts
q("select rank, value, context from 'results/atlas/topk.parquet' where kind='post' and layer=9 and unit=1000 order by rank")
# A residual dimension's most negative contexts (bottom-k)
q("select rank, value, context from 'results/atlas/topk.parquet' where kind='resid' and layer=9 and unit=447 and side='bottom' order by rank")
# Across units: the rarest-firing live neurons
q("select layer, unit, frac_pos, max from 'results/atlas/neurons.parquet' where kind='post' and frac_pos > 0 order by frac_pos limit 10")
# By token: the neurons whose top contexts most often land on ' John'
q("select layer, unit, count(*) n from 'results/atlas/topk.parquet' where kind='post' and token=' John' group by all order by n desc, layer, unit limit 10")
# SAE features: the densest at layer 9
q("select unit as feature, frac_pos from 'results/atlas-sae/neurons.parquet' where kind='sae' and layer=9 order by frac_pos desc limit 10")
# Per token, from the sparse events: the layer-9 features active at row 100, position 50
q("select feature, value from 'results/atlas-sae/events-L9-*.parquet' where row=100 and pos=50 order by value desc")

# Into pandas
neurons = duckdb.sql("select * from 'results/atlas/neurons.parquet' where kind='post'").df()
```

```python
import numpy as np
resid = np.load("results/atlas/resid.npz")
spectrum = np.linalg.eigvalsh(resid["cov"][9])[::-1]  # PCA of the residual stream before layer 9
```

The atlas answers only from what it stores: summaries and the top 20 per unit (plus, for SAE features at layer 9,
every activation). Anything else, such as a unit's activation on every token of a row, means rerunning that row from
`tokens.npy` through the model.


## Layout

    python/atlas.py             tokenize, model backends, mergeable accumulators, bootstrap, export (CLI: python -m atlas)
    notebooks/atlas.ipynb       exploration: residual geometry, neuron statistics, errors, single neurons
    scripts/run-atlas           pipeline end to end
    scripts/run-notebook        headless notebook execution, in place
    python/sae_gate.py          the gate into stage 2: SAELens SAEs on our pipeline (CLI: python -m sae_gate)
    scripts/run-sae-gate        the gate end to end
    python/ioi.py               the IOI circuit: prompts, head ablations, the logit difference in SAE features
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

