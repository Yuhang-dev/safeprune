# SliceGPT Reproduction Commands

SliceGPT is the first reproduction target because it is an accepted ICLR 2024
structured compression baseline with mature code and small-model support.

Paper:

```text
SliceGPT: Compress Large Language Models by Deleting Rows and Columns
https://arxiv.org/abs/2401.15024
```

Code:

```text
https://github.com/microsoft/TransformerCompression
```

## 1. Setup

Keep the external repository outside this project, or under an ignored scratch
directory. Do not vendor it into `src/`.

```bash
git clone https://github.com/microsoft/TransformerCompression
cd TransformerCompression
pip install -e ".[experiment]"
cd experiments
```

## 2. Smoke Run

```bash
python run_slicegpt.py \
  --model facebook/opt-125m \
  --save-dir outputs/opt125m_sliced_010 \
  --sparsity 0.10 \
  --device cuda:0 \
  --eval-baseline \
  --no-wandb
```

## 3. Main Matrix

Models:

```text
facebook/opt-125m
facebook/opt-1.3b
facebook/opt-2.7b
microsoft/phi-2
```

Sparsities:

```text
0.10
0.20
0.25
0.30
```

Example:

```bash
python run_slicegpt.py \
  --model microsoft/phi-2 \
  --save-dir outputs/phi2_sliced_025 \
  --sparsity 0.25 \
  --device cuda:1 \
  --eval-baseline \
  --no-wandb
```

## 4. Downstream Eval

```bash
python run_lm_eval.py \
  --model microsoft/phi-2 \
  --sliced-model-path outputs/phi2_sliced_025 \
  --sparsity 0.25 \
  --tasks piqa \
  --no-wandb
```

## 5. Required Logging

Record one JSON row per run:

```json
{"model":"microsoft/phi-2","sparsity":0.25,"ppl":null,"piqa":null,"peak_mem_gb":null,"tokens_per_s":null,"model_size_gb":null}
```

The expected first milestone is trend reproduction, not exact table matching:

```text
higher sparsity -> lower model size / memory
higher sparsity -> worse perplexity
small models degrade faster than larger models
tokens/s only counts if wall-clock measurement is real
```
