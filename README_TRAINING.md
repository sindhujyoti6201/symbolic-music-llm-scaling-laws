# Transformer Scaling Study - Training Guide

## Overview

This directory contains the implementation for training a family of decoder-only transformer models of varying sizes to study scaling laws for symbolic music generation.

## Project Structure

```
.
├── models/
│   ├── __init__.py
│   └── transformer.py          # Decoder-only transformer architecture
├── utils/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading utilities
│   └── metrics.py              # Training metrics tracking
├── configs/
│   ├── tiny.yaml               # ~1M parameters
│   ├── small.yaml              # ~5M parameters
│   ├── medium.yaml             # ~20M parameters
│   ├── large.yaml              # ~50M parameters
│   └── xl.yaml                 # ~100M+ parameters
├── experiments/
│   └── run_scaling_study.py    # Run all model sizes
├── analysis/
│   └── plot_scaling.py         # Generate scaling plots
└── train.py                    # Main training script
```

## Quick Start

### 1. Train a Single Model

```bash
python train.py --config configs/tiny.yaml
```

### 2. Run Full Scaling Study

Train all model sizes sequentially:

```bash
python experiments/run_scaling_study.py
```

### 3. Generate Analysis Plots

After training, generate scaling plots and analysis:

```bash
python analysis/plot_scaling.py --results-dir outputs/scaling_study
```

## Model Configurations

The project includes 5 pre-configured model sizes:

| Model | Parameters | d_model | n_layers | n_heads | d_ff |
|-------|------------|---------|----------|---------|------|
| Tiny  | ~1M        | 128     | 2        | 2       | 512  |
| Small | ~5M        | 256     | 4        | 4       | 1024 |
| Medium| ~20M       | 512     | 6        | 8       | 2048 |
| Large | ~50M       | 768     | 8        | 8       | 3072 |
| XL    | ~100M+     | 1024    | 12       | 12      | 4096 |

## Training Parameters

All models use **consistent** training setup:

- **Batch size**: 512K tokens (configurable via `--batch-size-tokens`)
- **Learning rate**: 3e-4 (configurable via `--learning-rate`)
- **Learning rate schedule**: Cosine annealing
- **Optimizer**: AdamW with weight decay 0.01
- **Gradient clipping**: 1.0
- **Epochs**: 1 (for scaling study comparison)
- **Max sequence length**: 5000 tokens

## Data Format

The training script expects tokenized data in JSON format:

```json
[
  {
    "midi_path": "path/to/file.mid",
    "abc": "X:1\nT:...",
    "tokens": [1, 2, 3, ...]
  },
  ...
]
```

Or JSONL format (one object per line):

```jsonl
{"midi_path": "...", "abc": "...", "tokens": [...]}
{"midi_path": "...", "abc": "...", "tokens": [...]}
```

## Output Structure

Training outputs are saved to `outputs/<model_name>/`:

```
outputs/
├── tiny/
│   ├── best_model.pt           # Best model checkpoint
│   ├── training_results.json   # Training results
│   └── metrics.json            # Detailed metrics
├── small/
│   └── ...
└── scaling_study/
    └── scaling_study_results.json  # Combined results
```

## Scaling Analysis

The analysis script generates:

1. **Scaling Plot**: Validation loss vs. model size (log scale) with power law fit
2. **Training Curves**: Training loss over time for all models
3. **Summary Table**: Architecture details and training statistics

Power law fit: `L = a · N^(-α) + c`

Where:
- `L` = validation loss
- `N` = number of parameters
- `α` = scaling exponent (reported in analysis)

## Requirements

All dependencies are in `requirements.txt`. Key packages:

- PyTorch >= 2.0.0
- NumPy, SciPy (for analysis)
- Matplotlib (for plotting)
- PyYAML (for configs)

## Notes

- **Data Requirement**: The project requires at least 100M training tokens. Currently you have ~10.5M tokens. Consider processing more MIDI files if needed.
- **GPU Memory**: Larger models (Large, XL) may require significant GPU memory. Adjust batch size if needed.
- **Training Time**: Training time scales roughly with model size. Start with Tiny/Small models to test the pipeline.

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size:
```bash
python train.py --config configs/large.yaml --batch-size-tokens 256000
```

### Data Loading Issues

Ensure your tokenized data is in the correct format. Check:
```bash
head -1 data/processed/tokenized/train/data.json
```

### Import Errors

Make sure you're in the project root directory and all dependencies are installed:
```bash
pip install -r requirements.txt
```


