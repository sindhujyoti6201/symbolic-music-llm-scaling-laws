# Symbolic Music LLM Scaling Laws

A research project investigating neural scaling laws for large language models applied to symbolic music generation. This project trains transformer and RNN models across multiple scales (1M to 100M+ parameters) on the Lakh MIDI Dataset to study how model performance scales with size.

## Overview

This project studies the power-law relationship between model size and performance: **L = a · N^(-α) + c**, where L is validation loss, N is the number of parameters, and α is the scaling exponent. We compare transformer and RNN architectures to understand which scales better for symbolic music generation.

## Key Findings

- **Power-law scaling confirmed**: Validation loss follows a predictable relationship with model size
- **Transformers outperform RNNs**: Better performance and scaling exponent (α = 0.142 vs. 0.118)
- **Predictable scaling**: Enables performance prediction for resource planning
- **Quality improves with scale**: Larger models generate more coherent musical sequences

## Complete Results

**All detailed experimental results, training curves, generated samples, scaling plots, and analysis are available in the Jupyter notebook:**

📓 [`symbolic_music_llm_complete.ipynb`](symbolic_music_llm_complete.ipynb)

The notebook contains:
- Complete data processing pipeline
- Model training code and configurations
- Scaling study results for all model sizes
- Power-law fits and scaling exponents
- Generated music samples
- Visualization plots and analysis

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Pipeline

```bash
# Make script executable
chmod +x run_pipeline.sh

# Run complete pipeline
./run_pipeline.sh
```

This will:
1. Download and process the Lakh MIDI Dataset
2. Train transformer and RNN models of different sizes
3. Generate music samples
4. Create scaling analysis plots

## Project Structure

```
symbolic-music-llm-scaling-laws/
├── symbolic_music_llm_complete.ipynb  # Complete notebook with all results
├── data-collection-and-preprocessing/  # Data processing scripts
├── models/                            # Model architectures
├── experiments/                       # Training and generation scripts
├── analysis/                          # Scaling analysis
├── project-pdf-and-report/            # Report and documentation
├── run_pipeline.sh                    # Automated pipeline script
└── requirements.txt                   # Python dependencies
```

## Model Configurations

We train models across five size categories:

| Model | Parameters | Architecture |
|-------|------------|--------------|
| Tiny  | ~1M        | Transformer, RNN |
| Small | ~5M        | Transformer, RNN |
| Medium| ~20M       | Transformer, RNN |
| Large | ~50M       | Transformer, RNN |
| XL    | ~100M+     | Transformer only |

## Dataset

- **Source**: Lakh MIDI Dataset (LMD-matched)
- **Processing**: Converted to ABC notation, tokenized with music-aware tokenizer
- **Size**: 1,000 MIDI files processed → 343 sequences after filtering
- **Vocabulary**: 159 tokens (5 special + 154 regular)
- **Split**: 98% train, 1% validation, 1% test

## Training Setup

- **Optimizer**: AdamW (lr=3e-4, weight decay=0.01)
- **Batch Size**: 50,000 tokens
- **Learning Rate**: Cosine annealing
- **Epochs**: 1 (scaling study), 3 (best model for generation)
- **Max Sequence Length**: 5,000 tokens

## Results Summary

- **Transformer scaling exponent**: α = 0.142 ± 0.018
- **RNN scaling exponent**: α = 0.118 ± 0.021
- **Best model**: XL transformer (test loss: 1.723, perplexity: 5.60)

For complete results, see the Jupyter notebook.

## Citation

```bibtex
@misc{symbolic-music-llm-scaling-laws,
  title={Symbolic Music LLM Scaling Laws},
  author={Sindhu Jyoti Dutta},
  year={2024},
  institution={New York University}
}
```

## Author

**Sindhu Jyoti Dutta**  
Machine Learning  
New York University

---

For complete experimental results, analysis, and visualizations, please refer to [`symbolic_music_llm_complete.ipynb`](symbolic_music_llm_complete.ipynb).
