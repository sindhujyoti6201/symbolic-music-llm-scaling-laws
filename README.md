# Symbolic Music LLM Scaling Laws

Research project investigating scaling laws for Large Language Models (LLMs) applied to symbolic music representation.

## Project Structure

This project follows a modular structure designed for machine learning research:

```
symbolic-music-llm-scaling-laws/
├── data/              # Raw and processed symbolic music data
├── src/               # Source code (models, training, evaluation)
├── configs/           # Configuration files (YAML)
├── experiments/       # Experiment outputs and results
├── notebooks/         # Jupyter notebooks for analysis
├── tests/             # Unit tests
├── docs/              # Documentation
└── scripts/           # Shell scripts for automation
```

For detailed structure information, see:
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Complete project structure documentation
- **[STRUCTURE_TREE.txt](STRUCTURE_TREE.txt)** - Visual tree representation

## Quick Start

### Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables (see `.env.example`)

### Data Preparation

1. Place raw symbolic music data in `data/raw/`
2. Run preprocessing:
   ```bash
   python src/scripts/preprocess_data.py
   ```

### Training

```bash
python src/scripts/train.py --config configs/training/default.yaml
```

### Evaluation

```bash
python src/scripts/evaluate.py --checkpoint experiments/runs/[run_name]/checkpoints/best.pt
```

## Project Components

### Data Pipeline
- **Formats**: MIDI, MusicXML, ABC notation
- **Processing**: Tokenization, normalization, sequence preparation
- **Splits**: Train/validation/test splits

### Models
- Transformer-based architectures
- Various model sizes for scaling experiments
- Custom embeddings for symbolic music

### Experiments
- Scaling law analysis (model size, data size, compute)
- Ablation studies
- Baseline comparisons

## Documentation

- Project structure: `PROJECT_STRUCTURE.md`
- Visual tree: `STRUCTURE_TREE.txt`
- Project PDF: `project-pdf/cs_gy_6923_project.pdf`

## License

[Add license information]

## Authors

[Add author information]

