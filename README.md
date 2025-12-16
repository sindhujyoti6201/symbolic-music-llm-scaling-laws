# Symbolic Music LLM Scaling Laws

A research project that studies how well language models work for generating music as they get bigger. We train different sizes of models (from 1 million to 100+ million parameters) and measure how their performance changes. This helps us understand the relationship between model size and performance.

## What This Project Does

This project answers these questions:
1. How does model performance change as we make models bigger?
2. Do transformer models and RNN models scale differently?
3. What's the best model size for generating music?
4. What kind of music patterns do different sized models learn?

We measure this by training models and finding a mathematical relationship: **L = a · N^(-α) + c**

Where:
- **L** = How well the model performs (lower is better)
- **N** = Number of parameters in the model
- **α** = The scaling exponent (this is what we're trying to find)
- **a, c** = Constants we calculate

## Project Goals

1. **Build a data processing pipeline** - Convert MIDI music files into a format models can learn from
2. **Train models of different sizes** - From tiny (1M parameters) to extra large (100M+ parameters)
3. **Compare transformers and RNNs** - See which architecture scales better
4. **Generate music samples** - Create new music from trained models
5. **Analyze results** - Create plots and understand the scaling laws

## How It Works

### The Data

- **Lakh MIDI Dataset**: About 116,000 MIDI music files
- **Processing**: Convert MIDI to ABC notation (a text format for music), then break it into tokens
- **Splits**: 98% for training, 1% for validation, 1% for testing

### The Models

We train two types of models:

**Transformers** (like GPT):
- Tiny: ~1M parameters
- Small: ~5M parameters
- Medium: ~20M parameters
- Large: ~50M parameters
- XL: ~100M+ parameters

**RNNs** (LSTM):
- Tiny: ~1M parameters
- Small: ~5M parameters
- Medium: ~20M parameters
- Large: ~50M parameters

### Training Setup

All models use the same training settings:
- Learning rate: 3e-4
- Batch size: 50,000 tokens
- Optimizer: AdamW
- Training: 1 epoch (for comparing scaling)

### How Music is Represented

Music is converted to tokens like:
- Notes: `C`, `D`, `E`, `F`, `G`, `A`, `B` (with sharps `^` and flats `_`)
- Durations: `DUR:1`, `DUR:2`, etc.
- Rests: `z`
- Bar lines: `|`
- Special tokens: `<PAD>`, `<UNK>`, `<START>`, `<END>`, `<SEP>`

## Project Structure

```
symbolic-music-llm-scaling-laws/
├── data-collection-and-preprocessing/
│   ├── data_collection.py          # Downloads the music dataset
│   ├── midi_to_abc.py              # Converts MIDI to ABC notation
│   ├── tokenizer.py                # Breaks music into tokens
│   └── process_pipeline.py         # Complete data processing
├── models/
│   ├── transformer.py              # Transformer model (GPT-style)
│   └── rnn.py                      # LSTM model
├── utils/
│   ├── data_loader.py              # Loads data for training
│   ├── training.py                 # Training functions
│   └── metrics.py                  # Tracks training metrics
├── experiments/
│   ├── scaling_study.py            # Trains all model sizes
│   └── sample_generation.py         # Generates music samples
├── analysis/
│   └── plot_scaling.py             # Creates scaling plots
├── configs/
│   ├── tiny.yaml                   # Model size configurations
│   ├── small.yaml
│   ├── medium.yaml
│   ├── large.yaml
│   └── xl.yaml
├── data/
│   ├── lmd_matched/                # Raw MIDI files
│   └── processed/                  # Processed data
│       ├── abc/                    # ABC notation files
│       ├── tokenized/               # Tokenized sequences
│       └── tokenizer.pkl           # Saved tokenizer
├── outputs/
│   ├── scaling_study/              # Scaling study results
│   └── samples/                    # Generated music samples
├── requirements.txt                # Python packages needed
├── run_pipeline.sh                 # Automated script to run everything
└── README.md                       # This file
```

## Installation

### What You Need

- Python 3.8 or newer
- A GPU with CUDA (recommended, but CPU works for small models)
- About 5GB space for the dataset
- About 10GB space for processed data and results

### Setup Steps

1. **Clone the repository** (if you have it):
   ```bash
   git clone <repository-url>
   cd symbolic-music-llm-scaling-laws
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: If you have trouble installing `tables`, you may need to install HDF5:
   ```bash
   # macOS
   brew install hdf5
   
   # Ubuntu/Debian
   sudo apt-get install libhdf5-dev
   ```

4. **Check that everything works**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Quick Start

### Run Everything Automatically

The easiest way is to use the automated script:

```bash
# Make the script executable
chmod +x run_pipeline.sh

# Run everything
./run_pipeline.sh
```

This will:
1. Download the music dataset
2. Process MIDI files to ABC notation
3. Build vocabulary and tokenize
4. Train transformer models of different sizes
5. Train RNN models of different sizes
6. Train the best model and generate music samples
7. Create analysis plots

### Run Steps Manually

If you want to run each step separately:

#### Step 1: Download the Dataset
```bash
python data-collection-and-preprocessing/data_collection.py
```

#### Step 2: Process the Data
```bash
python data-collection-and-preprocessing/process_pipeline.py \
    --midi-dir data/lmd_matched \
    --output-dir data/processed \
    --max-files 1000 \
    --num-workers 14
```

#### Step 3: Run Scaling Study
```bash
# Train both transformer and RNN models
python experiments/scaling_study.py \
    --data-dir data/processed \
    --output-dir outputs/scaling_study \
    --architecture both \
    --num-epochs 1

# Or train only transformers
python experiments/scaling_study.py --architecture transformer

# Or train only RNNs
python experiments/scaling_study.py --architecture rnn
```

#### Step 4: Generate Music Samples
```bash
python experiments/sample_generation.py \
    --data-dir data/processed \
    --output-dir outputs/samples \
    --model-name xl \
    --num-epochs 3 \
    --num-samples 10
```

#### Step 5: Create Analysis Plots
```bash
python analysis/plot_scaling.py \
    --results-dir outputs/scaling_study
```

## Detailed Usage Guide

### Data Collection

Downloads and extracts the Lakh MIDI Dataset:

```bash
python data-collection-and-preprocessing/data_collection.py
```

**What you get**:
- `data/lmd_matched/` - All the MIDI files
- `data/match_scores.json` - Metadata about the files

**Time**: About 5-15 minutes to download, then 5-10 minutes to extract

### Data Processing

Converts MIDI files to ABC notation, tokenizes them, and creates train/validation/test splits:

```bash
python data-collection-and-preprocessing/process_pipeline.py \
    --midi-dir data/lmd_matched \
    --output-dir data/processed \
    --max-files 1000 \
    --num-workers 14 \
    --chunk-size 500 \
    --min-freq 2 \
    --min-length 10 \
    --max-length 5000
```

**Options**:
- `--midi-dir`: Where the MIDI files are
- `--output-dir`: Where to save processed data
- `--max-files`: How many files to process (default: all)
- `--num-workers`: How many parallel workers to use (default: 14)
- `--chunk-size`: How many files to process at once (default: 500)
- `--min-freq`: Minimum times a token must appear (default: 2)
- `--min-length`: Shortest sequence to keep (default: 10)
- `--max-length`: Longest sequence to keep (default: 5000)

**What you get**:
- `data/processed/abc/` - ABC notation files
- `data/processed/tokenized/` - Tokenized sequences (train/val/test)
- `data/processed/tokenizer.pkl` - The tokenizer (for later use)
- `data/processed/statistics.json` - Statistics about the data

**Time**: About 3-6 minutes for 1,000 files with 14 workers

### Scaling Study

Trains models of different sizes to study how performance scales:

```bash
python experiments/scaling_study.py \
    --data-dir data/processed \
    --output-dir outputs/scaling_study \
    --architecture both \
    --num-epochs 1 \
    --batch-size-tokens 50000 \
    --learning-rate 3e-4
```

**Options**:
- `--data-dir`: Where the processed data is
- `--output-dir`: Where to save results
- `--architecture`: `transformer`, `rnn`, or `both` (default: `both`)
- `--num-epochs`: How many training epochs (default: 1)
- `--batch-size-tokens`: Batch size in tokens (default: 50000)
- `--learning-rate`: Learning rate (default: 3e-4)

**What you get**:
- `outputs/scaling_study/scaling_results.json` - Results for all models
- Model checkpoints (if saved)

**Time**: About 1-5 hours (depends on your GPU and how many models)

### Sample Generation

Trains the best model and generates music samples:

```bash
python experiments/sample_generation.py \
    --data-dir data/processed \
    --output-dir outputs/samples \
    --model-name xl \
    --num-epochs 3 \
    --num-samples 10 \
    --max-new-tokens 500 \
    --batch-size-tokens 50000 \
    --learning-rate 3e-4
```

**Options**:
- `--data-dir`: Where the processed data is
- `--output-dir`: Where to save samples
- `--model-name`: Model size (`tiny`, `small`, `medium`, `large`, `xl`)
- `--num-epochs`: How many training epochs (default: 3)
- `--num-samples`: How many samples to generate (default: 10)
- `--max-new-tokens`: Maximum tokens per sample (default: 500)
- `--batch-size-tokens`: Batch size in tokens (default: 50000)
- `--learning-rate`: Learning rate (default: 3e-4)

**What you get**:
- `outputs/samples/best_model.pt` - The trained model
- `outputs/samples/samples/` - Generated ABC files
- `outputs/samples/samples/samples_metadata.json` - Information about samples
- `outputs/samples/evaluation_results.json` - Quality metrics

**Time**: About 2-4 hours for XL model with 3 epochs

### Analysis

Creates plots showing scaling laws:

```bash
python analysis/plot_scaling.py \
    --results-dir outputs/scaling_study \
    --output-dir outputs/analysis
```

**What you get**:
- `outputs/analysis/scaling_plots.png` - Plots showing scaling laws
- Power-law fit parameters and statistics

## Model Configurations

The project includes 5 pre-configured model sizes:

| Model | Parameters | d_model | Layers | Heads | d_ff |
|-------|------------|---------|--------|-------|------|
| Tiny  | ~1M        | 128     | 2      | 2     | 512  |
| Small | ~5M        | 256     | 4      | 4     | 1024 |
| Medium| ~20M       | 512     | 6      | 8     | 2048 |
| Large | ~50M       | 768     | 8      | 8     | 3072 |
| XL    | ~100M+     | 1024    | 12     | 12    | 4096 |

All models use the same training settings:
- Batch size: 50,000 tokens
- Learning rate: 3e-4
- Learning rate schedule: Cosine annealing
- Optimizer: AdamW with weight decay 0.01
- Gradient clipping: 1.0
- Epochs: 1 (for scaling study comparison)

## Code Organization

### Data Collection and Preprocessing

- **`data_collection.py`**: Downloads and extracts the Lakh MIDI Dataset
- **`midi_to_abc.py`**: Converts MIDI files to ABC notation using music21
- **`tokenizer.py`**: Creates a tokenizer that understands music notation
- **`process_pipeline.py`**: Runs the complete pipeline: convert MIDI, tokenize, filter, and split data

### Models

- **`transformer.py`**: Decoder-only transformer model (like GPT)
  - Multi-head causal self-attention
  - Feedforward networks with GELU activation
  - Layer normalization and residual connections
  - Can generate music samples

- **`rnn.py`**: LSTM-based model for comparison
  - Embedding layer
  - Multi-layer LSTM
  - Language modeling head
  - Can generate music samples

### Utilities

- **`data_loader.py`**: Loads tokenized data for training
  - Handles variable-length sequences
  - Batches by token count (not sequence count)
  - Supports JSON and JSONL formats

- **`training.py`**: Training functions
  - Learning rate scheduling (cosine annealing)
  - Training loop with gradient clipping
  - Validation evaluation

- **`metrics.py`**: Tracks training metrics
  - Loss, learning rate, GPU memory, training time

### Experiments

- **`scaling_study.py`**: Trains multiple model sizes
  - Configurations: tiny, small, medium, large, xl
  - Same hyperparameters for all models
  - Collects metrics: parameters, train/val loss, training time, GPU memory
  - Saves results to JSON

- **`sample_generation.py`**: Trains the best model and generates samples
  - Trains largest model for multiple epochs
  - Generates unconditional music samples
  - Evaluates sample quality
  - Saves samples as ABC files

### Analysis

- **`plot_scaling.py`**: Creates scaling plots
  - Validation loss vs. model size (log scale)
  - Fits power law: L = a · N^(-α) + c
  - Compares transformer vs. RNN scaling
  - Creates publication-ready plots

## Expected Outputs

### Data Processing

- **ABC Files**: `data/processed/abc/*.abc` - Music in ABC notation
- **Tokenized Data**: `data/processed/tokenized/{train,val,test}/data.json` - Tokenized sequences
- **Tokenizer**: `data/processed/tokenizer.pkl` - Saved tokenizer for reuse
- **Statistics**: `data/processed/statistics.json` - Dataset statistics

### Scaling Study

- **Results JSON**: `outputs/scaling_study/scaling_results.json` - Contains results for all models:
  ```json
  {
    "transformer": [
      {
        "model_name": "tiny",
        "architecture": "transformer",
        "num_parameters": 1234567,
        "train_loss": 2.345,
        "val_loss": 2.456,
        "training_time_seconds": 1234.5,
        "gpu_memory_mb": 1024.0,
        "config": {...}
      },
      ...
    ],
    "rnn": [...]
  }
  ```

### Sample Generation

- **Model Checkpoint**: `outputs/samples/best_model.pt` - Trained model
- **Generated Samples**: `outputs/samples/samples/sample_*.abc` - Music in ABC notation
- **Evaluation Results**: `outputs/samples/evaluation_results.json` - Quality metrics

### Analysis

- **Scaling Plots**: `outputs/analysis/scaling_plots.png` - Visualizations of scaling laws
- **Power-Law Fits**: Included in plots and results JSON

## Expected Results

After running everything, you should see:

1. **Scaling Laws**: A mathematical relationship between model size and performance
2. **Architecture Comparison**: Transformers and RNNs scale differently
3. **Computational Efficiency**: How long training takes and how much memory is needed
4. **Sample Quality**: Generated music that gets better as models get bigger

### What We Expect to Find

- **Transformers** usually perform better (lower loss) for the same number of parameters
- **Scaling exponent (α)** is usually between 0.1 and 0.3 for language models
- **Larger models** generate more coherent and musically structured samples
- **RNNs** may use less memory but don't perform as well at large scales

## Troubleshooting

### Common Problems

1. **Out of Memory**: Reduce `--batch-size-tokens` or `--max-files`
2. **Slow Processing**: Increase `--num-workers` (if CPU-bound) or use a GPU
3. **Import Errors**: Make sure your virtual environment is activated and all packages are installed
4. **MIDI Conversion Failures**: Some MIDI files may be corrupted; this is normal

### Getting Help

- Check error messages for specific issues
- Look at logs in output directories
- Make sure all dependencies are installed correctly
- Verify that data paths are correct

## Academic Context

This project is designed for:
- **CS-GY 6923**: Deep Learning course project
- **Research**: Scaling laws for symbolic music generation
- **Education**: Understanding transformer architectures and scaling behavior

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{symbolic-music-llm-scaling-laws,
  title={Symbolic Music LLM Scaling Laws},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/symbolic-music-llm-scaling-laws}
}
```

## License

[Add your license information here]

## Authors

[Add author information here]

---

**Happy Scaling! 🎵📈**
