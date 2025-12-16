#!/bin/bash

###############################################################################
# Symbolic Music LLM Scaling Laws - Complete Pipeline Script
#
# This script runs the entire project pipeline in sequence:
# 1. Data Collection (download Lakh MIDI Dataset)
# 2. Data Processing (MIDI to ABC, tokenization, splitting)
# 3. Transformer Scaling Study
# 4. RNN Scaling Study
# 5. Best Model Training and Sample Generation
# 6. Analysis and Plotting
#
# Usage:
#   ./run_pipeline.sh [options]
#
# Options:
#   --skip-data-collection    Skip downloading dataset (use existing)
#   --skip-data-processing    Skip data processing (use existing processed data)
#   --skip-scaling            Skip scaling studies
#   --skip-samples             Skip sample generation
#   --skip-analysis            Skip analysis and plotting
#   --max-files N              Limit number of MIDI files to process (default: 1000)
#   --num-workers N            Number of parallel workers (default: 14)
#   --num-epochs N              Number of epochs for scaling study (default: 1)
#   --best-model-epochs N      Number of epochs for best model (default: 3)
#   --num-samples N             Number of samples to generate (default: 10)
#   --help                      Show this help message
###############################################################################

set -e  # Exit on error

# Default configuration
SKIP_DATA_COLLECTION=false
SKIP_DATA_PROCESSING=false
SKIP_SCALING=false
SKIP_SAMPLES=false
SKIP_ANALYSIS=false
MAX_FILES=1000
NUM_WORKERS=14
NUM_EPOCHS=1
BEST_MODEL_EPOCHS=3
NUM_SAMPLES=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-data-collection)
            SKIP_DATA_COLLECTION=true
            shift
            ;;
        --skip-data-processing)
            SKIP_DATA_PROCESSING=true
            shift
            ;;
        --skip-scaling)
            SKIP_SCALING=true
            shift
            ;;
        --skip-samples)
            SKIP_SAMPLES=true
            shift
            ;;
        --skip-analysis)
            SKIP_ANALYSIS=true
            shift
            ;;
        --max-files)
            MAX_FILES="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --num-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --best-model-epochs)
            BEST_MODEL_EPOCHS="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --help)
            head -n 30 "$0" | tail -n +3
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print header
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Symbolic Music LLM Scaling Laws - Complete Pipeline${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}Warning: Virtual environment not detected.${NC}"
    echo "It's recommended to activate your virtual environment first:"
    echo "  source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}Python version: ${PYTHON_VERSION}${NC}"
echo ""

# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || {
    echo -e "${YELLOW}Warning: PyTorch not found or CUDA check failed${NC}"
}

# Set directories
DATA_DIR="data"
PROCESSED_DIR="${DATA_DIR}/processed"
OUTPUT_DIR="outputs"
SCALING_OUTPUT="${OUTPUT_DIR}/scaling_study"
SAMPLES_OUTPUT="${OUTPUT_DIR}/samples"
ANALYSIS_OUTPUT="${OUTPUT_DIR}/analysis"

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${ANALYSIS_OUTPUT}"

###############################################################################
# Step 1: Data Collection
###############################################################################
if [ "$SKIP_DATA_COLLECTION" = false ]; then
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}Step 1: Data Collection${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
    
    if [ -d "${DATA_DIR}/lmd_matched" ] && [ "$(ls -A ${DATA_DIR}/lmd_matched 2>/dev/null)" ]; then
        echo -e "${GREEN}✓ LMD dataset already exists${NC}"
        echo "  Location: ${DATA_DIR}/lmd_matched"
        MIDI_COUNT=$(find "${DATA_DIR}/lmd_matched" -name "*.mid" | wc -l)
        echo "  Found ${MIDI_COUNT} MIDI files"
        echo ""
        read -p "Re-download dataset? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python data-collection-and-preprocessing/data_collection.py
        fi
    else
        python data-collection-and-preprocessing/data_collection.py
    fi
    
    echo ""
    echo -e "${GREEN}✓ Step 1 Complete${NC}"
    echo ""
else
    echo -e "${YELLOW}⏭ Skipping data collection${NC}"
    echo ""
fi

###############################################################################
# Step 2: Data Processing
###############################################################################
if [ "$SKIP_DATA_PROCESSING" = false ]; then
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}Step 2: Data Processing${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
    
    if [ -f "${PROCESSED_DIR}/tokenizer.pkl" ] && [ -d "${PROCESSED_DIR}/tokenized" ]; then
        echo -e "${GREEN}✓ Processed data already exists${NC}"
        echo "  Location: ${PROCESSED_DIR}"
        echo ""
        read -p "Re-process data? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            SKIP_DATA_PROCESSING=true
        fi
    fi
    
    if [ "$SKIP_DATA_PROCESSING" = false ]; then
        echo "Processing MIDI files to ABC notation and tokenizing..."
        echo "  Max files: ${MAX_FILES}"
        echo "  Workers: ${NUM_WORKERS}"
        echo ""
        
        python data-collection-and-preprocessing/process_pipeline.py \
            --midi-dir "${DATA_DIR}/lmd_matched" \
            --output-dir "${PROCESSED_DIR}" \
            --max-files "${MAX_FILES}" \
            --num-workers "${NUM_WORKERS}" \
            --chunk-size 500 \
            --min-freq 2 \
            --min-length 10 \
            --max-length 5000
        
        echo ""
        echo -e "${GREEN}✓ Step 2 Complete${NC}"
        echo ""
    fi
else
    echo -e "${YELLOW}⏭ Skipping data processing${NC}"
    echo ""
fi

# Verify processed data exists
if [ ! -f "${PROCESSED_DIR}/tokenizer.pkl" ]; then
    echo -e "${RED}Error: Processed data not found!${NC}"
    echo "Please run data processing first or check paths."
    exit 1
fi

###############################################################################
# Step 3: Transformer Scaling Study
###############################################################################
if [ "$SKIP_SCALING" = false ]; then
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}Step 3: Scaling Studies${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
    
    echo "Training transformer and RNN models of varying sizes..."
    echo "  Epochs per model: ${NUM_EPOCHS}"
    echo "  Output directory: ${SCALING_OUTPUT}"
    echo ""
    
    python experiments/scaling_study.py \
        --data-dir "${PROCESSED_DIR}" \
        --output-dir "${SCALING_OUTPUT}" \
        --architecture both \
        --num-epochs "${NUM_EPOCHS}" \
        --batch-size-tokens 50000 \
        --learning-rate 3e-4
    
    echo ""
    echo -e "${GREEN}✓ Step 3 Complete${NC}"
    echo ""
else
    echo -e "${YELLOW}⏭ Skipping scaling studies${NC}"
    echo ""
fi

###############################################################################
# Step 4: Sample Generation
###############################################################################
if [ "$SKIP_SAMPLES" = false ]; then
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}Step 4: Best Model Training and Sample Generation${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
    
    echo "Training best (XL) model and generating samples..."
    echo "  Training epochs: ${BEST_MODEL_EPOCHS}"
    echo "  Number of samples: ${NUM_SAMPLES}"
    echo "  Output directory: ${SAMPLES_OUTPUT}"
    echo ""
    
    python experiments/sample_generation.py \
        --data-dir "${PROCESSED_DIR}" \
        --output-dir "${SAMPLES_OUTPUT}" \
        --model-name xl \
        --num-epochs "${BEST_MODEL_EPOCHS}" \
        --num-samples "${NUM_SAMPLES}" \
        --max-new-tokens 500 \
        --batch-size-tokens 50000 \
        --learning-rate 3e-4
    
    echo ""
    echo -e "${GREEN}✓ Step 4 Complete${NC}"
    echo ""
else
    echo -e "${YELLOW}⏭ Skipping sample generation${NC}"
    echo ""
fi

###############################################################################
# Step 5: Analysis and Plotting
###############################################################################
if [ "$SKIP_ANALYSIS" = false ]; then
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}Step 5: Analysis and Plotting${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
    
    if [ ! -f "${SCALING_OUTPUT}/scaling_results.json" ]; then
        echo -e "${YELLOW}Warning: Scaling results not found. Skipping analysis.${NC}"
        echo "  Expected: ${SCALING_OUTPUT}/scaling_results.json"
    else
        echo "Generating scaling plots and power-law fits..."
        echo "  Results directory: ${SCALING_OUTPUT}"
        echo "  Output directory: ${ANALYSIS_OUTPUT}"
        echo ""
        
        python analysis/plot_scaling.py \
            --results-dir "${SCALING_OUTPUT}" \
            --output-dir "${ANALYSIS_OUTPUT}" || {
            echo -e "${YELLOW}Warning: Analysis script failed or not implemented${NC}"
        }
        
        echo ""
        echo -e "${GREEN}✓ Step 5 Complete${NC}"
        echo ""
    fi
else
    echo -e "${YELLOW}⏭ Skipping analysis${NC}"
    echo ""
fi

###############################################################################
# Summary
###############################################################################
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Pipeline Complete!${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "${GREEN}Output Summary:${NC}"
echo ""

# Check and report outputs
if [ -f "${PROCESSED_DIR}/tokenizer.pkl" ]; then
    echo -e "  ${GREEN}✓${NC} Processed data: ${PROCESSED_DIR}"
fi

if [ -f "${SCALING_OUTPUT}/scaling_results.json" ]; then
    echo -e "  ${GREEN}✓${NC} Scaling results: ${SCALING_OUTPUT}/scaling_results.json"
fi

if [ -f "${SAMPLES_OUTPUT}/best_model.pt" ]; then
    echo -e "  ${GREEN}✓${NC} Best model: ${SAMPLES_OUTPUT}/best_model.pt"
fi

if [ -d "${SAMPLES_OUTPUT}/samples" ]; then
    SAMPLE_COUNT=$(find "${SAMPLES_OUTPUT}/samples" -name "sample_*.abc" | wc -l)
    echo -e "  ${GREEN}✓${NC} Generated samples: ${SAMPLE_COUNT} files in ${SAMPLES_OUTPUT}/samples"
fi

if [ -f "${ANALYSIS_OUTPUT}/scaling_plots.png" ]; then
    echo -e "  ${GREEN}✓${NC} Scaling plots: ${ANALYSIS_OUTPUT}/scaling_plots.png"
fi

echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Review scaling results: ${SCALING_OUTPUT}/scaling_results.json"
echo "  2. Check generated samples: ${SAMPLES_OUTPUT}/samples/"
echo "  3. View scaling plots: ${ANALYSIS_OUTPUT}/scaling_plots.png"
echo "  4. Analyze results and write report"
echo ""
echo -e "${GREEN}Done! 🎵📈${NC}"

