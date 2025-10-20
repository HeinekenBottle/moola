#!/bin/bash
# Sync files between local machine and RunPod
# Usage: bash scripts/sync_runpod.sh [upload|download]

RUNPOD_HOST="root@213.173.102.99"
RUNPOD_PORT="27424"
RUNPOD_KEY="~/.ssh/id_ed25519"
REMOTE_DIR="/workspace/moola"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to upload files to RunPod
upload() {
    echo -e "${GREEN}[UPLOAD] Syncing files to RunPod...${NC}\n"

    # Create remote directories
    echo "Creating remote directories..."
    ssh -p $RUNPOD_PORT -i $RUNPOD_KEY $RUNPOD_HOST "mkdir -p $REMOTE_DIR/{data/{processed,oof,raw},src,scripts,configs,models/ts_tcc}"

    # Upload source code
    echo -e "\n${YELLOW}Uploading source code...${NC}"
    scp -P $RUNPOD_PORT -i $RUNPOD_KEY -r src/ $RUNPOD_HOST:$REMOTE_DIR/

    # Upload scripts
    echo -e "\n${YELLOW}Uploading scripts...${NC}"
    scp -P $RUNPOD_PORT -i $RUNPOD_KEY scripts/runpod_*.sh $RUNPOD_HOST:$REMOTE_DIR/scripts/
    scp -P $RUNPOD_PORT -i $RUNPOD_KEY scripts/regenerate_oof_phase2.py $RUNPOD_HOST:$REMOTE_DIR/scripts/

    # Upload configs
    echo -e "\n${YELLOW}Uploading configs...${NC}"
    scp -P $RUNPOD_PORT -i $RUNPOD_KEY configs/*.yaml $RUNPOD_HOST:$REMOTE_DIR/configs/ 2>/dev/null || echo "No config files found"

    # Upload training data
    echo -e "\n${YELLOW}Uploading training data...${NC}"
    scp -P $RUNPOD_PORT -i $RUNPOD_KEY data/processed/train_clean.parquet $RUNPOD_HOST:$REMOTE_DIR/data/processed/

    # Upload existing OOF files (if any)
    echo -e "\n${YELLOW}Uploading existing OOF predictions...${NC}"
    scp -P $RUNPOD_PORT -i $RUNPOD_KEY data/oof/*.npy $RUNPOD_HOST:$REMOTE_DIR/data/oof/ 2>/dev/null || echo "No OOF files to upload yet"

    # Upload unlabeled data (if exists)
    if [ -f "data/raw/unlabeled_windows.parquet" ]; then
        echo -e "\n${YELLOW}Uploading unlabeled data for TS-TCC...${NC}"
        scp -P $RUNPOD_PORT -i $RUNPOD_KEY data/raw/unlabeled_windows.parquet $RUNPOD_HOST:$REMOTE_DIR/data/raw/
    else
        echo -e "\n${YELLOW}⚠️  No unlabeled data found - TS-TCC pre-training will be skipped${NC}"
    fi

    # Upload cleaned data (if exists from CleanLab)
    if [ -f "data/processed/train_clean_v2.parquet" ]; then
        echo -e "\n${YELLOW}Uploading CleanLab-cleaned data...${NC}"
        scp -P $RUNPOD_PORT -i $RUNPOD_KEY data/processed/train_clean_v2.parquet $RUNPOD_HOST:$REMOTE_DIR/data/processed/
    fi

    echo -e "\n${GREEN}✅ Upload complete!${NC}\n"
    echo "Next steps:"
    echo "  ssh -p $RUNPOD_PORT -i $RUNPOD_KEY $RUNPOD_HOST"
    echo "  cd $REMOTE_DIR"
    echo "  bash scripts/runpod_setup.sh"
    echo "  bash scripts/runpod_train.sh"
}

# Function to download results from RunPod
download() {
    echo -e "${GREEN}[DOWNLOAD] Fetching results from RunPod...${NC}\n"

    # Create local directories
    mkdir -p data/oof models/ts_tcc

    # Download OOF predictions
    echo -e "${YELLOW}Downloading OOF predictions...${NC}"
    scp -P $RUNPOD_PORT -i $RUNPOD_KEY $RUNPOD_HOST:$REMOTE_DIR/data/oof/*_clean.npy data/oof/ 2>/dev/null || echo "No OOF files found"

    # Download pre-trained encoder
    echo -e "\n${YELLOW}Downloading TS-TCC pre-trained encoder...${NC}"
    scp -P $RUNPOD_PORT -i $RUNPOD_KEY $RUNPOD_HOST:$REMOTE_DIR/models/ts_tcc/pretrained_encoder.pt models/ts_tcc/ 2>/dev/null || echo "No pre-trained encoder found"

    # Download any trained models
    echo -e "\n${YELLOW}Downloading trained models...${NC}"
    scp -P $RUNPOD_PORT -i $RUNPOD_KEY -r $RUNPOD_HOST:$REMOTE_DIR/models/stack/ models/ 2>/dev/null || echo "No stack models found"

    # Show what was downloaded
    echo -e "\n${GREEN}✅ Download complete!${NC}\n"
    echo "Downloaded files:"
    ls -lh data/oof/*.npy 2>/dev/null || echo "  No OOF files"
    ls -lh models/ts_tcc/*.pt 2>/dev/null || echo "  No pre-trained encoder"
    ls -lh models/stack/*.pkl 2>/dev/null || echo "  No stack models"
}

# Main
case "$1" in
    upload)
        upload
        ;;
    download)
        download
        ;;
    *)
        echo "Usage: $0 {upload|download}"
        echo ""
        echo "Examples:"
        echo "  bash scripts/sync_runpod.sh upload     # Upload code and data to RunPod"
        echo "  bash scripts/sync_runpod.sh download   # Download results from RunPod"
        exit 1
        ;;
esac
