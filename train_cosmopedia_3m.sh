#!/bin/bash

# Exit on error
set -e

OUTDIR=out_cosmo
VOCAB_SIZE=16384

# Make sure the HuggingFace datasets library is installed
pip install datasets --quiet

echo "--- Step 1: Data Preparation ---"
# Download the Cosmopedia subset (1.5M stories, 500k web samples)
python download_cosmopedia.py

# Train a custom vocabulary
python tinystories.py train_vocab --vocab_size=$VOCAB_SIZE

# Pretokenize the dataset with the new vocabulary
python tinystories.py pretokenize --vocab_size=$VOCAB_SIZE

echo "--- Step 2: Training (approx 3.4M parameters) ---"
python train.py \
    --dim=192 \
    --n_layers=6 \
    --n_heads=6 \
    --vocab_source=custom \
    --vocab_size=$VOCAB_SIZE \
    --batch_size=64 \
    --max_iters=5000 \
    --eval_interval=1000 \
    --eval_iters=100 \
    --out_dir=$OUTDIR

echo "--- Step 3: Export and Quantization ---"
mkdir -p $OUTDIR

# Convert the PyTorch checkpoint to an int8 quantized binary inside $OUTDIR
python export.py $OUTDIR/cosmopedia_3M_q80.bin --version 2 --checkpoint out_cosmo/ckpt.pt

# Convert the sentencepiece model to binary format
python tokenizer.py --tokenizer-model=data/tok$VOCAB_SIZE.model

# Move and rename the tokenizer binary
mv data/tok$VOCAB_SIZE.bin $OUTDIR/cosmopedia_$VOCAB_SIZE.bin

echo "--- Step 4: Build and Run Inference ---"
make clean && make run
#make runomp

# Run the model using the descriptive filenames
echo "Running sample inference..."
./runq $OUTDIR/cosmopedia_3M_q80.bin -z $OUTDIR/cosmopedia_$VOCAB_SIZE.bin -t 1.0 -p 0.9 -n 100 -i "To bake a cake, you need"
#OMP_NUM_THREADS=32 ./runq ....
