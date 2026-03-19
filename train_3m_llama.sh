#!/bin/bash

# Exit on error
set -e

OUTDIR=out_tinystories
VOCABS=4096

echo "--- Step 1: Data Preparation ---"
# Download the TinyStories dataset
python tinystories.py download

# Train a custom vocabulary
python tinystories.py train_vocab --vocab_size=$VOCABS

# Pretokenize the dataset with the custom vocabulary
python tinystories.py pretokenize --vocab_size=$VOCABS

# Convert the sentencepiece model to binary format
python tokenizer.py --tokenizer-model=data/tok$VOCABS.model

echo "--- Step 2: Training (approx 3.4M parameters) ---"
python train.py \
    --dim=192 \
    --n_layers=6 \
    --n_heads=6 \
    --n_kv_heads=2 \
    --vocab_source=custom \
    --vocab_size=$VOCABS \
    --batch_size=128 \
    --max_iters=200000 \
    --max_seq_len=512 \
    --eval_interval=2000 \
    --eval_iters=100 \
    --out_dir=$OUTDIR

echo "--- Step 3: Export and Quantization ---"
# Convert the PyTorch checkpoint to an int8 quantized binary inside out/
#python export.py $OUTDIR/tinystories_3M_q80.bin --version 2 --checkpoint $OUTDIR/ckpt.pt

# Move and rename the tokenizer binary
mv data/tok$VOCABS.bin $OUTDIR/tok$VOCABS.bin

echo "--- Step 4: Build and Run Inference ---"
make run

# Run the model using 32 threads
echo "Running sample inference..."
./run $OUTDIR/model.bin -z $OUTDIR/tok$VOCABS.bin -t 1.0 -p 0.9 -n 256 -i "Once upon a time"
#OMP_NUM_THREADS=32 ./run $OUTDIR/tinystories_3M_q80.bin -z $OUTDIR/tinystories_4096.bin -t 1.0 -p 0.9 -n 100 -i "Once upon a time"
