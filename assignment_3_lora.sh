#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=lora_out.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# ============================================================
# STEP 1: PREPROCESS (optional if already done)
# ============================================================
python preprocess.py \
    --source-lang pl \
    --target-lang en \
    --raw-data pl-en/data/raw \
    --dest-dir ./pl-en/data/prepared \
    --model-dir ./cz-en/tokenizers \
    --test-prefix test \
    --train-prefix train \
    --valid-prefix valid \
    --src-vocab-size 8000 \
    --tgt-vocab-size 8000 \
    --src-model ./cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-model ./cz-en/tokenizers/en-bpe-8000.model


# ============================================================
# STEP 2: LoRA FINETUNING
# ============================================================

python lora_train.py \
    --cuda \
    --data pl-en/data/prepared/ \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --source-lang pl \
    --target-lang en \
    --batch-size 64 \
    --arch transformer \
    --max-epoch 5 \
    --log-file pl-en/logs/train_lora.log \
    --save-dir pl-en/checkpoints_lora/ \
    --restore-file cz-en/checkpoints/checkpoint_best.pt \
    --encoder-dropout 0.1 \
    --decoder-dropout 0.1 \
    --dim-embedding 256 \
    --attention-heads 4 \
    --dim-feedforward-encoder 1024 \
    --dim-feedforward-decoder 1024 \
    --max-seq-len 300 \
    --n-encoder-layers 3 \
    --n-decoder-layers 3 \
    \
    --lora \
    --lora-r 8 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --lora-target-modules q_proj,k_proj,v_proj,out_proj

python lora_translate.py \
  --cuda \
  --input pl-en/data/raw/test.pl \
  --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
  --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
  --checkpoint-path pl-en/checkpoints_lora/checkpoint_best.pt \
  --output pl-en/output_lora.txt \
  --max-len 300 \
  \
  --lora \
  --lora-r 8 \
  --lora-alpha 32 \
  --lora-target-modules q_proj,k_proj,v_proj,out_proj
