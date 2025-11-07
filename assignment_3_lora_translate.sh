#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=lora_translate_out.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

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
