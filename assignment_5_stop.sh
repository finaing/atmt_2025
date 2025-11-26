#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_a5_stop_crit.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# Define arrays for stopping criteria and corresponding thresholds
stop_criteria=(relative_threshold_pruning absolute_threshold_pruning)
thresholds=(0.6 2.5) 

# Loop over indices
for i in "${!stop_criteria[@]}"; do
    stop_crit="${stop_criteria[i]}"
    thresh="${thresholds[i]}"
    
    echo "Running translation with stopping criterion = $stop_crit and threshold = $thresh"
    
    python translate.py \
        --cuda \
        --input ./cz-en/data/decode_test_raw/decode_test.cz \
        --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
        --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
        --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
        --output cz-en/output_len_norm.txt \
        --max-len 300 \
        --beam-size 5 \
        --stopping-criterion "$stop_crit" \
        --threshold "$thresh"
done