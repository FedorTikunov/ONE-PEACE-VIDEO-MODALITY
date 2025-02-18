#!/bin/bash
#SBATCH --job-name=train_video_adapter1
#SBATCH --error=/userspace/tfv/logs/train_video_adapter1.err
#SBATCH --output=/userspace/tfv/logs/train_video_adapter1.log
#SBATCH --partition=a100
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=ngpu09

# Log start time
echo "Script started at $(date)" >> /userspace/tfv/logs/train_video_adapter1.log
echo "Script started at $(date)" >> /userspace/tfv/logs/train_video_adapter1.err

# Activate conda environment
source /userspace/tfv/miniconda3/etc/profile.d/conda.sh
echo "Loading conda environment" >> /userspace/tfv/logs/train_video_adapter1.log
conda activate dip_env

# Set Conda cache directories to writable locations
export CONDA_PKGS_DIRS=/userspace/tfv/cache/conda_pkgs
export CONDA_CACHE_DIR=/userspace/tfv/cache

# Set environment variables for CUDA and others
export CUDA_HOME=/usr/local/cuda-11.7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
export MASTER_PORT=29500
export GPUS_PER_NODE=9

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export PIP_CACHE_DIR=/userspace/tfv/cache
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
export HF_HOME=/userspace/tfv/cache/huggingface
export DO_NOT_TRACK=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HAYSTACK_TELEMETRY_ENABLED=False
export ANONYMIZED_TELEMETRY=False
export OUTLINES_CACHE_DIR=/userspace/tfv/cache/outlines
export NLTK_DATA=/userspace/tfv/cache/nltk_data
export MPLCONFIGDIR=/userspace/tfv/cache/matplotlib
export FASTEMBED_CACHE_PATH=/userspace/tfv/cache/fastembed
export PATH=/userspace/dra/ffmpeg-release:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export DEEPSPEED_AUTOTUNE_CACHE=/userspace/tfv/cache/deepspeed
export DEEPSPEED_CACHE_DIR=/userspace/tfv/cache/deepspeed
export TRITON_CACHE_DIR=/userspace/tfv/cache/deepspeed

echo "Environment variables set" >> /userspace/tfv/logs/train_video_adapter1.log

# Print CUDA details
echo "Printing CUDA details" >> /userspace/tfv/logs/train_video_adapter1.log
ls -d /usr/local/cuda* >> /userspace/tfv/logs/train_video_adapter1.log
nvidia-smi -L >> /userspace/tfv/logs/train_video_adapter1.log
nvcc -V >> /userspace/tfv/logs/train_video_adapter1.log
python -V >> /userspace/tfv/logs/train_video_adapter1.log

# Define configuration paths
config_dir=.
config_name=pretrain_video_3B
save_dir=./checkpoints/one_peace_video
restore_file=/userspace/tfv/project_menon/models_weights/one-peace.pt

# Launch training using DeepSpeed.
# Option 1: Use DeepSpeed's launcher directly.
#deepspeed --num_gpus=${GPUS_PER_NODE} /userspace/tfv/diplom/ONE-PEACE-VIDEO-MODALITY/one_peace/train.py \
#    --config-dir=${config_dir} \
#    --config-name=${config_name} \
#    checkpoint.save_dir=${save_dir} \
#    checkpoint.restore_file=${restore_file}

# Option 2 (if your training code requires torchrun with deepspeed integration):
torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} /userspace/tfv/diplom/ONE-PEACE-VIDEO-MODALITY/one_peace/train.py \
     --config-dir=${config_dir} \
     --config-name=${config_name} \
     checkpoint.save_dir=${save_dir} \
     checkpoint.restore_file=${restore_file}

echo "Script ended at $(date)" >> /userspace/tfv/logs/train_video_adapter1.log
