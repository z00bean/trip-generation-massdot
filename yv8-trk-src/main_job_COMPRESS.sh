#!/bin/bash
#SBATCH -c 8  # Number of Cores per Task
#SBATCH --time=23:59:00
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=[a100|rtx8000]
#SBATCH --gpus-per-node=1
pwd
module load miniconda
conda init bash

nvidia-smi
echo
cd ~/DATA/trip-gen/processed-OUT_DIR/BATCH3
echo

module load ffmpeg/4.4.1-libx264
sh compress-processed-vids.sh
