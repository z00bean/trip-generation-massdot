#!/bin/bash
#SBATCH -c 8  # Number of Cores per Task
#SBATCH --time=24:00:00
#SBATCH --mem=32000
#SBATCH --nodes=1
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=[a100|rtx8000]
#SBATCH --gpus-per-node=1
pwd
module load miniconda
conda init bash
conda activate yolov8-track
nvidia-smi
echo
echo
cd ~/CODE/YOLOv8-DeepSORT/yolo/v8/detect
echo
sh B1-374E-tracking-veh.sh
