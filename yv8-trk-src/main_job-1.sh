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
conda activate /home/zubin_bhuyan_student_uml_edu/.conda/envs/yolov8-track
nvidia-smi
echo
echo
cd /work/pi_yuanchang_xie_uml_edu/zubin/CODE/YOLOv8-DeepSORT/yolo/v8/detect
echo
sh B1-374E-tracking-veh.sh