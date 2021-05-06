#!/bin/bash
#
#SBATCH --job-name=copy_files
#SBATCH --output=res.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=500

module load anaconda3/5.1.0
module load tqdm/4.47.0
module load joblib/0.11
source activate /home/rgopala/.conda/envs/p2
python yolo_to_voc.py
