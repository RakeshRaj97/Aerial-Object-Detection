#!/bin/bash

# script for building virtual environment for project 2

module load anaconda3/5.1.0
module load gcc/6.4.0

conda create -n p2 python=3.7
source activate p2
conda install pytorch==1.8.1 torchvision cudatoolkit=10.2 -c pytorch
conda install pillow=6.2.1 -y
pip install matplotlib
pip install utility
pip install imageio
pip install opencv-python
pip install tqdm
pip install scikit-learn
pip install shapely
pip install PyYAML
pip install tensorboard
pip install pandas
pip install seaborn
pip install thop
pip install lxml
conda install ipykernel
python -m ipykernel install --user --name p2 --display-name p2env
source deactivate
