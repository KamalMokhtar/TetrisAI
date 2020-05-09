#!/bin/bash
#SBATCH --time=2-10:30:00
#SBATCH --nodes=8
#SBATCH --ntasks=20
#SBATCH --job-name=Tetris
#SBATCH --mem=20000
#SBATCH --output=Job-%j-q_learning.log
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=k.b.a.mokhtar@student.rug.nl

module load Python/3.6.4-foss-2018a
echo "Modules Loaded"
module list

echo "creating venv"
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
pip install tensorflow==1.14.0
pip install tensorboard==1.14.0
pip install Keras==2.2.4
pip install opencv-python==4.1.0.25
pip install numpy==1.16.4
pip install Pillow==5.4.1
pip install tqdm==4.31.1

echo "Starting execution"
python run.py
   