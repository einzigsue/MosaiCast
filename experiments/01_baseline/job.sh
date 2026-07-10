#!/bin/bash
#PBS -l walltime=05:00:00
#PBS -l ncpus=16
#PBS -l ngpus=1
#PBS -l mem=250GB
#PBS -l jobfs=400GB
#PBS -l wd
#PBS -q dgxa100
#PBS -l storage=scratch/z00+gdata/z00+gdata/dk92+gdata/pp66+gdata/lm70
#PBS -P fp0

module use /g/data/pp66/apps/modulefiles/
module load aurora/microsoft-1.8.0

#python3 train.py --config configs/default.yaml --exp test --smoke
python3 train.py --config configs/default.yaml --exp baseline
