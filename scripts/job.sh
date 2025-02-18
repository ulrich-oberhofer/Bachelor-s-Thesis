#!/bin/bash
#SBATCH --time=120
#SBATCH --ntasks-per-node=40

source ../../drift/bin/activate
python 04_only_mlp_v3.py