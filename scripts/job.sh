#!/bin/bash
#SBATCH --time=480
#SBATCH --ntasks-per-node=40

source ../../drift/bin/activate
python 04_v3_fit_models_drift_diffusion_with_linear_regression.py