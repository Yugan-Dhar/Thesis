#!/bin/bash

#SBATCH --job-name=thesis_tune      # Job name
#SBATCH --time=20:0:0               # Time limit hrs:min:sec
#SBATCH --partition=gpua100         # Partition to submit to
#SBATCH --exclusive                 # Request nodes exclusively
#SBATCH --output=/storage/scratch/6603726/Thesis/slurm_outputs/%j.out  # Output file
#SBATCH --mail-type=END             # Email user when job finishes
#SBATCH --mail-user=m.s.y.sie@students.uu.nl

#TODO: Create new scripts for running locally instead of on the cluster

# Add arguments to the python script and run it
/storage/scratch/6603726/Thesis/venv2/bin/python3.9 training.py "$@"