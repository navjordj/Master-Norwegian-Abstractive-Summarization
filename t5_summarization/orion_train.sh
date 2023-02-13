#!/bin/bash
#SBATCH --ntasks=1	                # 1 core (CPU)
#SBATCH --nodes=1	                # Use 1 node
#SBATCH --job-name=t5_test    	# Name of job
#SBATCH --mem=32G 			# Default memory per CPU is 3GB
#SBATCH --partition=gpu                 # Use GPU partition
#SBATCH --gres=gpu:2                    # Use two GPU
#SBATCH --output=t5_%j.out
#SBATCH --error=t5%j.err
#SBATCH --mail-user=jorgenav@nmbu.no # Email me when job is done.
#SBATCH --mail-type=ALL

## Script commands
singularity exec t5_latest.sif python ./training/orion_train.py
