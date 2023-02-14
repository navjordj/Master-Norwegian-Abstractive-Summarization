#!/bin/bash
#SBATCH --ntasks=1	                # 1 core (CPU)
#SBATCH --nodes=1	                # Use 1 node
#SBATCH --job-name=TM_test      	# Name of job
#SBATCH --mem=24G 			# Default memory per CPU is 3GB
#SBATCH --partition=smallmem            # Use GPU partition
#SBATCH --output=tuning_%j.out
#SBATCH --error=tuning%j.err
#SBATCH --mail-user=kristian.liland@nmbu.no # Email me when job is done.
#SBATCH --mail-type=ALL

## Script commands
singularity exec my_container.sif python ./dnaTMopt.py
