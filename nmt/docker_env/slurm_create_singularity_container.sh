#!/bin/bash
#SBATCH --ntasks=1	                # 1 core (CPU)
#SBATCH --nodes=1	                # Use 1 node
#SBATCH --job-name=build_singularity	# Name of job
#SBATCH --mem=3G 			# Default memory per CPU is 3GB
#SBATCH --partition=gpu                 # Use GPU partition
#SBATCH --gres=gpu:1                    # Use one GPU
#SBATCH --output=log_create_cont-%j.out # Stdout and stderr file


## Script commands
module load singularity

if [ $# = 0 ]
then
    echo "Error. No definition file given"
elif [ $# -gt 1 ]
then
    echo "Too many arguments given. Do one file at the time"
else
    sif_filename="${1:0:end-4}.sif"
    echo "Making container $sif_filename from file $1"
    singularity build --fakeroot $sif_filename $1
    exit
fi 

