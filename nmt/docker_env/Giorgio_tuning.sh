#!/bin/bash
#SBATCH --ntasks=4	                # 1 core (CPU)
#SBATCH --nodes=1	                # Use 1 node
#SBATCH --job-name=traeno    	# Name of job
#SBATCH --mem=8G 			# Default memory per CPU is 3GB
#SBATCH --partition=gpu                 # Use GPU partition
#SBATCH --gres=gpu:1                    # Use one GPU
#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR
#SBATCH --mail-user=jonkors@nmbu.no # Email me when job is done.
#SBATCH --mail-type=ALL



# Script commands
module purge
module load singularity

NAME="Translate_job" 
SAVEFOLDER="$TMPDIR/$NAME"

mkdir $SAVEFOLDER

#module singularity
singularity pull torch_gpu.sif docker://jkorsvik/torch_gpu_translate:latest
singularity exec --fakeroot torch_gpu.sif python ./translate.py -dataset_name cnn_dailymail -split test -batchsize 128 output_dir $SAVEFOLDER

# Copy files to location $HOME/dat300
cp -r $SAVEFOLDER $HOME/jonkors
# 
## Script commands
#singularity pull --force torch_gpu.sif docker://navjordj/t5:latest
#singularity exec torch_gpu.sif python ./translate.py -d cnn_dailymail -s training -b 128 


# Delete the files and folder from $TMPDIR
rm $SAVEFOLDER/*
rmdir $SAVEFOLDER