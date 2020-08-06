#!/bin/bash

#SBATCH --export=ALL
#SBATCH -J flowers
#SBATCH --exclude=node28
#SBATCH -o /cbio/donnees/aimbert/logs/log-%A.log
#SBATCH -e /cbio/donnees/aimbert/logs/log-%A.err
#SBATCH --gres=gpu:1
#SBATCH -t 0-100:00             # Time (DD-HH:MM)
#SBATCH --mem 16000             # Memory per node in MB (0 allocates all the memory)
#SBATCH --ntasks=1              # Number of processes to run (default is 1)
#SBATCH --cpus-per-task=8       # CPU cores per process (default 1)
#SBATCH -p gpu-cbio             # Name of the partition to use

echo 'Running train_dcgan.sh...'

echo
nvidia-smi
echo

# directories
input_directory='/mnt/data3/aimbert/data/flowers'
output_directory='/mnt/data3/aimbert/output/flowers'
log_directory="/mnt/data3/aimbert/output/flowers/dcgan"

# python script
script='/cbio/donnees/aimbert/flower-garden/flowers/dcgan/train.py'

python "$script" "$input_directory" \
       "$output_directory" \
        --log_directory "$log_directory"