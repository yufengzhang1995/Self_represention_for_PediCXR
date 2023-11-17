#!/bin/bash
#SBATCH --job-name=SSL_vit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12GB
#SBATCH --time=4-00:00:00
#SBATCH --account=kayvan0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mail-user=chloezh@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./Logs/%x-%j.log
#SBATCH --error=./Logs/%x-%j.log

python main_ssl.py -c=dino_train_config.json
