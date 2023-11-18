#!/bin/bash
#SBATCH --job-name=exp_vit_SSL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=4-00:00:00
#SBATCH --account=kayvan0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:4
#SBATCH --mail-user=chloezh@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./Logs/%x-%j.log
#SBATCH --error=./Logs/%x-%j.log

# python main_ssl.py -c=config/dino_train_config.json --nnode=1 --node_rank=0 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nnode=1 --node_rank=0 --nproc_per_node=4 main_dino_ddp.py -c "config/dino_train_config.json"
