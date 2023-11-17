#!/bin/bash
#SBATCH --job-name=mimic_cxr_note_chexpert
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=20
#SBATCH --mem=20GB
#SBATCH --time=48:00:00
#SBATCH --account=kayvan99
#SBATCH --partition=standard
#SBATCH --mail-user=chloezh@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./Logs/%x-%j.log
#SBATCH --error=./Logs/%x-%j.log


python build_dataframe.py
