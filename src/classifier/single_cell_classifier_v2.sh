#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=sc_classifier
#SBATCH --time=10-00:00:00
#SBATCH --partition=batch
#SBATCH --qos=long_jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128000
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=FAIL,REQUEUE,TIME_LIMIT_80,END,BEGIN
#SBATCH --requeue
#SBATCH --mail-user=kirchgae@ohsu.edu

patient=$1

python3 src/classifier/single_cell_classifier_v2.py -p ${patient}