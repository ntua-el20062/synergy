#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_kmeans

## Output and error files
#PBS -o run_gpu_info.out
#PBS -e run_gpu_info.err

## How many machines should we get? 
#PBS -l nodes=1:ppn=8

##How long should the job run for?
#PBS -l walltime=00:30:00

## Start 
## Run make in the src folder (modify properly)

cd /home/parallel/parlab29/a3/gpu_info

export CUDA_VISIBLE_DEVICES=1

./gpu_info

export CUDA_VISIBLE_DEVICES=2

./gpu_info
