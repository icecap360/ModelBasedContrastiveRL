#!/bin/bash  
export CUDA_VISIBLE_DEVICES=0
python impls/main.py --run_group "Runs" --algo_name "crl_model_based" --frame_stack 15 --env_name "pointmaze-medium-stitch-v0" 
python impls/main.py --run_group "Runs" --algo_name "crl_model_based" --frame_stack 15 --env_name "pointmaze-large-stitch-v0" 


# python impls/main.py --run_group "Runs" --algo_name "crl_model_based" --frame_stack 15 --env_name "pointmaze-medium-stitch-v0" --seed 2
# python impls/main.py --run_group "Runs" --algo_name "crl_model_based" --frame_stack 15 --env_name "pointmaze-large-stitch-v0" --seed 2


# python impls/main.py --run_group "Runs" --algo_name "crl_model_based" --frame_stack 15 --env_name "pointmaze-medium-stitch-v0" --seed 3
# python impls/main.py --run_group "Runs" --algo_name "crl_model_based" --frame_stack 15 --env_name "pointmaze-large-stitch-v0" --seed 3