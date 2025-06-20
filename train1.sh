#!/bin/bash  
export CUDA_VISIBLE_DEVICES=1
# python impls/main.py --run_group "Runs" --algo_name "crl" --frame_stack 0
python impls/main.py --run_group "Runs" --algo_name "crl_model_based" --frame_stack 15 --env_name "puzzle-3x3-play-v0" 
python impls/main.py --run_group "Runs" --algo_name "crl_model_based" --frame_stack 15 --env_name "puzzle-4x4-play-v0" 

# python impls/main.py --run_group "Runs" --algo_name "crl_model_based" --frame_stack 15 --env_name "puzzle-3x3-play-v0" --seed 2
# python impls/main.py --run_group "Runs" --algo_name "crl_model_based" --frame_stack 15 --env_name "puzzle-3x3-noisy-v0" --seed 2


# python impls/main.py --run_group "Runs" --algo_name "crl_model_based" --frame_stack 15 --env_name "puzzle-3x3-play-v0" --seed 3
# python impls/main.py --run_group "Runs" --algo_name "crl_model_based" --frame_stack 15 --env_name "puzzle-3x3-noisy-v0" --seed 3