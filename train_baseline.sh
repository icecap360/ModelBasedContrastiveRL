#!/bin/bash  
export CUDA_VISIBLE_DEVICES=2
# python impls/main.py --run_group "Runs" --algo_name "crl" --frame_stack 0
python impls/main.py --run_group "Runs" --algo_name "crl_model_based" --frame_stack 5