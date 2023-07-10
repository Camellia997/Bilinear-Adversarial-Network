#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 nohup python dann_customize.py -s S -t W -a m -e 15 -n 281 --log logs_final/resnet50/CompCars_S2W2
CUDA_VISIBLE_DEVICES=1 nohup python dann_customize.py -s S -t W -a d -e 15 -n 281 --log logs_final/vgg16/CompCars_S2W2
CUDA_VISIBLE_DEVICES=1 nohup python dann_customize.py -s S -t W -a bcnn_m -e 15 -n 281 --log logs_final/bcnn_m/CompCars_S2W2
CUDA_VISIBLE_DEVICES=1 nohup python dann_customize.py -s S -t W -a bcnn_d -e 15 -n 281 --log logs_final/bcnn_d/CompCars_S2W2

CUDA_VISIBLE_DEVICES=1 nohup python dann_customize.py -s S -t W -a m -e 15 -n 281 -f f --log logs_final/dann_m/CompCars_S2W2
CUDA_VISIBLE_DEVICES=1 nohup python dann_customize.py -s S -t W -a d -e 15 -n 281 -f f --log logs_final/dann_d/CompCars_S2W2
CUDA_VISIBLE_DEVICES=1 nohup python dann_customize.py -s S -t W -a bcnn_m -e 15 -n 281 -f f --log logs_final/ban_m/CompCars_S2W2
CUDA_VISIBLE_DEVICES=1 nohup python dann_customize.py -s S -t W -a bcnn_d -e 15 -n 281 -f f --log logs_final/ban_d/CompCars_S2W2

# CUDA_VISIBLE_DEVICES=1 nohup python dann_customize.py -s S -t W -a bcnn_mm -e 15 -n 281 -f f --log logs_final/ban_mm_f/CompCars_S2W2
# CUDA_VISIBLE_DEVICES=1 nohup python dann_customize.py -s S -t W -a bcnn_mm -e 15 -n 281 -f f1 --log logs_final/ban_mm_f1/CompCars_S2W2
# CUDA_VISIBLE_DEVICES=1 nohup python dann_customize.py -s S -t W -a bcnn_mm -e 15 -n 281 -f f2 --log logs_final/ban_mm_f2/CompCars_S2W2

# CUDA_VISIBLE_DEVICES=1 nohup python dann_customize.py -s S -t W -a bcnn_dd -e 15 -n 281 -f f --log logs_final/ban_dd_f/CompCars_S2W2
# CUDA_VISIBLE_DEVICES=1 nohup python dann_customize.py -s S -t W -a bcnn_dd -e 15 -n 281 -f f1 --log logs_final/ban_dd_f1/CompCars_S2W2
# CUDA_VISIBLE_DEVICES=1 nohup python dann_customize.py -s S -t W -a bcnn_dd -e 15 -n 281 -f f2 --log logs_final/ban_dd_f2/CompCars_S2W2
