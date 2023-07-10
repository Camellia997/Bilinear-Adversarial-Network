#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=2 nohup python dann_customize.py -s C -t P -a m -e 50 --log logs_final/resnet50/CUB_C2P2
# CUDA_VISIBLE_DEVICES=2 nohup python dann_customize.py -s C -t P -a d -e 50 --log logs_final/vgg16/CUB_C2P2
# CUDA_VISIBLE_DEVICES=2 nohup python dann_customize.py -s C -t P -a bcnn_m -e 50 --log logs_final/bcnn_m/CUB_C2P2
# CUDA_VISIBLE_DEVICES=2 nohup python dann_customize.py -s C -t P -a bcnn_d -e 50 --log logs_final/bcnn_d/CUB_C2P2

# CUDA_VISIBLE_DEVICES=2 nohup python dann_customize.py -s C -t P -a m -e 50 -f f --log logs_final/dann_m/CUB_C2P2
# CUDA_VISIBLE_DEVICES=2 nohup python dann_customize.py -s C -t P -a d -e 50 -f f --log logs_final/dann_d/CUB_C2P2
# CUDA_VISIBLE_DEVICES=2 nohup python dann_customize.py -s C -t P -a bcnn_m -e 50 -f f --log logs_final/ban_m/CUB_C2P2
# CUDA_VISIBLE_DEVICES=2 nohup python dann_customize.py -s C -t P -a bcnn_d -e 50 -f f --log logs_final/ban_d/CUB_C2P2

# CUDA_VISIBLE_DEVICES=2 nohup python dann_customize.py -s C -t P -a bcnn_mm -e 50 -f f --log logs_final/ban_mm_f/CUB_C2P2
# CUDA_VISIBLE_DEVICES=2 nohup python dann_customize.py -s C -t P -a bcnn_mm -e 50 -f f1 --log logs_final/ban_mm_f1/CUB_C2P2
# CUDA_VISIBLE_DEVICES=2 nohup python dann_customize.py -s C -t P -a bcnn_mm -e 50 -f f2 --log logs_final/ban_mm_f2/CUB_C2P2

CUDA_VISIBLE_DEVICES=2 nohup python dann_customize.py -s C -t P -a bcnn_dd -e 50 -f f --log logs_final/ban_dd_f/CUB_C2P2
CUDA_VISIBLE_DEVICES=2 nohup python dann_customize.py -s C -t P -a bcnn_dd -e 50 -f f1 --log logs_final/ban_dd_f1/CUB_C2P2
CUDA_VISIBLE_DEVICES=2 nohup python dann_customize.py -s C -t P -a bcnn_dd -e 50 -f f2 --log logs_final/ban_dd_f2/CUB_C2P2
