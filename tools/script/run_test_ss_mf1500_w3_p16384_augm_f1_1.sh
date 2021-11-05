#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16                    # 1 node
#SBATCH --ntasks-per-node=1         # 36 tasks per node
#SBATCH --time=4:00:00               # time limits: 500 hour
#SBATCH --partition=amdgpufast 		#gpuextralong	  # gpufast
#SBATCH --gres=gpu:1
#SBATCH --output=logs/test_ss_argo_train_mf1500_w3_p16384_agum_f1_1_%j.log     # file name for stdout/stderr
# module
ml PointRCNN
## test

python ../eval_rcnn.py --cfg_file ../cfgs/val/argo_train_mf1500_w3_p16384_agum_f1_1.yaml --ckpt ../output/teacher/rpn/argo_train_mf1500_w3_p16384_agum_f1_1/ckpt/best.pth --batch_size 1 --eval_mode rpn --output_dir ../val/SS/val/teacher/argo_train_mf1500_w3_p16384_agum_f1_1
