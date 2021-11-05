#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16                    # 1 node
#SBATCH --ntasks-per-node=1         # 36 tasks per node
#SBATCH --time=24:00:00               # time limits: 500 hour
#SBATCH --partition=amdgpu 		#gpuextralong	  # gpufast
#SBATCH --gres=gpu:1
#SBATCH --output=logs/student_rpn_argo_train_mf1500_w3_p16384_agum_S1_0_T1_1_%j.log     # file name for stdout/stderr
ml PointRCNN #/0.0.0-fosscuda-20210b-PyTorch-1.7.1
## RPN
python ../train_rcnn.py --cfg_file ../cfgs/student/argo_train_mf1500_w3_p16384_agum_S1_0_T1_1.yaml --batch_size 16 --train_mode rpn --epochs 200 --output ../output/student/rpn/argo_train_mf1500_w3_p16384_agum_S1_0_T1_1



