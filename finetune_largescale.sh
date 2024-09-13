#!/bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=71:0:0     # XX hours runtime
#$ -l h_vmem=11G      # 11G RAM per core
#$ -pe smp 16          # 8 cores per GPU
#$ -l gpu=2           # request 1 GPU
#$ -l gpu_type=ampere   
##$ -l cluster=andrena
##$ -m se
# bash
# module load anaconda3/2020.02
source ~/.bashrc
# conda activate
conda activate py37

####$ -l cluster=andrena  # use the Andrena nodes  #$ -l cluster=andrena
# pip install -r requirements
# python -m torch.distributed.launch --nproc_per_node=2 dino_landmark_web.py   #> output.log
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0  --master_port 47770 train_webface_dis.py
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0  --master_port 47771 train_largescale.py \
--pretrain_path /data/home/acw569/precheck/webface_196land_sp/Backbone_VIT_land_8_Epoch_34_Batch_327225_Time_2022-05-05-10-34_checkpoint.pth \
--model_dir /data/scratch/acw569/checkpoint/ssl_check/SSL_Webface_webland_partViTB.pth \
--dataset_path /data/scratch/acw569/webface_4m/webface_rec --eval_root /data/scratch/acw569/webface_4m/ms1m-retinaface-t1 \
--outdir /data/scratch/acw569/checkpoint/sp_check/webface_webland_34epoch_mixup0505_aug01
# python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 --node_rank=0  --master_port 47771 train_withatt.py
# OMP_NUM_THREADS=1 python train_withatt.py
# OMP_NUM_THREADS=1 torchrun 
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0  --master_port 47770 ssl_finetune_adamw.py
# OMP_NUM_THREADS=1 python ssl_finetune_adamw.py
#train_webface_dis.py
#train_VGG_dis.py
#ijb_C_new.py
#python ijb_C_new.py
#ssl_finetune_dis.py   #sgd
#ssl_finetune_adamw.py