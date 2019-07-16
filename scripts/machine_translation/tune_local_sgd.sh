#!/bin/bash

MXNET_GPU_MEM_POOL_TYPE=Round python train_transformer_local_sgd_impl_8.py --dataset WMT2014BPE --src_lang en --tgt_lang de --batch_size 2700 --optimizer adam --num_accumulated 16 --lr 3.0 --warmup_steps 4000 --save_dir transformer_en_de_u512 --epochs 30 --gpus 0,1,2,3,4,5,6,7 --scaled --average_start 5 --num_buckets 20 --bucket_scheme exp --bleu 13a --log_interval 10 --local_sgd 1
MXNET_GPU_MEM_POOL_TYPE=Round python train_transformer_local_sgd_impl_8.py --dataset WMT2014BPE --src_lang en --tgt_lang de --batch_size 2700 --optimizer adam --num_accumulated 16 --lr 3.0 --warmup_steps 3000 --save_dir transformer_en_de_u512 --epochs 30 --gpus 0,1,2,3,4,5,6,7 --scaled --average_start 5 --num_buckets 20 --bucket_scheme exp --bleu 13a --log_interval 10 --local_sgd 1


for LR in 2.8 3.0 3.2
do
	for WARMUP in 2500 3000 3500 4000
	do
		MXNET_GPU_MEM_POOL_TYPE=Round python train_transformer_local_sgd_impl_9.py --dataset WMT2014BPE --src_lang en --tgt_lang de --batch_size 2700 --optimizer adam --num_accumulated 16 --lr $LR --warmup_steps $WARMUP --save_dir transformer_en_de_u512 --epochs 30 --gpus 0,1,2,3,4,5,6,7 --scaled --average_start 5 --num_buckets 20 --bucket_scheme exp --bleu 13a --log_interval 10 --local_sgd 10
	done
done