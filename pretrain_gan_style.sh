OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py --batch_size 64 --world_size 4 --model mae_vit_base_patch16 --mask_ratio 0.25 --epochs 800 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --data_path ../imagenet2012