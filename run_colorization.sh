python colorization.py --arch mae_vit_base_patch16 \
                       --ckpt ./pretrained_ckpt/checkpoint-640.pth \
                       --image_dir ./data/imagenet_val/ILSVRC2012_img_val \
                       --save_dir ./mae_colorization_output \
                       --input_size 224

python -m pytorch_fid ./data/imagenet_val/ILSVRC2012_img_val ./mae_colorization_output --device cuda:0