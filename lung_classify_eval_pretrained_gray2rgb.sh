python main_finetune.py --eval \
        --resume ./cls_output_dir/checkpoint-10.pth \
        --model vit_base_patch16 \
        --batch_size 32 \
        --cls_token \
        --data_path data/images \
        --nb_classes 14 \
        --image_info ./data/Data_Entry_2017.csv \
        --train_split ./data/train_val_list.txt \
        --val_split ./data/test_list.txt