import os
import shutil
import pandas as pd
import random

def move_image(data_root_path):
    data_paths = []
    for item in os.listdir(data_root_path):
        if 'images' in item:
            data_paths.append(item)
    os.mkdir((os.path.join(data_root_path, 'images')))
    for dir in data_paths:
        img_lst = os.listdir(os.path.join(data_root_path, dir, 'images'))
        for img_name in img_lst:
            src = os.path.join(os.path.join(data_root_path, dir, 'images', img_name))
            target = os.path.join(data_root_path, 'images', img_name)
            print(f'Moving {src} to {target}...')
            shutil.move(src, target)
        shutil.rmtree(os.path.join(data_root_path, dir))
    
def split_data(ori_path, target_path, num):
    with open(ori_path, 'r') as f:
        train_val_lst = f.readlines()
    select = random.sample(train_val_lst, num)
    with open(target_path, 'w') as f:
        for item in select:
            f.write(item)

    
    
    

if __name__ == '__main__':
    # move_image('./data')
    num = 1000
    split_data('data/train_val_list.txt',
               f'data/train_sample_{num}.txt',
                num
    )
            
    