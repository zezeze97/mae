import os
import shutil
import pandas as pd

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

    
    
    

if __name__ == '__main__':
    move_image('./data')
    # create_train_val_info('./data')
    # print(len(os.listdir('./data/images')))
            
    