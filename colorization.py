import torch
import models_mae
import argparse
from util.gray2rgb_datasets import Gray2RGBDataset
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import cv2
import os

def get_args_parser():
    parser = argparse.ArgumentParser('MAE Colorization Eval', add_help=False)
    parser.add_argument('--arch', default='mae_vit_base_patch16', type=str)
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--image_dir', default=None, type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--save_dir', default='mae_colorization_output', type=str)
    return parser

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cuda:0')
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    print(msg)
    return model


def main(args):
    mae_model = prepare_model(args.ckpt, args.arch)
    print('Model loaded.')
    size = (args.input_size, args.input_size)
    transform = transforms.Compose([
            transforms.Resize(size),  
            transforms.ToTensor(),
            ])
    normal = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    gray_normal = transforms.Normalize(mean=[0.299*0.485+0.587*0.456+0.114*0.406], std=[0.299*0.299+0.587*0.224+0.114*0.225])
    dataset_eval = Gray2RGBDataset(args.image_dir, transform, normal, gray_normal, eval_mode=True)
    data_loader_eval = torch.utils.data.DataLoader(dataset_eval, 
                                                    batch_size=1,
                                                    drop_last=False)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    for i, (rgb_image, gray_image) in tqdm(enumerate(data_loader_eval)):
        # run MAE
        latent, mask, ids_restore = mae_model.forward_encoder(gray_image, mask_ratio=0.0)
        y = mae_model.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        y = mae_model.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu().squeeze(0)
        colorize_img = torch.clip((y * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255, 0, 255).int()
        colorize_img = colorize_img.numpy()
        cv2.imwrite(os.path.join(args.save_dir, f'{i}_pred.JPEG'), colorize_img[:,:,::-1])
        
        
        

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
    