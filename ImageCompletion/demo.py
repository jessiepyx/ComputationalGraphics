import sys
sys.path.extend(['.', '..'])

import os
import argparse
import torch
import json
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from model import CompletionNetwork
from PIL import Image
from utils import poisson_blend, gen_input_mask



def show_img(tensor, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(ndarr)
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()


def main(input_img):
    model_name = 'data/model/model_cn'
    config = 'config.json'
    max_holes = 5
    img_size = 160
    hole_min_w = 24
    hole_max_w = 48
    hole_min_h = 24
    hole_max_h = 48



    # =============================================
    # Load model
    # =============================================
    with open(config, 'r') as f:
        config = json.load(f)
    mpv = torch.tensor(config['mpv']).view(1,3,1,1)
    model = CompletionNetwork()
    if config['data_parallel']:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_name, map_location='cpu'))


    # =============================================
    # Predict
    # =============================================
    # convert img to tensor
    img = Image.open(input_img)
    img = transforms.Resize(img_size)(img)
    img = transforms.RandomCrop((img_size, img_size))(img)
    x = transforms.ToTensor()(img)
    x = torch.unsqueeze(x, dim=0)

    # create mask
    mask = gen_input_mask(
        shape=(1, 1, x.shape[2], x.shape[3]),
        hole_size=(
            (hole_min_w, hole_max_w),
            (hole_min_h, hole_max_h),
        ),
        max_holes=max_holes,
    )

    # inpaint
    with torch.no_grad():
        x_mask = x - x * mask + mpv * mask
        input = torch.cat((x_mask, mask), dim=1)
        output = model(input)
        inpainted = poisson_blend(x, output, mask)
        imgs = torch.cat((x, x_mask, inpainted), dim=0)
        show_img(imgs, nrow=3)


def demo():
    import glob
    input_imgs = glob.glob('data/images/*')
    for input_img in input_imgs:
        main(input_img)


if __name__ == '__main__':
    demo()

