import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb

import torch


def output2rgb(l_input, ab_input):
    """
    l_input: [1, H, W]
    ab_input: [2, H, W]
    """
    ab_input = ab_input.detach().cpu()
    l_input = l_input.detach().cpu()

    color_image = torch.cat((l_input, ab_input), 0).numpy()
    color_image = color_image.transpose((1, 2, 0))
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
    color_image = lab2rgb(color_image.astype(np.float64))
    l_input = l_input.squeeze().numpy()

    _, axarr = plt.subplots(1, 2)
    axarr[0].imshow(l_input, cmap='gray')
    axarr[1].imshow(color_image)
    plt.show()
