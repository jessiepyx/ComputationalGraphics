from model import Net
from datasets import GrayImageDataset
from utils import output2rgb
import torch


def demo():
    dataset = GrayImageDataset('data/val', train=False)
    model = Net()
    model.load_state_dict(torch.load('data/model/color.pth', map_location='cpu'))
    model.eval()
    for l_input, ab_input in dataset:
        ab_input = model(l_input.unsqueeze(0))[0]
        output2rgb(l_input, ab_input)


if __name__ == '__main__':
    demo()
