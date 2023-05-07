import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from model import Net
from utils import ConfigL, ConfigS, download_weights


def predict(img_path, size='S', checkpoint_name='model.pt', res_path='./data/result/prediction', temperature=1.0, plot=True):
    config = ConfigL() if size.upper() == 'L' else ConfigS()

    # set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    is_cuda = torch.cuda.is_available()
    device = 'cuda' if is_cuda else 'cpu'

    ckp_path = os.path.join(config.weights_dir, checkpoint_name)

    assert os.path.isfile(img_path), 'Image does not exist'

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    img = Image.open(img_path)

    model = Net(
        clip_model=config.clip_model,
        text_model=config.text_model,
        ep_len=config.ep_len,
        num_layers=config.num_layers,
        n_heads=config.n_heads,
        forward_expansion=config.forward_expansion,
        dropout=config.dropout,
        max_len=config.max_len,
        device=device
    )

    if not os.path.exists(config.weights_dir):
        os.makedirs(config.weights_dir)

    if not os.path.isfile(ckp_path):
        download_weights(ckp_path, size)

    checkpoint = torch.load(ckp_path, map_location=device)
    model.load_state_dict(checkpoint)

    model.eval()

    with torch.no_grad():
        caption, _ = model(img, temperature)

    if plot:
        plt.imshow(img)
        plt.title(caption)
        plt.axis('off')

        img_save_path = f'{os.path.split(img_path)[-1].split(".")[0]}-R{size.upper()}.jpg'
        plt.savefig(os.path.join(res_path, img_save_path), bbox_inches='tight')

        plt.clf()
        plt.close()

    return caption
