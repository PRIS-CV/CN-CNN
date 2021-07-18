import yaml
import torch
import numpy as np
import cv2

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# plot attention masks
def plot_mask_cat(inputs, mask_cat, unorm, vis, mode='train'):
    with torch.no_grad():
        for i in range(inputs.size(0)):
            img = unorm(inputs.data[i].cpu()).numpy().copy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            img = np.transpose(img, [1, 2, 0])
            r, g, b = cv2.split(img)
            img = cv2.merge([b, g, r])
            img = np.transpose(img, [2, 0, 1])
            vis.img('%s_img_%d' % (mode, i), img)
            for j in range(mask_cat.size(1)):
                mask = mask_cat[i, j, :, :].data.cpu().numpy()
                img_mask = (255.0 * (mask - np.min(mask)) / (np.max(mask) - np.min(mask))).astype(np.uint8)
                # img_mask = (255.0 * mask).astype(np.uint8)
                img_mask = cv2.resize(img_mask, dsize=(448, 448))
                vis.img('%s_img_%d_mask%d' % (mode, i, j), img_mask)