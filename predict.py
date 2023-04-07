from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import CustomSegmentation
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

from PIL import ImageOps
import sys

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image or image directory")
    parser.add_argument("--num_classes", type=int, help='Number of classes (add 1 for the background)')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_dir", type=str, default="predictions",
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def get_labels():
    """Load the mapping that associates classes with label colors.
       Our electrical substation dataset has 14 objects + background.
    Returns:
        np.ndarray with dimensions (15, 3)
    """
    return np.asarray([ (0, 0, 0),       # Background
                        (162, 0, 255),   # Chave seccionadora lamina (Aberta)
                        (97, 16, 162),   # Chave seccionadora lamina (Fechada)
                        (81, 162, 0),    # Chave seccionadora tandem (Aberta)
                        (48, 97, 165),   # Chave seccionadora tandem (Fechada)
                        (121, 121, 121), # Disjuntor
                        (255, 97, 178),  # Fusivel
                        (154, 32, 121),  # Isolador disco de vidro
                        (255, 255, 125), # Isolador pino de porcelana
                        (162, 243, 162), # Mufla
                        (143, 211, 255), # Para-raio
                        (40, 0, 186),    # Religador
                        (255, 182, 0),   # Transformador
                        (138, 138, 0),   # Transformador de Corrente (TC)
                        (162, 48, 0),    # Transformador de Potencial (TP)
                        (162, 0, 96)     # Chave tripolar
                        ]) 

def decode_fn(label_mask):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
    Returns:
        (np.ndarray): the resulting decoded color image.
    """
    label_colors = get_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for l in range(0, len(label_colors)):
        r[label_mask == l] = label_colors[l, 0]
        g[label_mask == l] = label_colors[l, 1]
        b[label_mask == l] = label_colors[l, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb



def main():
    try:
        opts = get_argparser().parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: %s" % device)

        # Setup dataloader
        image_files = []
        if os.path.isdir(opts.input):
            for ext in ['png', 'jpeg', 'jpg', 'JPG', 'JPEG', 'tif']:
                files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
                if len(files)>0:
                    image_files.extend(files)
        elif os.path.isfile(opts.input):
            image_files.append(opts.input)

        # Set up model (all models are 'constructed at network.modeling)
        model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(model.backbone, momentum=0.01)

        if opts.ckpt is not None and os.path.isfile(opts.ckpt):
            # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
            checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["model_state"])
            model = nn.DataParallel(model)
            model.to(device)
            print("Resume model from %s" % opts.ckpt)
            #del checkpoint
        else:
            print("[!] Retrain")
            model = nn.DataParallel(model)
            model.to(device)

        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

        if opts.crop_val:
            transform = T.Compose([
                    T.Resize(opts.crop_size),
                    T.CenterCrop(opts.crop_size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])
        else:
            transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])

        os.makedirs(opts.save_dir, exist_ok=True)
        with torch.no_grad():
            model = model.eval()
            for img_path in tqdm(image_files):
                ext = os.path.basename(img_path).split('.')[-1]
                img_name = os.path.basename(img_path)[:-len(ext)-1]
                img = Image.open(img_path).convert('RGB')
                # A: need to rotate those that have exif, for the correct orientation
                img = ImageOps.exif_transpose(img)
                img = transform(img).unsqueeze(0) # To tensor of NCHW
                img = img.to(device)

                #temp = img.max(1)[1].cpu().numpy()[0] # HW
                #temp = (denorm(temp)).transpose(1, 2, 0).astype(np.uint8)
                #temp = Image.fromarray(temp)
                #temp.save(os.path.join(opts.save_dir, img_name+'_resized.png'))    

                pred = model(img).max(1)[1].cpu().numpy()[0] # HW
                colorized_preds = decode_fn(pred).astype('uint8')
                colorized_preds = Image.fromarray(colorized_preds)
                colorized_preds.save(os.path.join(opts.save_dir, img_name+'.png'))
        sys.exit(0)
    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    main()
