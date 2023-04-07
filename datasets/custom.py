import os
import sys
import glob
#import tarfile
#import collections
import torch.utils.data as data
import shutil
import numpy as np

from PIL import Image
from PIL import ImageOps

class CustomSegmentation(data.Dataset):
    """
    Args:
        image_dir: Directory that contains the images to be used.
        
        mask_dir: Directory that contains the corresponding .png segmentation masks
        for the images present in image_dir.
        
        transform (callable, optional): A function/transform that  takes in an PIL
        image and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    def __init__(self,
                 image_dir,  
                 mask_dir,
                 transform=None):
        self.transform = transform
        # A: checking if the jpg directory exists
        if not os.path.isdir(image_dir):
            raise RuntimeError('Dataset not found or corrupted.' +
                               'Check the source image folder.')
        # A: checking if the annotation mask directory exists
        if not os.path.exists(mask_dir):
            raise RuntimeError("Segmentation not found or corrupted.")

        # A: aggregating all files in the image_dir
        file_list = os.path.join(os.path.join(image_dir, "*.*"))
        
        # A: using glob to get the image names, since they can have more than one extension
        file_list = glob.glob(file_list)
        image_list = []
        for file in file_list:
            # A: filtering since the directory also contains the .txt files used for YOLO annotation
            if ".txt" not in file:
                # A: keeping only the filename and its extension, without the path
                file = file.split("/")[-1]
                #file = os.path.splitext(file)[0]
                image_list.append(file)
        
        self.images = [os.path.join(image_dir, x) for x in image_list]
        # A: masks are always .png.
        self.masks = [os.path.join(mask_dir, os.path.splitext(x)[0] + ".png") for x in image_list]
        assert (len(self.images) == len(self.masks)) 

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        # A: portrait mode jpg images need to be transposed, because they are saved in landscape
        # and rotated by an exif tag  
        img = ImageOps.exif_transpose(img)
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)
        # A: the target has to be encoded after the transform is applied. Since the transform also
        # changes the target into a tensor (on top of resizing the image and whatnot), np.array is
        # needed to pass it into the encode function.
        result, flag = self.encode_target(np.array(target))
        if not flag:
            print(self.images[index])
        return img, result 
        
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is the image segmentation.
#         """
#         try:
#             img = Image.open(self.images[index]).convert('RGB')
#             # A: portrait mode jpg images need to be transposed, because they are saved in landscape
#             # and rotated by an exif tag  
#             img = ImageOps.exif_transpose(img)
#             target = Image.open(self.masks[index])
#             if self.transform is not None:
#                 img, target = self.transform(img, target)
#             # A: the target has to be encoded after the transform is applied. Since the transform also
#             # changes the target into a tensor (on top of resizing the image and whatnot), np.array is
#             # needed to pass it into the encode function.
#             return img, self.encode_target(np.array(target))
#         except Exception as e:
#             print(e)
#             print(self.images[index])
#             print(self.masks[index])
#             img = Image.open("/home/jovyan/work/deeplab/DeepLabV3Plus-Pytorch/datasets/dummy.png").convert('RGB')
#             img = ImageOps.exif_transpose(img)
#             target = Image.open("/home/jovyan/work/deeplab/DeepLabV3Plus-Pytorch/datasets/dummy.png")
#             if self.transform is not None:
#                 img, target = self.transform(img, target)
#             # A: the target has to be encoded after the transform is applied. Since the transform also
#             # changes the target into a tensor (on top of resizing the image and whatnot), np.array is
#             # needed to pass it into the encode function.
#             return img, self.encode_target(np.array(target))

    def __len__(self):
        return len(self.images)
    
    # A: Needed to encode the colors as (M, N) labels for the loss.
    # Since the loss can't be calculated with a tensor of shape 
    # [batch_size, height, width, channels] if the three channels are
    # present, we need this to be able to encode the rgb class colorsself.images[index], 
    # into a single value to represent them.
    def get_labels():
        """Load the mapping that associates classes with label colors.
           Our electrical substation dataset has 15 objects + background.
        Returns:
            np.ndarray with dimensions (16, 3)
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


#     @classmethod
#     def decode_target(cls, mask):
#         """decode semantic mask to RGB image"""
#         return cls.cmap[mask]

    @classmethod
    def decode_target(self, label_mask):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
        Returns:
            (np.ndarray): the resulting decoded color image.
        """
        label_colors = self.get_labels()
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
    
    # A: in order for the mask to go through the loss function, the classes need to be
    # represented as a single value, opposed to three channels.
    @classmethod
    # https://github.com/meetps/pytorch-semseg/blob/master/ptsemseg/loader/pascal_voc_loader.py
    def encode_target(self, mask):
        """Encode segmentation label images as classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the classes are encoded as colors.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        try:
            mask = mask.astype(int)
            label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint16)
            for i, label in enumerate(self.get_labels()):
                index = np.where(np.all(mask == label, axis=-1))[:2]
                # A: if, for whatever reason, an index bigger than the mask shape is returned, it'll
                # give an out of bounds error. I don't know why this happens, but sometimes, np.where
                # does return some junk here. 
                # Edit: apparently, those errors happens because a higher bit in the number is set
                # by mistake, making a value that would be within 0... 639 to come out like 112589990684325.
                # Bit masking that should help, perhaps? & 0xFFFF
        
                # A: treating indexes out of bounds
                index = (index[0] & 0xFFFF, index[1] & 0xFFFF)
                label_mask[index] = i
            label_mask = label_mask.astype(int)
            return label_mask, True
        except Exception as e:
            print(e)
            print(f'Label {label} {i}')
            #file = open("index.txt", "w")
            #file.writelines(index)
            #file.close()
            print(mask.shape[0], mask.shape[1])
            return np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16), False