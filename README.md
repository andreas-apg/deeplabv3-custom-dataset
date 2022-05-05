# DeepLabv3Plus-Pytorch

Modification of the [work](https://github.com/VainF/DeepLabV3Plus-Pytorch) by [Gongfan Fang](https://github.com/VainF).

Some tinkering of their implementation of DeepLab with a custom dataset loader.

## Quick Start 

### 1. Available Architectures
Specify the model architecture with '--model ARCH_NAME' and set the output stride using '--output_stride OUTPUT_STRIDE'.

| DeepLabV3    |  DeepLabV3+        |
| :---: | :---:     |
|deeplabv3_resnet50|deeplabv3plus_resnet50|
|deeplabv3_resnet101|deeplabv3plus_resnet101|
|deeplabv3_mobilenet|deeplabv3plus_mobilenet ||
|deeplabv3_hrnetv2_48 | deeplabv3plus_hrnetv2_48 |
|deeplabv3_hrnetv2_32 | deeplabv3plus_hrnetv2_32 |

### 2. Making a Custom Dataset
Modify the function get_labels in the custom.py in the Dataset directory, so that it returns the RGB colors of the segmentation mask annotations of your dataset. The encoder and decoder class methods `decode_target` and `encode_target` will handle the rest.

Since this implementation uses PIL, do take care to save your segmentation masks as RGB if they were made using openCV with cv2.cvtColor and the argument cv2.COLOR_BGR2RGB.

```python
def get_labels():
    """Load the mapping that associates classes with label colors.
       Don't forget to include background as black (0, 0, 0).
    Returns:
        np.ndarray with dimensions (n, 3)
    """
    return np.asarray([ (0, 0, 0),       # Background
                        (162, 0, 255),   # Object 1
                        (97, 16, 162),   # Object 2
                        ...
                        (162, 48, 0)]    # Object n
                        )

```
If you rename custom.py or are going to have multiple datasets, import those .py files in the `__init__.py` inside the datasets directory.

The dataset is made of an original image and its segmentation mask. Keep the original images and the segmentation masks in their own, separate directories. 

The segmentation mask has the exact same name as the original image, with a .png extension and should use the colors listed in the get_labels function. Any color not listed, such as outlines, will be considered the same as background. 

<div>
<img src="samples/1_image.png"   width="20%">
<img src="samples/1_target.png"  width="20%">
</div>

### 3. Training With a Custom Dataset

Pass the directories for the original images and the segmentation masks when calling train.py, with the arguments:
* `--train_dir` and `--train_seg_dir` for the training dataset;
* `--val_dir` and `--val_seg_dir` for the validation dataset.

The number of classes passed in the `--num_classes` argument should be the same as the n in your get_labels function: the number of objects of interest plus background.

For example, if you want to identify cats and dogs, you would have **3** classes: **background**, **cat** and **dog**.

To train, you would use something like this:

```bash
python train.py --model deeplabv3plus_mobilenet --gpu_id 0 --crop_val --lr 0.01 --crop_size 640 --batch_size 16 --output_stride 16 --train_dir /path/to/original/training/images/ --train_seg_dir /path/to/segmentation/training/images/ --val_dir /path/to/original/validation/images/ --val_seg_dir /path/to/segmentation/validation/images/ --save_val_results --num_classes 3 --dataset custom --model_name cats-and-dogs
```
The .pth weight file will be saved under the `checkpoints` directory. The argument passed in --model_name will be concatenated along with the name of the model passed in the `--model` argument and the `--dataset` argument, with both a latest and a best weight files: 
* latest_deeplabv3plus_mobilenet_custom_cats-and-dogs.pth
* best_deeplabv3plus_mobilenet_custom_cats-and-dogs.pth


#### 3.1. Continue Training

Run train.py with '--continue_training' to restore the state_dict of optimizer and scheduler from YOUR_CKPT.

```bash
python main.py ... --ckpt YOUR_CKPT --continue_training
```

#### 3.2. Testing

Results will be saved at ./results.

```bash
python main.py --model deeplabv3plus_mobilenet --enable_vis --vis_port 28333 --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --ckpt checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth --test_only --save_val_results
```

### 4. Prediction
You load the weight file using the `--ckpt` argument. Be sure to also pass the respective `--dataset` and `--model` that were used during training to generate those weights.

Single image:
```bash
python predict.py --input path/to/your/image.png  --dataset custom --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_custom_cats-and-dogs.pth --save_val_results_to test_results
```

Image folder:
```bash
python predict.py --input path/to/your/dir  --dataset custom --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_custom_cats-and-dogs.pth --save_val_results_to test_results
```
