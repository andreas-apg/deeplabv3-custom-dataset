from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import CustomSegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import os
import glob

from mail import * # A: for email

import sys
import traceback

import warnings
warnings.filterwarnings("ignore")

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    #parser.add_argument('--data', type=str, help='data.yaml path')
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--val_dir', type=str)
    parser.add_argument('--train_seg_dir', type=str)
    parser.add_argument('--val_seg_dir', type=str)
    
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='custom',
                        choices=['voc', 'cityscapes', 'custom'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

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
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=50e3,
                        help="epoch number (default: 50k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    parser.add_argument("--model_name", type=str, default="",
                        help="name of the model to be used for the weights")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    parser.add_argument("--mail_subject", type=str,
                        help='subject to be used on the emailed progress report')
    parser.add_argument("--training_round", type=int,
                        help='number of training round for simple does it')
    parser.add_argument("--total_epochs", type=int, default=100,
                        help="total epochs (default: 100)")
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = et.ExtCompose([
        # et.ExtResize(size=opts.crop_size),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    if opts.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    train_dst = CustomSegmentation(image_dir=opts.train_dir, mask_dir=opts.train_seg_dir, transform=train_transform)
    val_dst = CustomSegmentation(image_dir=opts.val_dir, mask_dir=opts.val_seg_dir, transform=val_transform)
 
    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    # A: uncomment later
                    #Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    #Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    first_val = False # A: this flag is to control the first validation for the mail
    
    # A: added this to keep logs instead of using Visdom vis
    check_path = os.path.isdir("./logs/")
    if not check_path:
        os.makedirs("./logs/")
        print("created folder : ", "./logs/")   
    
    mail_server = get_mail_server() # A: for email

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    #if opts.dataset == 'voc' and not opts.crop_val:
    #    opts.val_batch_size = 1
    opts.val_batch_size = 1
    
    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=8,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=8)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "cur_epochs": cur_epochs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            cur_epochs = checkpoint["cur_epochs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        try:
            # =====  Train  =====
            model.train()
            
            # A: mail begin
            if cur_epochs%10 == 0 and first_val:
                local_time = get_time()         # A: for email
                mail_reply = f"Training: round {opts.training_round}\nEpoch {cur_epochs}/{opts.total_epochs}, iteration {cur_itrs} at {local_time}.\nLoss: {np_loss}\nOverall Acc: {val_score['Overall Acc']}\nMean Acc: {val_score['Mean Acc']}\nFreqW Acc: {val_score['FreqW Acc']}\nMean IoU: {val_score['Mean IoU']}\nClass IoU: {val_score['Class IoU']}"
                for dest in get_mail_addresses():
                    send_mail(server = mail_server["server_address"],
                              port = mail_server["server_port"],
                              user = mail_server["user"],
                              password = mail_server["password"],
                              to = dest,
                              subject = opts.mail_subject,
                              body = mail_reply)
            # A: mail end  

            cur_epochs += 1
            for (images, labels) in train_loader:
                cur_itrs += 1

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                np_loss = loss.detach().cpu().numpy()
                interval_loss += np_loss
                if vis is not None:
                    vis.vis_scalar('Loss', cur_itrs, np_loss)

                if (cur_itrs) % 10 == 0:
                    interval_loss = interval_loss / 10
                    #print("Epoch %d, Itrs %d/%d, Loss=%f" %
                    #      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                    print("Epoch %d/%d, Itrs %d, Loss=%f" %
                          (cur_epochs + 1, opts.total_epochs, cur_itrs, interval_loss))
                    interval_loss = 0.0

                if (cur_itrs) % opts.val_interval == 0:
                    save_ckpt('checkpoints/latest_%s_%s_os%d_%s.pth' %
                              (opts.model, opts.dataset, opts.output_stride, opts.model_name))
                    print("validation...")
                    model.eval()
                    val_score, ret_samples = validate(
                        opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                        ret_samples_ids=vis_sample_id)
                    first_val = True
                    print(metrics.to_str(val_score))
                    if val_score['Mean IoU'] > best_score:  # save best model
                        best_score = val_score['Mean IoU']
                        save_ckpt('checkpoints/best_%s_%s_os%d_%s.pth' %
                                  (opts.model, opts.dataset, opts.output_stride, opts.model_name))  
                    
                    # A: added these to keep logs instead of using Visdom
                    output = [f"Epoch: {cur_epochs}", 
                              f"Iteration: {cur_itrs}", 
                              f"Loss: {np_loss}",
                              f"Overall Acc: {val_score['Overall Acc']}",
                              f"Mean Acc: {val_score['Mean Acc']}",
                              f"FreqW Acc: {val_score['FreqW Acc']}",
                              f"Mean IoU: {val_score['Mean IoU']}",
                              f"Class IoU: {val_score['Class IoU']}"]
                    with open(f'./logs/{opts.model_name}_logs.txt', 'w') as f:
                        f.writelines('\n'.join(output))
                    f.close()
                    if vis is not None:  # visualize validation score and samples
                        vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                        vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                        vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                        for k, (img, target, lbl) in enumerate(ret_samples):
                            img = (denorm(img) * 255).astype(np.uint8)
                            # A: this transpose is done because pytorch uses channel, height, width
                            # https://github.com/isl-org/MiDaS/issues/79
                            target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                            lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                            concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                            vis.vis_image('Sample %d' % k, concat_img)
                    model.train()
                scheduler.step()
                    
                if cur_epochs >= opts.total_epochs:
                    save_ckpt('checkpoints/latest_%s_%s_os%d_%s.pth' %
                              (opts.model, opts.dataset, opts.output_stride, opts.model_name))
                    print("validation...")
                    model.eval()
                    val_score, ret_samples = validate(
                        opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                        ret_samples_ids=vis_sample_id)
                    print(metrics.to_str(val_score))
                    if val_score['Mean IoU'] > best_score:  # save best model
                        best_score = val_score['Mean IoU']
                        save_ckpt('checkpoints/best_%s_%s_os%d_%s.pth' %
                                  (opts.model, opts.dataset, opts.output_stride, opts.model_name))
                    # A: mail begin
                    local_time = get_time()         # A: for email
                    mail_reply = f"Training: round {opts.training_round}\nEpoch {cur_epochs}/{opts.total_epochs}, iteration {cur_itrs} at {local_time}.\nLoss: {np_loss}\nOverall Acc: {val_score['Overall Acc']}\nMean Acc: {val_score['Mean Acc']}\nFreqW Acc: {val_score['FreqW Acc']}\nMean IoU: {val_score['Mean IoU']}\nClass IoU: {val_score['Class IoU']}"
                    for dest in get_mail_addresses():
                        send_mail(server = mail_server["server_address"],
                                  port = mail_server["server_port"],
                                  user = mail_server["user"],
                                  password = mail_server["password"],
                                  to = dest,
                                  subject = opts.mail_subject,
                                  body = mail_reply)
                    # A: mail end 
                    # A: added these to keep logs instead of using Visdom
                    output = [f"Epoch: {cur_epochs}", 
                              f"Iteration: {cur_itrs}", 
                              f"Loss: {np_loss}",
                              f"Overall Acc: {val_score['Overall Acc']}",
                              f"Mean Acc: {val_score['Mean Acc']}",
                              f"FreqW Acc: {val_score['FreqW Acc']}",
                              f"Mean IoU: {val_score['Mean IoU']}",
                              f"Class IoU: {val_score['Class IoU']}"]
                    with open(f'./logs/{opts.model_name}_logs.txt', 'w') as f:
                        f.writelines('\n'.join(output))
                    f.close()
                    
                    sys.exit(0)
                    return
        except Exception as e:
            print(traceback.format_exc())
            sys.exit(1)

if __name__ == '__main__':
    main()
