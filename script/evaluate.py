from data.taskonomy.taskonomy_dataset_s3 import TaskonomyDatasetS3
from matplotlib import pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import transformers
from tqdm import tqdm
import numpy as np
import os
import pickle
import cv2
import json
import argparse


from torchvision.utils import make_grid, save_image


from data.nyuv2_same_batch import NYUv2SameBatchDataset
from model.swin_transformer import SwinTransformer
from loss.losses import berHuLoss
from loss.metrics import iou_pytorch, eval_depth
from data.nyuv2 import NYUv2Dataset


def get_config():
    parser = argparse.ArgumentParser(description='Train the network')
    parser.add_argument('--config', help='train config file path')

    args = parser.parse_args()

    with open(args.config, "r") as jsonfile:
        config = json.load(jsonfile)

    return config


def get_dataloaders(tasks, batch_size, setting="nyu", task=None):

    if setting == "taskonomy":

        test_dataset = TaskonomyDatasetS3(
            tasks=["rgb", "segment_semantic", "depth_euclidean"], split="val", variant="tiny", image_size=224)

        g = torch.Generator()
        g.manual_seed(61)

        k_samples = 16*100
        perm = torch.randperm(len(test_dataset), generator=g)
        idx = perm[:k_samples].tolist()

        subset_dataset_test = torch.utils.data.Subset(test_dataset,  idx)

        dataloader = DataLoader(subset_dataset_test,
                                batch_size=batch_size, shuffle=False)

        return dataloader

    if setting == "nyu":

        IMAGE_SIZE = (480, 640)

        test_t = torch.nn.Sequential(
            transforms.CenterCrop(480), transforms.Resize(224))
        train_t_input_image = torch.nn.Sequential(transforms.ColorJitter(brightness=(
            0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1)))

        test_dataset = NYUv2SameBatchDataset(root="./data/nyuv2", tasks=tasks, download=False, train=False,
                                             rgb_transform=test_t, seg_transform=test_t, sn_transform=test_t, depth_transform=test_t)

        dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        return dataloader

    if setting == "nyu_single_task":

        IMAGE_SIZE = (480, 640)

        test_t = torch.nn.Sequential(
            transforms.CenterCrop(480), transforms.Resize(224))
        train_t_input_image = torch.nn.Sequential(transforms.ColorJitter(brightness=(
            0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1)))

        test_dataset = NYUv2Dataset(root="./data/nyuv2", tasks=tasks, download=False, train=False,
                                    rgb_transform=test_t, seg_transform=test_t, sn_transform=test_t, depth_transform=test_t)

        if task == "segmentation":
            test_dataset = torch.utils.data.Subset(
                test_dataset,  range(len(test_dataset)//2))

        if task == "depth":
            test_dataset = torch.utils.data.Subset(
                test_dataset,  range(len(test_dataset)//2, len(test_dataset)))

        dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        return dataloader


def calc_seg_metrics(logit_task, label_task):

    max_labels = torch.argmax(logit_task, dim=1, keepdim=True)
    iou = iou_pytorch(max_labels, label_task)

    return max_labels, iou


def disp2meters(d):
    return (65536.0 / d - 1) / 1e4


def load_model(model, PATH, device):
    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    return model


def evaluate(model, dataloader, device, task=None):
    test_loss = 0
    epoch_ious = []
    epoch_eval_depths_d1 = []

    epoch_loss_seg_test = []
    epoch_loss_depth_test = []

    model.eval()
    for i, (img, label, task_id) in enumerate(dataloader, 0):

        img = img.view((-1, 3, 224, 224)).to(device)
        label = label.view((-1, 1, 224, 224)).to(device)
        task_id = task_id.view(-1).to(device)

        if task is not None:
            task_id = torch.zeros_like(task_id)

        logits, unique_task_ids_list = model(img, task_id)

        loss = 0

        for j, unique_task_id in enumerate(unique_task_ids_list):

            task_id_filter = task_id == unique_task_id

            logit_task = logits[j]
            label_task = label[task_id_filter]
            B = logit_task.shape[0]

            if unique_task_id == 0 and task != "depth":

                label_task = label_task.long()

                max_labels, iou = calc_seg_metrics(logit_task, label_task)

                epoch_ious.append(iou.cpu().numpy())

            else:

                logit_task = torch.nn.functional.sigmoid(logit_task)*65535 + 1
                label_task = 65536.0 / (label_task + 1)

                evaluation = eval_depth(disp2meters(
                    logit_task), disp2meters(label_task))
                epoch_eval_depths_d1.append(evaluation["d1"])

    print("Mean IOU: ", np.mean(epoch_ious))
    print("D1: ", np.mean(epoch_eval_depths_d1))


def save_images(model, dataloader, device, num_images=1,  task=None, setting=None):

    img_count = 0
    for i, (img, label, task_id) in enumerate(dataloader, 0):

        img = img.view((-1, 3, 224, 224)).to(device)
        label = label.view((-1, 1, 224, 224)).to(device)
        task_id = task_id.view(-1).to(device)

        if task is not None:
            task_id = torch.zeros_like(task_id)

        logits, unique_task_ids_list = model(img, task_id)

        for j in range(len(img)):
            if len(logits) == 1:
                fig, axs = plt.subplots(1, 3, figsize=(12, 5))

                axs[0].imshow(torch.permute(img[j].cpu(), (1, 2, 0)))
                axs[0].set_xlabel('RGB Image')

                if task == "segmentation":
                    k = torch.argmax(logits[0][j], dim=0, keepdim=True)
                    k[0][-1][-1] = torch.max(label[j])
                    label[j][0][-1][-1] = torch.max(k)
                else:
                    k = disp2meters(torch.nn.functional.sigmoid(
                        logits[0][j])*65535 + 1)

                axs[1].imshow(torch.permute(label[j].cpu(), (1, 2, 0)))
                axs[1].set_xlabel(f'{task.capitalize()} Label')

                axs[2].imshow(k.detach().view(224, 224, 1).cpu())
                axs[2].set_xlabel(f'{task.capitalize()} Prediction')

                plt.savefig(f'./images/{img_count}.png')
                img_count += 1

            else:
                if j % 2 == 1:
                    continue

                c = j//2
                fig, axs = plt.subplots(1, 5, figsize=(20, 5))
                axs[0].imshow(torch.permute(img[j].cpu(), (1, 2, 0)))
                axs[0].set_xlabel('RGB Image')

                k = torch.argmax(logits[0][c], dim=0, keepdim=True)

                k[0][-1][-1] = 18 if setting == "taskonomy" else 13
                label[j][0][-1][-1] = 18 if setting == "taskonomy" else 13

                axs[1].imshow(torch.permute(label[j].cpu(), (1, 2, 0)))
                axs[1].set_xlabel('Segmentation Label')

                axs[2].imshow(k.detach().view(224, 224, 1).cpu())
                axs[2].set_xlabel('Segmentation Prediction')

                label[j+1][label[j+1] == 65535] = 0
                axs[3].imshow(torch.permute(label[j+1].cpu(), (1, 2, 0)))
                axs[3].set_xlabel('Depth Label')
                k2 = disp2meters(torch.nn.functional.sigmoid(
                    logits[1][c])*65535 + 1)
                axs[4].imshow(k2.detach().view(224, 224, 1).cpu())
                axs[4].set_xlabel('Depth Prediction')

                plt.savefig(f'./images/{img_count}.png')
                img_count += 1

            if img_count == num_images:
                return


def main():

    config = get_config()

    if config["setting"] != "nyu_single_task" and "task" in config.keys():
        print("Do not put task parameter on multitask networks!")
        return

    torch.manual_seed(61)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tasks = {0: "segmentation", 1: "depth"}
    print("Creating dataset...")
    dataloader = get_dataloaders(
        tasks, 16, config["setting"], config["task"] if "task" in config.keys() else None)

    print("Loading model...")

    tasks = ["segmentation", "depth"]
    task_classes = [14, 1] if config["setting"] != "taskonomy" else [18, 1]
    if config["setting"] == "nyu_single_task":
        tasks = [config["task"]]
        task_classes = [14 if config["task"] == "segmentation" else 1]

    model = SwinTransformer(img_size=224,
                            patch_size=4,
                            in_chans=3,
                            num_classes=21841,
                            embed_dim=96,
                            depths=[2, 2, 18, 2],
                            depths_decoder=[2, 2, 2, 2],
                            num_heads=[3, 6, 12, 24],
                            window_size=7,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=True,
                            drop_rate=0,
                            drop_rate_decoder=0.6,
                            drop_path_rate=0.2,
                            ape=False,
                            patch_norm=True,
                            use_checkpoint=False,
                            tasks=tasks,
                            task_classes=task_classes,
                            conditioned_blocks=config["conditioned_blocks"] if config["setting"] != "nyu_single_task" else [
                                [], [], [], []],
                            adapter=config["adapter"] if config["setting"] != "nyu_single_task" else False,
                            use_conditional_layer=config["use_conditional_layer_norm"] if config["setting"] == "nyu" else False)

    model = load_model(model, config["model_path"], device)

    print("Evaluating...")
    evaluate(model, dataloader, device,
             task=config["task"] if "task" in config.keys() else None)

    print("Saving Images...")
    save_images(model, dataloader, device, num_images=config["num_generated_images"],
                task=config["task"] if "task" in config.keys() else None, setting=config["setting"])


if __name__ == '__main__':
    main()
