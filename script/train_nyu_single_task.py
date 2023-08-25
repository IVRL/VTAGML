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

from data.nyuv2 import NYUv2Dataset  
from model.swin_transformer import SwinTransformer
from loss.losses import berHuLoss
from loss.metrics import iou_pytorch, eval_depth


def get_config():
    parser = argparse.ArgumentParser(description='Train the network')
    parser.add_argument('--config', help='train config file path')
    
    args = parser.parse_args()
    
    with open(args.config, "r") as jsonfile:
        config = json.load(jsonfile)
    
    return config

def freeze_encoder_layers(
        model,
        conditioned_blocks= [[], [], [*range(12, 18)], []],
        unfrozen_modules=[
            "random_weight_matrix",
            "film.gb_weights",
            "ln_weight_modulation.gb_weights",
            "adapter",
            "task_type_embeddings",
            "patch_embed",
            "decoder",
            "bottleneck"
        ],
        frozen_encoder = False
    ):
        for name, param in model.named_parameters():
            param.requires_grad = not frozen_encoder
            
            for module in unfrozen_modules:
                if module in name:
                    param.requires_grad = True
                
            if name.startswith("layers"):
                splitted = name.split(".")
                
                if len(conditioned_blocks[int(splitted[1])]) > 0 and splitted[2]=="blocks"  and (int(splitted[3]) in conditioned_blocks[int(splitted[1])]):
                    param.requires_grad = True
            elif name.startswith("norm"):
                param.requires_grad = True

def disp2meters(d):
    return (65536.0 / d - 1 ) / 1e4

def calc_seg_metrics(logit_task, label_task):
                     
    max_labels = torch.argmax(logit_task, dim = 1, keepdim=True)
    iou = iou_pytorch(max_labels, label_task)
    
    return max_labels, iou

def train(model, train_loader, test_loader, optimizer, scheduler, criterion, epochs, tensorboard_name, start_epoch = 0, device = "cuda", tb_writer=None, task="segmentation"):

    # Training loop
    model.train()
        
    iters = len(train_loader)
        
    for e in tqdm(range(epochs)):
        
        
        epoch = e + start_epoch + 1
        
        epoch_loss = 0.0
        epoch_loss_seg = []
        epoch_loss_depth = []
        
        train_ious = []
        train_depths_rmse = []
        train_depths_d1 = []
        
        for i, (img, label, task_id) in enumerate(train_loader, 0):
            model.train()
              
            img = img.view((-1, 3, 224, 224)).to(device)
            label = label.view((-1, 1, 224, 224)).to(device)
            task_id = torch.zeros_like(task_id.view(-1).to(device))
            
            logits, unique_task_ids_list = model(img, task_id)
            
            loss = 0
            
            for j, unique_task_id in enumerate(unique_task_ids_list):
                
                task_id_filter = task_id == unique_task_id
                
                logit_task = logits[j]
                label_task = label[task_id_filter]
                                
                B = logit_task.shape[0]
                
                
                # Task is segmentation                
                if task == "segmentation":
                    label_task = label_task.long()

                    a = criterion[0](logit_task.view(B,14,-1), label_task.view(B,-1)) 
                    epoch_loss_seg.append(a.item())
                    
                    loss += a
                    
                    # compute metrics every 10 epochs
                    if epoch%10==0:
                        max_labels, iou = calc_seg_metrics(logit_task, label_task)
                        train_ious.append(iou.cpu().numpy())
                    
                else:
                    label_task = 65536.0 / (label_task + 1)
                    logit_task = torch.nn.functional.sigmoid(logit_task)*65535 + 1

                    a = criterion[1](logit_task, label_task) 
                    epoch_loss_depth.append(a.item())
                    
                    loss += a
                    
                    
                    
                    # compute metrics every 10 epochs
                    if epoch%10==0:
                        evaluation = eval_depth(disp2meters(logit_task), disp2meters(label_task))
                        
                        train_depths_rmse.append(evaluation["rmse"])
                        train_depths_d1.append(evaluation["d1"])
                    
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            epoch_loss += loss.item()
            
            scheduler.step()

            
        # Compute validation metrics every 5 epochs
        if epoch % 5==0:
            
            test_loss = 0
            epoch_ious = []
            epoch_eval_depths_rmse = []
            epoch_eval_depths_d1 = []

            epoch_loss_seg_test = []
            epoch_loss_depth_test = []

            model.eval()
            for i, (img, label, task_id) in enumerate(test_loader, 0):

                img = img.view((-1, 3, 224, 224)).to(device)
                label = label.view((-1, 1, 224, 224)).to(device)
                task_id = torch.zeros_like(task_id.view(-1).to(device))

                logits, unique_task_ids_list = model(img, task_id)

                loss = 0

                for j, unique_task_id in enumerate(unique_task_ids_list):


                    task_id_filter = task_id == unique_task_id

                    logit_task = logits[j]
                    label_task = label[task_id_filter]
                    B = logit_task.shape[0]

                    if task == "segmentation":

                        label_task = label_task.long()

                        a = criterion[0](logit_task.view(B,14,-1), label_task.long().view(B,-1))
                        epoch_loss_seg_test.append(a.item())
                        
                        loss += a 

                        max_labels, iou = calc_seg_metrics(logit_task, label_task)
                        
                        epoch_ious.append(iou.cpu().numpy())

                    else:

                        
                        logit_task = torch.nn.functional.sigmoid(logit_task)*65535 + 1
                        label_task = 65536.0 / (label_task + 1)
                        
                        a = criterion[1](logit_task, label_task)
                        epoch_loss_depth_test.append(a.item())
                        
                        loss += a
          
                        evaluation = eval_depth(disp2meters(logit_task), disp2meters(label_task))
                        epoch_eval_depths_rmse.append(evaluation["rmse"])
                        epoch_eval_depths_d1.append(evaluation["d1"])

                test_loss += loss.item()                

            tb_writer.add_scalar(f"{tensorboard_name}/learning_rate", scheduler.get_last_lr()[0] , epoch)
            tb_writer.add_scalar(f"{tensorboard_name}/train_loss", epoch_loss/ len(train_loader) , epoch)
            tb_writer.add_scalar(f"{tensorboard_name}/test_loss", test_loss/len(test_loader) , epoch)
            tb_writer.add_scalar(f"{tensorboard_name}/mean_iou", np.mean(epoch_ious) , epoch)
            tb_writer.add_scalar(f"{tensorboard_name}/depth_rmse", np.mean(epoch_eval_depths_rmse) , epoch)
            tb_writer.add_scalar(f"{tensorboard_name}/depth_d1", np.mean(epoch_eval_depths_d1) , epoch)
            tb_writer.add_scalar(f"{tensorboard_name}/seg_loss", np.mean(epoch_loss_seg) , epoch)
            tb_writer.add_scalar(f"{tensorboard_name}/depth_loss", np.mean(epoch_loss_depth) , epoch)
            tb_writer.add_scalar(f"{tensorboard_name}/seg_loss_test", np.mean(epoch_loss_seg_test) , epoch)
            tb_writer.add_scalar(f"{tensorboard_name}/depth_loss_test", np.mean(epoch_loss_depth_test) , epoch)

            # Save training metrics every 10 epochs
            if epoch%10 == 0:
                tb_writer.add_scalar(f"{tensorboard_name}/train_mean_iou", np.mean(train_ious) , epoch)
                tb_writer.add_scalar(f"{tensorboard_name}/train_depth_rmse", np.mean(train_depths_rmse) , epoch)
                tb_writer.add_scalar(f"{tensorboard_name}/train_depth_d1", np.mean(train_depths_d1) , epoch)
                
            
            
        # save the model every 500 epochs
        if epoch % 500 == 0 or epoch == ((epochs)-1):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, f"{tensorboard_name}.pt")


def load_model(model, optimizer, scheduler, PATH):
    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, scheduler, epoch

def get_dataloaders(tasks, task, batch_size):
    
    IMAGE_SIZE = (480, 640)
    
    train_t = torch.nn.Sequential(transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), transforms.RandomHorizontalFlip())
    test_t = torch.nn.Sequential(transforms.CenterCrop(480), transforms.Resize(224))
    train_t_input_image = torch.nn.Sequential(transforms.ColorJitter(brightness=(0.8, 1.2),contrast =(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1,0.1)))

    train_dataset = NYUv2Dataset(root="./data/nyuv2", tasks=tasks, download=False, train=True,
          rgb_transform=train_t, rgb_transform2=train_t_input_image, seg_transform=train_t, sn_transform=train_t, depth_transform=train_t)

    test_dataset = NYUv2Dataset(root="./data/nyuv2", tasks=tasks, download=False, train=False,
          rgb_transform=test_t, seg_transform=test_t, sn_transform=test_t, depth_transform=test_t)
    
    if task == "segmentation":
        train_dataset = torch.utils.data.Subset(train_dataset,  range(len(train_dataset)//2))
        test_dataset = torch.utils.data.Subset(test_dataset,  range(len(test_dataset)//2))
        
    if task == "depth":
        
        train_dataset = torch.utils.data.Subset(train_dataset,  range(len(train_dataset)//2, len(train_dataset)))
        test_dataset = torch.utils.data.Subset(test_dataset,  range(len(test_dataset)//2, len(test_dataset)))
        
    
    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    return train_dataloader, test_dataloader
        
        
    
def main():
    # default `log_dir` is "runs" - we'll be more specific here
    
    config = get_config()
    
    tb_writer = SummaryWriter(f'runs/{config["experiment_name"]}')
    
    torch.manual_seed(61)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    tasks = {0:"segmentation", 1:"depth"}
    batch_size = config["batch_size"]
    
    print("Creating datasets...")
    train_dataloader, test_dataloader = get_dataloaders(tasks, config["task"], batch_size)
    
    print("Loading model...")

    model = SwinTransformer(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=21841,
                                embed_dim=96,
                                depths=[2, 2, 18, 2 ],
                                depths_decoder =[2, 2, 2, 2 ],
                                num_heads=[ 3, 6, 12, 24 ],
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
                                tasks = [config["task"]],
                                task_classes = [14 if config["task"]=="segmentation" else 1],
                                conditioned_blocks = [[],[],[],[]],
                                adapter = False,
                                use_conditional_layer = False)

    epochs = config["epochs"]
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.98))
        

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 4e-5, epochs = epochs, steps_per_epoch = len(train_dataloader), pct_start = 0.1)  

    scheduler_batch_step = True
    use_scheduler = True

    if config["continue_training"]:
        model, optimizer, scheduler, start_epoch = load_model(model, optimizer, scheduler, config["experiment_name"]+".pt")
        print("Continue model loaded")

    else:
        start_epoch = -1
        model.load_state_dict(torch.load('./pretrained/swin_small_patch4_window7_224_22k.pth')['model'],strict=False)

        model = model.to(device)
        print("Pretrained model loaded")

    
    freeze_encoder_layers(model, conditioned_blocks = [[],[],[],[]], frozen_encoder = config["frozen_encoder"])
    model = model.to(device)

    criterion = []
    segmentation_criteon = torch.nn.CrossEntropyLoss()
    criterion.append(segmentation_criteon)

    depth_criterion = berHuLoss()
    criterion.append(depth_criterion)
   
    print("Training",config["experiment_name"],"...")

          
    train(model, train_dataloader, test_dataloader, optimizer, scheduler, criterion, epochs, config["experiment_name"], start_epoch, device, tb_writer, config["task"])


if __name__ == '__main__':
    main()