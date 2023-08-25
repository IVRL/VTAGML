from data.taskonomy.taskonomy_dataset_s3 import TaskonomyDatasetS3
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

# Small adjustments are made to the Swin Transformer model for parallel run, not related to architecture
from model.swin_transformer_parallel import SwinTransformer
from loss.losses import berHuLoss
from loss.metrics import iou_pytorch, eval_depth
import transformers


unique_task_ids_list = [0,1]

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

def check_val(model, dataloader, criterion, index, tensorboard_name, seg_weight, depth_weight, device, tb_writer):
    test_loss = 0
    epoch_ious = []
    epoch_eval_depths_rmse = []
    epoch_eval_depths_d1 = []

    epoch_loss_seg_test = []
    epoch_loss_depth_test = []

    model.eval()
    
    for i, (img, label, task_id) in enumerate(dataloader, 0):

        img = img.view((-1, 3, 224, 224)).to(device)
        label = label.view((-1, 1, 224, 224)).to(device)
        task_id = task_id.view(-1).to(device)

        logits = model(img, task_id)

        loss = 0

        for j, unique_task_id in enumerate(unique_task_ids_list):


            task_id_filter = task_id == unique_task_id

            logit_task = logits[j]
            if logit_task is None:
                continue
                
            label_task = label[task_id_filter]
            B = logit_task.shape[0]


            if unique_task_id == 0:

                label_task = label_task.long()

                a = criterion[unique_task_id](logit_task.view(B,18,-1), label_task.long().view(B,-1))
                loss += a * seg_weight

                epoch_loss_seg_test.append(a.item() )

                max_labels = torch.argmax(logit_task, dim = 1, keepdim=True)

                iou = iou_pytorch(max_labels, label_task)

                epoch_ious.append(iou.cpu().numpy())

            else:
      
                logit_task = torch.nn.functional.sigmoid(logit_task)*65535 + 1
                label_task = 65536.0 / (label_task + 1)

                a = criterion[unique_task_id](logit_task, label_task, mask_val = 1.0)#* len(logit_task)
                loss += a* depth_weight
                epoch_loss_depth_test.append(a.item() )

                
                evaluation = eval_depth(disp2meters(logit_task), disp2meters(label_task))
                epoch_eval_depths_rmse.append(evaluation["rmse"])
                epoch_eval_depths_d1.append(evaluation["d1"])

        test_loss += loss.item()
        
    tb_writer.add_scalar(f"{tensorboard_name}/mid_test_loss", test_loss/len(dataloader) , index)
    tb_writer.add_scalar(f"{tensorboard_name}/mid_mean_iou", np.mean(epoch_ious) , index)
    tb_writer.add_scalar(f"{tensorboard_name}/mid_depth_rmse", np.mean(epoch_eval_depths_rmse) , index)
    tb_writer.add_scalar(f"{tensorboard_name}/mid_depth_d1", np.mean(epoch_eval_depths_d1) , index)
    tb_writer.add_scalar(f"{tensorboard_name}/mid_seg_loss_test", np.mean(epoch_loss_seg_test) , index)
    tb_writer.add_scalar(f"{tensorboard_name}/mid_depth_loss_test", np.mean(epoch_loss_depth_test) , index)

    
    

    
def train(model, train_loader, test_loader,mid_test_dataloader, optimizer, scheduler, criterion, epochs, tensorboard_name, seg_weight, depth_weight, start_epoch = 0, device = "cuda", tb_writer=None):

    model.train()
    
    train_losses = []
    test_losses = []
    ious = []
    eval_depths = []
    
    iters = len(train_loader)
    
            
    mid_iter_count = 0
    mid_iter_count = 30

    
    for e in tqdm(range(epochs), desc="epoch", position=0):
        
        epoch = e + start_epoch + 1
        
        epoch_loss = 0.0
        epoch_loss_seg = []
        epoch_loss_depth = []
        
        
        
        #model.train()
        train_ious = []
        train_depths_rmse = []
        train_depths_d1 = []
        
        
        for i, (img, label, task_id) in tqdm(enumerate(train_loader, 0), desc="iter", position=1, leave=False):
            model.train()
              
            img = img.view((-1, 3, 224, 224)).to(device)
            label = label.view((-1, 1, 224, 224)).to(device)
            task_id = task_id.view(-1).to(device)
            

            logits = model(img, task_id)
      
            loss = 0
            
            
            for j, unique_task_id in enumerate(unique_task_ids_list):
                
                task_id_filter = task_id == unique_task_id
                
                
                logit_task = logits[j]
                if logit_task is None:
                    continue
                label_task = label[task_id_filter]
                
                B = logit_task.shape[0]
                                
                if unique_task_id == 0:
                    label_task = label_task.long()

                    a = criterion[unique_task_id](logit_task.view(B,18,-1), label_task.view(B,-1)) #* len(logit_task)
                    loss += a * seg_weight
                    epoch_loss_seg.append(a.item() )
                    
                    if epoch%1==0 :
                        max_labels = torch.argmax(logit_task, dim = 1, keepdim=True)
                        iou = iou_pytorch(max_labels, label_task)
                        train_ious.append(iou.cpu().numpy())
                    
                    
                else:
                    logit_task = torch.nn.functional.sigmoid(logit_task)*65535 + 1
                    label_task = 65536.0 / (label_task + 1)
                    
                    a = criterion[unique_task_id](logit_task, label_task, mask_val = 1.0) #* len(logit_task)
                    loss += a* depth_weight
                    epoch_loss_depth.append(a.item())
                    
                    if epoch%1==0:
                        evaluation = eval_depth(disp2meters(logit_task), disp2meters(label_task))
                        train_depths_rmse.append(evaluation["rmse"])
                        train_depths_d1.append(evaluation["d1"])
                    
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            epoch_loss += loss.item()
            
            scheduler.step()
                
                
            if i % 1000 == 0:
                check_val(model, mid_test_dataloader, criterion, mid_iter_count*1000, tensorboard_name, seg_weight, depth_weight, device, tb_writer)
                mid_iter_count += 1
                
                tb_writer.add_scalar(f"{tensorboard_name}/mid_train_loss", epoch_loss/ (i+1) , mid_iter_count*1000)
                
                torch.save({
                    'epoch': epoch,
                    'iter': i,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    }, f"{tensorboard_name}.pt")
          

            
        if epoch % 1==0:
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
                task_id = task_id.view(-1).to(device)
                

                logits = model(img, task_id)
           

                loss = 0

                for j, unique_task_id in enumerate(unique_task_ids_list):


                    task_id_filter = task_id == unique_task_id

                    logit_task = logits[j]
                    if logit_task is None:
                        continue
                    label_task = label[task_id_filter]
                    B = logit_task.shape[0]


                    if unique_task_id == 0:

                        label_task = label_task.long()

                        a = criterion[unique_task_id](logit_task.view(B,18,-1), label_task.long().view(B,-1))
                        loss += a * seg_weight

                        epoch_loss_seg_test.append(a.item() )
                        
                        if epoch%1==0:

                            max_labels = torch.argmax(logit_task, dim = 1, keepdim=True)

                            iou = iou_pytorch(max_labels, label_task)
                         
                            epoch_ious.append(iou.cpu().numpy())

                    else:
                        
                        
                        logit_task = torch.nn.functional.sigmoid(logit_task)*65535 + 1
                        label_task = 65536.0 / (label_task + 1)
                        
                        a = criterion[unique_task_id](logit_task, label_task, mask_val = 1.0)#* len(logit_task)
                        loss += a* depth_weight

                        epoch_loss_depth_test.append(a.item() )
                        
                        if epoch%1==0:
                            evaluation = eval_depth(disp2meters(logit_task), disp2meters(label_task))
                            epoch_eval_depths_rmse.append(evaluation["rmse"])
                            epoch_eval_depths_d1.append(evaluation["d1"])
                        
           

                test_loss += loss.item()
           
        
        
        if epoch % 1==0:
                    
            
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
                                     
            if epoch%1 == 0:
                tb_writer.add_scalar(f"{tensorboard_name}/train_mean_iou", np.mean(train_ious) , epoch)
                tb_writer.add_scalar(f"{tensorboard_name}/train_depth_rmse", np.mean(train_depths_rmse) , epoch)
                tb_writer.add_scalar(f"{tensorboard_name}/train_depth_d1", np.mean(train_depths_d1) , epoch)
                
            
            
        if epoch % 1 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, f"{tensorboard_name}.pt")
        
    return train_losses, test_losses, ious, eval_depths


def load_model(model, optimizer, scheduler, PATH):
    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, scheduler, epoch


def get_dataloaders(tasks, batch_size):

    train_dataset = TaskonomyDatasetS3(tasks=["rgb", "segment_semantic","depth_euclidean"], split="train", variant="tiny", image_size=224)
    test_dataset = TaskonomyDatasetS3(tasks=["rgb", "segment_semantic","depth_euclidean"], split="val", variant="tiny", image_size=224)

    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))
    
    g = torch.Generator()
    g.manual_seed(61)

    k_samples = 16*100
    perm = torch.randperm(len(test_dataset), generator=g)
    idx = perm[:k_samples].tolist()

    subset_dataset_test = torch.utils.data.Subset(test_dataset,  idx)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    mid_test_dataloader = DataLoader(subset_dataset_test, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, mid_test_dataloader, test_dataloader
    
class _CustomDataParallel(torch.nn.DataParallel):
    def __init__(self, model):
        super(_CustomDataParallel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(_CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def main():
    
    config = get_config()
    
    tb_writer = SummaryWriter(f'runs/{config["experiment_name"]}')
    
    torch.manual_seed(61)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    tasks = {0:"segmentation", 1:"depth"}
    
    batch_size = config["batch_size"]
    print("Creating datasets...")
    train_dataloader, mid_test_dataloader, test_dataloader = get_dataloaders(tasks, batch_size)
    

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
                                tasks = ["segmentation", "depth"],
                                task_classes = [18, 1],
                                 conditioned_blocks = config["conditioned_blocks"],
                                adapter=config["adapter"])


    epochs = config["epochs"]
    

    optimizer = optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.98))#, weight_decay=0.001)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 4e-5, epochs = epochs, steps_per_epoch = len(train_dataloader), pct_start = 0.1) 


    if config["continue_training"]:

        model, optimizer, scheduler, start_epoch = load_model(model, optimizer, scheduler, config["experiment_name"]+".pt")
        print("Continue model loaded")

    else:
        start_epoch = -1
        model.load_state_dict(torch.load('./pretrained/swin_small_patch4_window7_224_22k.pth')['model'],strict=False)
        model = model.to(device)
        print("Model loaded")



    freeze_encoder_layers(model, conditioned_blocks = config["conditioned_blocks"], frozen_encoder = config["frozen_encoder"])
    model = model.to(device)

    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    print("Model on cuda:",next(model.parameters()).is_cuda)

    criterion = []
    segmentation_criteon = torch.nn.CrossEntropyLoss(ignore_index = 0) 
    criterion.append(segmentation_criteon)

    depth_criterion = berHuLoss()
    criterion.append(depth_criterion)
    
    print("Training",config["experiment_name"],"...")


    train(model, train_dataloader, test_dataloader, mid_test_dataloader, optimizer, scheduler, criterion, epochs, config["experiment_name"], config["seg_weight"], config["depth_weight"], start_epoch, device, tb_writer)


if __name__ == '__main__':
    main()