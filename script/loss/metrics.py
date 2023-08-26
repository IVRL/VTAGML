import functools

import torch.nn.functional as F
import torch


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    """
    Taken from:
    https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy/comments
    """

    SMOOTH = 1e-6
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)

    intersection = (outputs & labels).float().sum(
        (1, 2))  # Will be zero if Truth=0 or Prediction=0
    
    union = (outputs | labels).float().sum(
        (1, 2))         # Will be zzero if both are 0

    # We smooth our devision to avoid 0/0
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    # This is equal to comparing with thresolds
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10

    # Or thresholded.mean() if you are interested in average across the batch
    return thresholded.mean()



def eval_depth(pred, target):
    
    """
    Taken from:
    https://github.com/wl-zhao/VPD/blob/main/depth/utils_depth/metrics.py
    """

    rmse_temp = 0
    d1_temp = 0
    
    for current_target, current_pred in zip(target, pred):
        ##assert current_gt_sparse.shape == current_pred.shape

        thresh = torch.max((current_target / current_pred), (current_pred / current_target))

        d1 = (thresh < 1.25).float().mean()#torch.sum(thresh < 1.25).float().mean()# / len(thresh)
        #d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
        #d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

        diff = current_pred - current_target
        diff_log = torch.log(current_pred) - torch.log(current_target)

        #abs_rel = torch.mean(torch.abs(diff) / target)
        #sq_rel = torch.mean(torch.pow(diff, 2) / target)

        rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
        rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

        #log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
        #silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

        #return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(),
        #        'sq_rel': sq_rel.item(), 'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 
        #        'log10':log10.item(), 'silog':silog.item()}
        
        rmse_temp += rmse
        d1_temp += d1

    return {'d1': d1_temp.item()/len(pred),'rmse': rmse_temp.item()/len(pred)}
