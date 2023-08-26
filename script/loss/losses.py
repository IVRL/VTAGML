import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.cuda.amp as amp
    
class berHuLoss(nn.Module):
    def __init__(self):
        """
        https://github.com/lhoyer/improving_segmentation_with_selfsupervised_depth/
        """
        super(berHuLoss, self).__init__()

    
    def make_valid_mask(self, tens, mask_val, conf=1e-7):
        
        valid_mask = (tens > (mask_val+conf) ) | (tens < (mask_val-conf))
        
        return valid_mask
        
    
    def forward(self, inp, target, apply_log=False, threshold=.2, mask_val=None): 
        if apply_log:
            inp, target = torch.log(1 + inp), torch.log(1 + target)
            
        if mask_val is None:
            valid_mask = (target > 0).detach()
        else:
            valid_mask = self.make_valid_mask(target, mask_val)

        absdiff = torch.abs(target - inp) * valid_mask #* mask
        C = threshold * torch.max(absdiff).item()
        loss = torch.mean(torch.where(absdiff <= C,
                                      absdiff,
                                      (absdiff * absdiff + C * C) / (2 * C)))
        return loss