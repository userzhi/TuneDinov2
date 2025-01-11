
import numpy as np

import math
import torch

def bbox_to_label_binary_classify_attn(feats, bbox, labels):
    """
       将bbox内的patch打上相应标签
       对于batch_size=4来说, 
           feats.shape: torch.Size([4, 1849, 1024])
           bbox.shape: torch.Size([4, 500, 4])
           labels.shape: torch.Size([4, 500])
    """
    patch_size = 16
    # 1、确定bbox坐标以及对应的标签
    for i, (box, label) in enumerate(zip(bbox, labels)):

        # keep = box[0] != [0.0, 0.0, 0.0, 0.0]
        keep = torch.any(box[0] != 0.0)
        box = box[keep]
        label = label[keep]
 
        feat = feats[i]
        k, D = feat.shape
        h, w = np.sqrt(k)

        



    pass



def bbox_to_label_multiple_classify_attn(bbox, img):
    pass



def bbox_to_label_multiple_classify_propto(bbox, img):
    pass