
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
    patch_size = 14
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



def bbox_to_label_multiple_classify_proto(feats, bboxes, labels):
    """
       将bbox内的patch打上相应标签
       对于batch_size=4来说, 
           feats.shape: torch.Size([4, 1849, 1024])
           bbox.shape: torch.Size([4, 500, 4])
           labels.shape: torch.Size([4, 500])
    """
    patch_size = 14
    B, K= feats.shape[0], feats.shape[1]

    # 1、确定bbox坐标以及对应的标签
    batch_patch_label = []
    for i, (boxes, label) in enumerate(zip(bboxes, labels)):
        
        #------------------------- 去除掉非gt boxes -------------------------
        keep = torch.any(box[0] != 0.0)         
        boxes = boxes[keep]
        label = label[keep]
 
        feat = feats[i]
        k, D = feat.shape
        h, w = np.sqrt(k)

        patch_label = torch.full((h, w), -1)
        #------------------------- 找到在有效box内的patch -------------------------
        for b, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            start_patch_x = int(round(x1 / patch_size)) if int(round(x1 / patch_size)) >= 0 else 0
            start_patch_y = int(round(y1 / patch_size)) if int(round(y1 / patch_size)) >= 0 else 0
            end_patch_x = int(round(x2 / patch_size)) if int(round(x2 / patch_size)) >= 0 else 0
            end_patch_y = int(round(y2 / patch_size)) if int(round(y2 / patch_size)) >= 0 else 0

            #------------------------- 为patch打上gt label -------------------------
            """切记: 衡量行时是y坐标"""
            for row in range(end_patch_y - 1, start_patch_y):
                for col in range(end_patch_x - 1, start_patch_x):
                    patch_label[row, col] = label[b]

        patch_label = patch_label.reshape(h, w)
        batch_patch_label.append(patch_label)

    batch_patch_label = torch.stack(batch_patch_label)
    assert batch_patch_label.shape == torch.Size([B, K])

    return batch_patch_label