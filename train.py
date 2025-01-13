from tqdm import tqdm 
from  argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR


from datasets import init_dataloaders
from model.fg_classifier import FgClassifier
from utils_dir.bbox_to_label import bbox_to_label_multiple_classify_proto
from utils_dir.backbones_utils import load_backbone, extract_backbone_features


def get_argparse():
    parser = ArgumentParser()
    parser.add_argument('--train_root_dir', type=str)
    parser.add_argument('--train_annotations_file', type=str)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_step_size', type=int, default=30)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=50)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--backbone_type', type=str, default='dinov2')
    parser.add_argument('--num_layers', type=int, default=2)

    args = parser.parse_args()

    return args


def train(args):
    
    writer = SummaryWriter('/hy-tmp/TuneDinov2/tensorboard')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    backbone = load_backbone(args.backbone_type, device)
    model = FgClassifier()
    model.to(device)

    # 加载dataloader  
    train_dataloader = init_dataloaders(args)

    # 加载优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    torch.autograd.set_detect_anomaly(True)
    scheduler = MultiStepLR(optimizer, milestones=[10, 100], gamma=args.lr_decay)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                             desc=f'Val Epoch {epoch + 1}/{args.num_epochs}', leave=False):
            
            images, boxes, labels, _ = batch

            feats = extract_backbone_features(images, backbone, "dinov2")
            batch_patch_label = bbox_to_label_multiple_classify_proto(feats, boxes, labels)
            batch_patch_label =  batch_patch_label.to(device)

            outputs = model(feats)

            loss = criterion(outputs, batch_patch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer

        scheduler.step()
        
        # 隔10 epoch，保存一次训练结果
        if epoch % 10 == 0:
            torch.save


        
        # 当训练到最后一个epoch, 保存最后的结果

if __name__ == '__main__':
    # args = get_argparse()
    # train(args)





    