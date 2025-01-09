
from typing import int

import torch
from torch import nn
from torch.nn import functional as F

class FgClassifier(nn.Module):
    """ 
       通过将dinov2的backbone frozen, 取最后一层的输出进行后续前景背景二值分类

    """
    def __init__(self, input_dim: int, hidden_dim: int, 
                 output_dim: int, num_layers: int, dp: float=0.1, sigmoid_output: bool = False,):
        super().__init__()

        self.num_layers = num_layers
        mid_layer = [hidden_dim] * (num_layers - 1)
        """# 创建一个列表mid_layer，表示所有隐藏层的大小, 每个隐藏层都有相同的维度hidden_dim, 并且有num_layers - 1个隐藏层"""

        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + mid_layer, mid_layer + [output_dim])
        )
        """
           通过zip函数将输入层、隐藏层和输出层的维度配对，并为每一层创建一个nn.Linear层
           nn.Linear(n, k)表示一个全连接层，n是输入维度，k是输出维度
           ModuleList用来存储所有层，以便在forward函数中访问
        """
        self.dp = dp
        self.sigmoid_output = sigmoid_output
        self.linear = nn.Linear(1024, 1024)
    
    
    def forward(self, x):
        
        x = self.linear(x)

        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            """如果当前层不是最后一层，使用ReLU激活函数"""
            if i < self.num_layers - 1:  # Optional: Avoid dropout after the last layer
                x = F.dropout(x, self.p, training=self.training)

        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x