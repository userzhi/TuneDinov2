import torch


def load_backbone(backbone_type):
    '''
    Load a pre-trained backbone model.

    Args:
        backbone_type (str): Backbone type
    '''
    if backbone_type == 'dinov2':
        # 分为两种情况，可以分别从github加载和本地加载
        path = '/hy-tmp/dinov2'  # 本地加载
        model = torch.hub.load(path, 'dinov2_vitl14', source='local')
        # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')  # github加载
    
    for name, paramater in model.named_parameters():
        paramater.requires_grad = False

    model.eval()

    return model


def prepare_image_for_backbone(input_tensor, backbone_type):
    '''
    Preprocess an image for the backbone model given an input tensor and the backbone type.

    Args:
        input_tensor (torch.Tensor): Input tensor with shape (B, C, H, W)
        backbone_type (str): Backbone type
    '''

    # Define mean and std for normalization depending on the backbone type
    mean = torch.tensor([0.485, 0.456, 0.406]).to(input_tensor.device) if 'dinov2' in backbone_type else torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(input_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(input_tensor.device) if 'dinov2' in backbone_type else torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(input_tensor.device)
    
    # Scale the values to range from 0 to 1
    input_tensor /= 255.0
    
    # Normalize the tensor
    normalized_tensor = (input_tensor - mean[:, None, None]) / std[:, None, None]
    return normalized_tensor


def extract_backbone_features(images, model, backbone_type, scale_factor=1):
    '''
    Extract features from a pre-trained backbone for any of the supported backbones.

    Args:
        images (torch.Tensor): Input tensor with shape (B, C, H, W)
        model (torch.nn.Module): Backbone model
        backbone_type (str): Backbone type
        scale_factor (int): Scale factor for the input images. Set to 1 for no scaling.
    '''
    # images = F.interpolate(images, scale_factor=scale_factor, mode='bicubic')   ## zhouzhi

    if 'dinov2' in backbone_type:
        with torch.no_grad():
            feats = model.forward_features(images)['x_prenorm'][:, 1:]

    return feats
