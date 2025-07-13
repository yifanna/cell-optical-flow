import torch
from .position import PositionEmbeddingSine
import torch.nn.functional as F

def split_feature(feature, num_splits=2, channel_last=False):
    original_size = feature.shape[2:]  # 记录原始H和W
    if channel_last:  # [B, H, W, C]
        b, h, w, c = feature.size()
        pad_h = (num_splits - h % num_splits) % num_splits
        pad_w = (num_splits - w % num_splits) % num_splits
        feature = F.pad(feature, (0, pad_w, 0, pad_h), mode='constant', value=0)
        # Rest of the code...
    else:  # [B, C, H, W]
        b, c, h, w = feature.size()
        pad_h = (num_splits - h % num_splits) % num_splits
        pad_w = (num_splits - w % num_splits) % num_splits
        feature = F.pad(feature, (0, pad_w, 0, pad_h), mode='constant', value=0)
        # Rest of the code...

    return feature, original_size
def merge_splits(splits, original_size, num_splits=2, channel_last=False):
    if channel_last:  # [B*K*K, H/K, W/K, C]
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits
        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        merge = splits.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            new_b, num_splits * h, num_splits * w, c)
        merge = merge[:, :original_size[0], :original_size[1], :]  # Adjust size
    else:  # [B*K*K, C, H/K, W/K]
        b, c, h, w = splits.size()
        new_b = b // num_splits // num_splits
        splits = splits.view(new_b, num_splits, num_splits, c, h, w)
        merge = splits.permute(0, 3, 1, 4, 2, 5).contiguous().view(
            new_b, c, num_splits * h, num_splits * w)
        merge = merge[:, :, :original_size[0], :original_size[1]]  # Adjust size

    return merge
def normalize_img(img0, img1):
    # loaded images are in [0, 255]
    # normalize by ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
    img0 = (img0 / 255. - mean) / std
    img1 = (img1 / 255. - mean) / std

    return img0, img1
def feature_add_position(feature0, feature1, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)
    if attn_splits > 1:
        feature0_splits, original_size = split_feature(feature0, num_splits=attn_splits)
        feature1_splits, _ = split_feature(feature1, num_splits=attn_splits)
        position = pos_enc(feature0_splits)
        feature0_splits = feature0_splits + position
        feature1_splits = feature1_splits + position
        feature0 = merge_splits(feature0_splits, original_size, num_splits=attn_splits)
        feature1 = merge_splits(feature1_splits, original_size, num_splits=attn_splits)
    else:
        position = pos_enc(feature0)
        feature0 = feature0 + position
        feature1 = feature1 + position

    return feature0, feature1