import torch
import torch.nn as nn
import torchvision
from torchvision.io import read_image
import re

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
def init_encoder():
    # Load pretrained weights
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    # Load ViT with pretrained weights
    model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights(weights))
    # Remove classification head
    model.heads = Identity()
    model.eval()
    return model
    
def encode_image(model, path, image_size=(224, 224)):
    # ViT mean and std for normalizing 
    # see https://pytorch.org/vision/main/models/vision_transformer.html
    mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]])
    std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]])
    # Read image
    image = read_image(path).float()
    # If image is .png and is read with RGB or RGBA ignore all but first dimension
    if re.search('.png$', path) and not image.shape[0] == 1: image = image[0:1, :, :]
    # Resize image
    image = torchvision.transforms.functional.resize(image, size=image_size)
    # Expand image
    image = image.expand(3, image_size[0], image_size[1])
    # Normalize image
    image = ((image/255)-mean)/std
    # Encode image
    with torch.no_grad():
        if device.type == "cuda": image = image.to(device)
        encoded_image = model(image[None, :, :, :])
        if device.type == "cuda": encoded_image = encoded_image.cpu()
    return encoded_image