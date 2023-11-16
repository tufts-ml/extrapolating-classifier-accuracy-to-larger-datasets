import re
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.io import read_image
# Importing neuroimaging packages
import nibabel as nib

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

def encode_image(model, path, grayscale=True, image_size=(224, 224)):
    model_device = next(model.parameters()).device
    device = torch.device(model_device)
    # Default from https://pytorch.org/vision/main/models/vision_transformer.html
    mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]])
    std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]])
    # Read image
    image = read_image(path).float()
    # Make sure image dimensions are (1, w, h)
    if grayscale and len(image.shape) == 2: image = image[None, :, :]
    # For grayscale images read with RGB values all dimension will be the same
    if grayscale and len(image.shape) == 3 and not image.shape[0] == 1: image = image[0:1, :, :]
    # Center-crop the images using a window size equal to the length of the shorter edge and rescale them to (1, 224, 224)
    print(image.shape)
    center_crop = torchvision.transforms.CenterCrop(min(image.shape[-1], image.shape[-2]))
    image = center_crop(image)
    image = torchvision.transforms.functional.resize(image, size=image_size)
    # Expand image
    if grayscale: image = image.expand(3, image_size[0], image_size[1])
    # Normalize image
    image = ((image/255)-mean)/std
    # Encode image
    with torch.no_grad():
        if device.type == 'cuda': image = image.to(device)
        encoded_image = model(image[None, :, :, :])
        if device.type == 'cuda': encoded_image = encoded_image.cpu()
    return encoded_image

def strip_skull(image, minimum, maximum):
    image[image < minimum] = minimum
    image[image > maximum] = maximum
    return image

def encode_scan(model, path, image_size=(224, 224)):
    model_device = next(model.parameters()).device
    device = torch.device(model_device)
    # Default from https://pytorch.org/vision/main/models/vision_transformer.html
    mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]])
    std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]])
    # Read image
    image = torch.tensor(nib.load(path).get_fdata()).float()
    image = strip_skull(image, -100, 300)
    image = (image-100)/400
    # sub-OAS30246_sess-d1591_CT.nii.gz is formatted differently
    if len(image.shape) == 4: image = torch.concat((torch.flip(image[:,:,:,0], dims=[0, 2]), torch.flip(image[:,:,:,1], dims=[0, 2])), dim=2)
    # Swap image axes
    image = image.permute(2, 1, 0)
    # Rescale images to (N, 224, 224)
    image = torchvision.transforms.functional.resize(image, size=image_size)
    # Expand image
    image = image[:, None, :, :]
    image = image.expand(image.shape[0], 3, image_size[0], image_size[1])
    # Normalize image
    image = ((image)-mean)/std
    # Encode image
    with torch.no_grad():
        if device.type == 'cuda': image = image.to(device)
        encoded_image = model(image)
        if device.type == 'cuda': encoded_image = encoded_image.cpu()
    return encoded_image