import torch
import torchvision

image1 = torch.load('./image1.sprite')['sprite']
image2 = torch.load('./image2.sprite')['sprite']
torchvision.utils.save_image(image1,'./image1.png')
torchvision.utils.save_image(image2,'./image2.png')
