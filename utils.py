# utils.py

import torch
import numpy as np
from PIL import Image
import streamlit as st

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.step = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU()
        )

    def forward(self, x):
        return self.step(x)

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义左侧编码器的操作
        self.layer1 = ConvBlock(1,64)
        self.layer2 = ConvBlock(64,128)
        self.layer3 = ConvBlock(128,256)
        self.layer4 = ConvBlock(256,512)
        
        # 定义右侧解码器的操作
        self.layer5 = ConvBlock(256+512,256)
        self.layer6 = ConvBlock(128+256,128)
        self.layer7 = ConvBlock(64+128,64)
        
        #最后一个卷积
        self.layer8  = torch.nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1,padding=0,stride=1)
        
        # 定一些其他操作
        # 池化
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        #上采样
        self.upsample = torch.nn.Upsample(scale_factor=2,mode='bilinear')
        # sigmoid
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self,x):
        # 对输入数据进行处理
        
        # 定义下采样部分
        
        # input:1X256x256, output: 64x256x256
        x1 = self.layer1(x)
        # input:64x256x256, output: 64 x 128 x 128
        x1_p = self.maxpool(x1)
        
        # input:  64 x 128 x 128 , output: 128 x 128 x 128
        x2 = self.layer2(x1_p)
        # input:128 x 128 x 128 , output: 128 x 64 x 64
        x2_p = self.maxpool(x2)
        
        # input: 128 x 64 x 64, output: 256 x 64 x 64
        x3 = self.layer3(x2_p)
        #input:256 x 64 x 64, output: 256 x 32 x 32
        x3_p = self.maxpool(x3)
        
        #input: 256 x 32 x 32, output: 512 x 32 x 32
        x4 = self.layer4(x3_p)
        
        
        
        # 定义上采样
        # input: 512 x 32 x 32，output: 512 x 64 x 64
        x5 = self.upsample(x4)
        # 拼接,output: 768x 64 x 64
        x5 = torch.cat([x5,x3],dim=1)
        # input: 768x 64 x 64,output: 256 x 64 x 64
        x5 = self.layer5(x5)
        
        # input: 256 x 64 x 64,output: 256 x 128 x 128
        x6  = self.upsample(x5)
        # 拼接,output: 384 x 128 x 128
        x6 = torch.cat([x6,x2],dim=1)
        # input: 384 x 128 x 128, output: 128 x 128 x 128
        x6 = self.layer6(x6)
        
        
        # input:128 x 128 x 128, output: 128 x 256 x 256
        x7 = self.upsample(x6)
        # 拼接, output: 192 x 256 x256
        x7 = torch.cat([x7,x1],dim=1)
        # input: 192 x 256 x256, output: 64 x 256 x 256
        x7 = self.layer7(x7)
        
        # 最后一次卷积,input: 64 x 256 x 256, output: 1 x 256 x 256
        x8 = self.layer8(x7)
        
        #sigmoid
        # x9= self.sigmoid(x8)
        
        
        
        return x8
        
        
        

# Initialize and load the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load('./save_model/unet_best.pt'))
model.eval()

@st.cache_resource
def segment_single_image(pil_image):
    """
    Segment a single image using a pre-trained U-Net model.

    Parameters:
        pil_image (PIL.Image.Image): The input image in PIL format.

    Returns:
        PIL.Image.Image: The segmented image.
    """
    # Convert the PIL image to a NumPy array
    input_image = np.array(pil_image.convert('L'))

    # Normalize the image
    input_image = input_image / 255.0

    # Convert to tensor and expand dimensions
    input_tensor = np.expand_dims(input_image, axis=(0, 1))
    input_tensor = torch.tensor(input_tensor).to(device, dtype=torch.float32)

    # Forward pass
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Post-process the output
    output_mask = (output_tensor.detach().cpu().numpy()[0][0] > 0.5).astype(np.uint8)

    # Convert the NumPy array back to PIL image
    segmented_image = Image.fromarray((output_mask * 255).astype(np.uint8), 'L')

    return segmented_image

# utils.py
# ... (existing imports and model definitions)

from cv2 import VideoWriter, VideoWriter_fourcc

@st.cache_resource
def segment_images_to_video(pil_images, video_path, fps=30):
    """
    Segment a list of images and save the output as a video.

    Parameters:
        pil_images (list of PIL.Image.Image): List of input images in PIL format.
        video_path (str): Path where the output video will be saved.
        fps (int): Frames per second for the output video.

    Returns:
        None
    """
    # Initialize video writer
    width, height = pil_images[0].size
    fourcc = VideoWriter_fourcc(*'mp4v')
    video = VideoWriter(video_path, fourcc, float(fps), (width, height), isColor=False)

    for pil_image in pil_images:
        # Segment the image
        segmented_pil_image = segment_single_image(pil_image)
        
        # Convert the PIL image to a NumPy array
        frame = np.array(segmented_pil_image)
        
        # Write the frame to the video
        video.write(frame)
        
    video.release()
