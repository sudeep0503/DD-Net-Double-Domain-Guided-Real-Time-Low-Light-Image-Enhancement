import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import datetime

def extract_log_features(img, sigma=1.0):
    # Convert image to grayscale float format
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float64)

    # Apply Gaussian smoothing
    gaussian = cv2.GaussianBlur(img, (5, 5), sigma)

    # Calculate the Laplacian of Gaussian (LoG) operator
    laplacian = cv2.Laplacian(gaussian, cv2.CV_64F)

    # Define the LoG-based gradient feature extraction kernel
    log_kernel = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, -16, 2, 1],
        [1, 2, 1, 2, 1],
        [0, 0, 1, 0, 0]
    ])

    # Extract LoG-based gradient features
    log_features = cv2.filter2D(laplacian, -1, log_kernel)

    return log_features

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()

        #print(f"Kernel size passed: {kernel_size}")
        #assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        #padding = 3 if kernel_size == 7 else 1

        #self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class StandardConvolutionalModule(nn.Module):
  def __init__(self, channel):
    super(StandardConvolutionalModule, self).__init__()

    self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
    self.act = nn.PReLU()
    self.norm = nn.GroupNorm(num_channels=channel, num_groups=1)

  def forward(self, x):
    x = self.conv1(x)
    x = self.norm(x)
    x = self.act(x)
    return x

class LeftED(nn.Module):
  def __init__(self, in_channel, channel):
    super(LeftED, self).__init__()

    self.conv_in = nn.Conv2d(in_channel,channel,kernel_size=3,stride=1,padding=1,bias=False)

    self.e1 = ScCAModule(channel)

    self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    self.conv_e1te2 = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)
    self.e2 = ScCAModule(channel*2)

    self.conv_e2te3 = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)
    self.e3 = ScCAModule(channel*4)

    self.conv_e1_a = nn.Conv2d(channel,int(0.5*channel),kernel_size=1,stride=1,padding=0,bias=False)
    self.conv_e2_a = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)
    self.conv_e3_a = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)

    self.fd1 = ScCAModule(channel*2)
    self.fd2 = ScCAModule(channel*1)
    self.conv_fd1td2 = nn.Conv2d(2*channel,1*channel,kernel_size=1,stride=1,padding=0,bias=False)

    self.fd3 = ScCAModule(int(channel*0.5))
    self.conv_fd2td3 = nn.Conv2d(channel,int(0.5*channel),kernel_size=1,stride=1,padding=0,bias=False)

    self.ed1 = ScCAModule(channel*2)
    self.ed2 = ScCAModule(channel*1)
    self.ed3 = ScCAModule(int(channel*0.5))

    self.conv_ed1td2 = nn.Conv2d(2*channel,1*channel,kernel_size=1,stride=1,padding=0,bias=False)
    self.conv_ed2td3 = nn.Conv2d(channel,int(0.5*channel),kernel_size=1,stride=1,padding=0,bias=False)

    self.conv_eout = nn.Conv2d(int(0.5*channel),3,kernel_size=1,stride=1,padding=0,bias=False)
    self.conv_fout = nn.Conv2d(int(0.5*channel),1,kernel_size=1,stride=1,padding=0,bias=False)

  def _upsample(self,x,y):
    _,_,H,W = y.size()
    return F.upsample(x,size=(H,W),mode='bilinear')

  def forward(self, x, xgl):

    x_in = self.conv_in(torch.cat((x, xgl), dim=1))

    e1 = self.e1(x_in)
    e2 = self.e2(self.conv_e1te2(self.maxpool(e1)))
    e3 = self.e3(self.conv_e2te3(self.maxpool(e2)))

    e1_a = self.conv_e1_a(e1)
    e2_a = self.conv_e2_a(e2)
    e3_a = self.conv_e3_a(e3)

    fd1 = self.fd1(e3_a)
    #print("Hello")
    fd2 = self.fd2(self.conv_fd1td2(self._upsample(fd1,e2)) + e2_a)
    fd3 = self.fd3(self.conv_fd2td3(self._upsample(fd2,e1)) + e1_a)

    ed1 = self.ed1(e3_a + fd1)
    ed2 = self.ed2(self.conv_ed1td2(self._upsample(ed1,e2)) + fd2 + e2_a)
    ed3 = self.ed3(self.conv_ed2td3(self._upsample(ed2,e1)) + fd3 + e1_a)

    x_fout = self.conv_fout(fd3)
    x_eout = self.conv_eout(ed3)

    return x_fout, x_eout, ed1, ed2, ed3, fd1, fd2, fd3, e1, e2, e3

class RightED(nn.Module):
	def __init__(self,inchannel,channel):
		super(RightED,self).__init__()

		self.ee1 = ScCAModule(int(0.5*channel))
		self.ee2 = ScCAModule(channel)
		self.ee3 = ScCAModule(channel*2)

		self.fe1 = ScCAModule(int(0.5*channel))
		self.fe2 = ScCAModule(channel)
		self.fe3 = ScCAModule(channel*2)

		self.d1 = ScCAModule(channel*4)
		self.d2 = ScCAModule(channel*2)
		self.d3 = ScCAModule(channel)

		self.d4 = ScCAModule(channel)

		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)


		self.conv_out = nn.Conv2d(channel,3,kernel_size=3,stride=1,padding=1,bias=False)

		self.conv_fe0te1 = nn.Conv2d(int(0.5*channel),channel,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv_fe1te2 = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)

		self.conv_ee0te1 = nn.Conv2d(int(0.5*channel),channel,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv_ee1te2 = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)

		self.conv_e0te1 = nn.Conv2d(int(1*channel),channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_e1te2 = nn.Conv2d(int(2*channel),2*channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_e2te3 = nn.Conv2d(int(4*channel),4*channel,kernel_size=3,stride=1,padding=1,bias=False)

		self.conv_d1td2 = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)
		self.conv_d2td3 = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)


		self.act1 = nn.PReLU(channel)
		self.norm1 = nn.GroupNorm(num_channels=channel,num_groups=1)

		self.act2 = nn.PReLU(channel*2)
		self.norm2 = nn.GroupNorm(num_channels=channel*2,num_groups=1)

		self.act3 = nn.PReLU(channel*4)
		self.norm3 = nn.GroupNorm(num_channels=channel*4,num_groups=1)

	def _upsample(self,x,y):
		_,_,H,W = y.size()
		return F.upsample(x,size=(H,W),mode='bilinear')

	def forward(self,x, ed1, ed2, ed3, fd1, fd2, fd3, e1, e2, e3):


		fe1 = self.fe1(fd3)
		fe2 = self.fe2(self.conv_fe0te1(self.maxpool(fe1)) + fd2)
		fe3 = self.fe3(self.conv_fe1te2(self.maxpool(fe2)) + fd1)

		ee1 = self.ee1(ed3 + fe1)
		ee2 = self.ee2(self.conv_ee0te1(self.maxpool(ee1)) + fe2 + ed2)
		ee3 = self.ee3(self.conv_ee1te2(self.maxpool(ee2)) + fe3 + ed1)

		fde1 = self.act1(self.norm1(self.conv_e0te1(torch.cat((ee1 , fe1),1))))
		fde2 = self.act2(self.norm2(self.conv_e1te2(torch.cat((ee2 , fe2),1))))
		fde3 = self.act3(self.norm3(self.conv_e2te3(torch.cat((ee3 , fe3),1))))

		d1 = self.d1(fde3 + e3)
		d2 = self.d2(self.conv_d1td2(self._upsample(d1,e2)) + fde2 + e2)
		d3 = self.d3(self.conv_d2td3(self._upsample(d2,e1)) + fde1 + e1)


		x_out = self.conv_out(self.d4(d3))

		return x_out + x

def ssim(img1, img2):
  C1 = (0.01 * 255)**2
  C2 = (0.03 * 255)**2
  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  kernel = cv2.getGaussianKernel(11, 1.5)
  window = np.outer(kernel, kernel.transpose())
  mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # valid
  mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
  mu1_sq = mu1**2
  mu2_sq = mu2**2
  mu1_mu2 = mu1 * mu2
  sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
  sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
  sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
  ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                  (sigma1_sq + sigma2_sq + C2))
  return ssim_map.mean()

def calculate_ssim(img1, img2):
  if not img1.shape == img2.shape:
    raise ValueError('Input images must have the same dimensions.')
  if img1.ndim == 2:
    return ssim(img1, img2)
  elif img1.ndim == 3:
    if img1.shape[0] == 3:
      ssims = []
      for i in range(3):
        ssims.append(ssim(img1[i,:,:], img2[i,:,:]))
        return np.array(ssims).mean()
    elif img1.shape[0] == 1:
      return ssim(np.squeeze(img1), np.squeeze(img2))
  else:
    raise ValueError('Wrong input image dimensions.')

def laplacian_gradient_consistency_loss(log_features, ground_truth):
  return np.mean((log_features - ground_truth) ** 2)

def coarse_enhancement_loss(coarse_features, ground_truth):
  return np.mean((coarse_features - ground_truth) ** 2)

def final_enhancement_loss(final_features, ground_truth):
  x = [calculate_ssim(final_features[i], ground_truth[i]) for i in range(len(final_features))]
  return 1 - np.mean([x])
  #return 1 - x

def joint_loss_function(laplacian_loss, coarse_loss, final_loss):
  return 0.2 * laplacian_loss + 0.2 * coarse_loss + 0.6 * final_loss


class Main(nn.Module):
  def __init__(self,in_channel,channel):
    super(Main,self).__init__()

    self.left = LeftED(4,32)
    self.right = RightED(3,32)

  def forward(self,x,xgl):
    x_fout, x_eout, ed1, ed2, ed3, fd1, fd2, fd3, e1, e2, e3 = self.left(x,xgl)
    x_out = self.right(x, ed1, ed2, ed3, fd1, fd2, fd3, e1, e2, e3)

    return x_fout, x_eout, x_out

def hwc_to_chw(img):
  return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])

def GFLap(data):
    x = cv2.GaussianBlur(data, (3,3),0)
    x = cv2.Laplacian(np.clip(x*255,0,255).astype('uint8'),cv2.CV_8U,ksize =3)
    Lap = cv2.convertScaleAbs(x)
    return Lap/255.0


##################################################

#Training Loop


low_light_dir = '/low/image/directory'
high_light_dir = '/high/image/directory'

# Log file
log_file = open('/log/file/path', 'a')

# Initialize model, loss functions, and optimizer
model = Main(4, 32).cuda()           # in_channel=4, channel=32
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
batch_size = 16

log_write = f"Original:\n"
log_file.write(log_write)

for epoch in range(1, num_epochs + 1):

    # Collecting All the Low Light Images
    low_light_images = [cv2.imread(os.path.join(low_light_dir, filename)) for filename in os.listdir(low_light_dir) if cv2.imread(os.path.join(low_light_dir, filename)) is not None]
    # Collecting All the High Light Images
    high_light_images = [cv2.imread(os.path.join(high_light_dir, filename)) for filename in os.listdir(high_light_dir) if cv2.imread(os.path.join(high_light_dir, filename)) is not None]

    # Number of Batches
    num_batches = len(low_light_images) // batch_size

    print(f'Epoch: {epoch}')
    for i in tqdm(range(1, num_batches + 1)):

      batch_low_light_images = low_light_images[i*batch_size:(i+1)*batch_size]
      batch_high_light_images = high_light_images[i*batch_size:(i+1)*batch_size]

      total_loss = 0.0

      #print("Epoch: ", epoch, "Batch: ", i)

      for idx, low in enumerate(batch_low_light_images):
          #print(idx)
          #print(low.shape)
          low_img = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)
          high_img = cv2.cvtColor(batch_high_light_images[idx], cv2.COLOR_BGR2RGB)
          low_xgl = cv2.cvtColor(low_img, cv2.COLOR_RGB2GRAY)

          x = low_img / 255.0
          x_xgl = low_xgl / 255.0

          x = hwc_to_chw(np.array(x).astype('float32'))
          input_x = torch.from_numpy(x.copy()).type(torch.FloatTensor).unsqueeze(0).cuda()
          input_x_xgl = torch.from_numpy(GFLap(x_xgl)).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()

          x_fout, x_eout, x_out = model(input_x, input_x_xgl)

          # Convert high_img to tensor
          high_img_tensor = torch.from_numpy(high_img).permute(2, 0, 1).float() / 255.0  # Ensure proper dimensions and normalization
          high_img_tensor = high_img_tensor.unsqueeze(0)  # Add batch dimension

          laplacian_loss = laplacian_gradient_consistency_loss(x_fout.cpu().detach().numpy(), input_x_xgl.cpu().numpy())
          coarse_loss = coarse_enhancement_loss(x_eout.cpu().detach().numpy(), high_img_tensor.cpu().numpy())
          final_loss = final_enhancement_loss(x_out.cpu().detach().numpy(), high_img_tensor.cpu().numpy())

          loss = joint_loss_function(laplacian_loss, coarse_loss, final_loss)
          total_loss += loss

          # Convert loss to a tensor for backward pass
          loss_tensor = torch.tensor(loss, requires_grad=True)
          #print(f'Total Loss: {loss}')

          # Backward pass and optimization
          optimizer.zero_grad()
          loss_tensor.backward()
          optimizer.step()
      # Calculate average loss per batch
      avg_loss = total_loss / len(batch_low_light_images)

      # Log the training progress to file
      current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      log_entry = f"{current_time}, Epoch [{epoch}/{num_epochs}], Average Loss: {avg_loss}\n"
      log_file.write(log_entry)
      #print(log_entry)  # Print to console for immediate feedback

    print(f'Epoch [{epoch}/{num_epochs}], Loss: {total_loss/len(batch_low_light_images)}')

# Closing Log File
log_write = f"\n\n"
log_file.write(log_write)
log_file.close()
