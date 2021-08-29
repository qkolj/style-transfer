import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from os import listdir
from collections import OrderedDict
import argparse

from p2gan_models import Generator, Discriminator, Vgg16Partial

torch.backends.cudnn.deterministic = True
np.random.seed(1234)

parser = argparse.ArgumentParser(description='Transfer style from style image to content image using technique explained in paper P²-GAN: Efficient Style Transfer Using Single Style Image by Z. Zheng and J. Liu')
parser.add_argument('-m', '--mode', choices=['render', 'train'], required=True, type=str, help='Choose working mode')

parser.add_argument('-c', '--content-image', type=str, help='Path to content image (used in rendering mode)')
parser.add_argument('-mp', '--model-path', type=str, help='Path to trained model (used in rendering mode)')
parser.add_argument('--no-plot', action='store_true', help='Don\'t plot style transer result when finished (used in rendering mode) (optional)')
parser.add_argument('-o', '--save-output', metavar='FILENAME', type=str, help='Save style transer result to file (used in rendering mode) (optional)')

parser.add_argument('-ps', '--patch-size', choices=[9, 16], type=int, default=9, help='Discriminator patch size (used in training mode)')
parser.add_argument('-s', '--style-image', type=str, help='Path to style image (used in training mode)')
parser.add_argument('-sm', '--save-model-path', metavar='FILENAME', type=str, help='Path where to save model (used in training mode)')
parser.add_argument('-dp', '--dataset-path', type=str, help='Path to training dataset (used in training mode)')
parser.add_argument('--lambda', dest='lbd', type=float, default=1.e-3, help='Content loss weight parameter (used in training mode)')
parser.add_argument('--epochs', type=int, default=3, help='Training epochs (used in training mode)')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training (used in training mode)')
parser.add_argument('--print-every', metavar='N', type=int, default=100, help='Print current losses on every N iterations (used in training mode)')
parser.add_argument('--no-loss-plot', action='store_true', help='Don\'t plot losses when training is finished (used in training mode) (optional)')


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.mode == 'render':
    if (args.content_image is None) or (args.model_path is None):
        print('Both content image path and style model path should be provided in rendering mode.')
        exit()
    
    cimage = Image.open(args.content_image)
    content_image = torchvision.transforms.ToTensor()(cimage).unsqueeze(0).to(device)
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(args.model_path))
    out_image = Image.fromarray(torch.clamp((gen(content_image).cpu().detach().squeeze(0).permute(1,2,0) + 1) / 2 * 255, min=0, max=255).type(torch.uint8).numpy())
    
    if not args.no_plot:
        plt.imshow(out_image)
        plt.show()
        
    if args.save_output:
        out_image.save(args.save_output)
        
    exit()


if args.dataset_path is None:
    print('Dataset path should be provided in training mode.')
    exit()
elif args.style_image is None:
    print('Style image path should be provided in training mode.')
    exit()
elif args.save_model_path is None:
    print('Path where to save the trained model should be provided in training mode.')
    exit()
    
DATASET_PATH = args.dataset_path
paths = listdir(DATASET_PATH)
BATCH_SIZE = args.batch_size
DATASET_SIZE = len(paths)
BATCHES = DATASET_SIZE // BATCH_SIZE
EPOCHS = args.epochs

LAMBDA = args.lbd
C = 64

PATCH_SIZE = args.patch_size

if PATCH_SIZE == 9:
    W = 216
    H = 216
    T = 24
    disc_kernel_size = 3
else:
    W = 256
    H = 256
    T = 16
    disc_kernel_size = 4


def path_permutation(image, patch_size=PATCH_SIZE, n=T):
  pp = np.zeros((n*patch_size, n*patch_size, 3), dtype=np.float32)
  im_shape = image.shape
  for i in range(n):
    for j in range(n):
      y_random = int((im_shape[0] - patch_size) * np.random.uniform())
      x_random = int((im_shape[1] - patch_size) * np.random.uniform())
      pp[i*patch_size:i*patch_size+patch_size, j*patch_size:j*patch_size+patch_size, :] = image[y_random:y_random+patch_size, x_random:x_random+patch_size, :]
  return pp

def prepare_content_batch(image_paths):
  transform = torchvision.transforms.Compose([
                                            torchvision.transforms.Resize((H, W)),
                                            torchvision.transforms.ToTensor()
  ])
  first = True
  for path in image_paths:
    if first:
      image = Image.open(DATASET_PATH + path)
      batch = (transform(image).unsqueeze(0) - 0.5)*2
      first = False
    else:
      image = Image.open(DATASET_PATH + path)
      batch = torch.vstack((batch, (transform(image).unsqueeze(0)-0.5)*2))
  return batch

def prepare_style_batch(style_image, batch_size):
  batch = torchvision.transforms.ToTensor()(path_permutation(style_image)).unsqueeze(0)
  for i in range(1, batch_size):
    batch = torch.vstack((batch, torchvision.transforms.ToTensor()(path_permutation(style_image)).unsqueeze(0)))
  return batch

image = Image.open(args.style_image)
simage = (np.array(image) / 255.0 - 0.5) * 2

gen = Generator().to(device)
disc = Discriminator(T, disc_kernel_size).to(device)
vgg = Vgg16Partial().to(device)

g_optimizer = torch.optim.RMSprop(gen.parameters(), lr=5e-4, alpha=0.9)
d_optimizer = torch.optim.RMSprop(disc.parameters(), lr=5e-4, alpha=0.9)
        
g_losses = []
d_losses = []

vgg_transform = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

for epoch in range(EPOCHS):
  np.random.shuffle(paths)
  for batch_num in range(BATCHES):    
    style_images = prepare_style_batch(simage, BATCH_SIZE).to(device)
    content_images = prepare_content_batch(paths[batch_num*BATCH_SIZE:batch_num*BATCH_SIZE+BATCH_SIZE]).to(device)
    generator_images = gen(content_images).to(device)
     
    
    d_optimizer.zero_grad()
    
    d_real_d = torch.mean(disc(style_images))
    d_fake_d = torch.mean(disc(generator_images.detach()))
    d_loss = -(torch.log(d_real_d) + torch.log(1.0 - d_fake_d))
    d_loss.backward()
    d_losses.append((d_real_d.detach().cpu() + d_fake_d.detach().cpu())/2)
    d_optimizer.step()
    
    for _ in range(2):
        g_optimizer.zero_grad()
        
        generator_images = gen(content_images).to(device)
        
        d_fake = disc(generator_images)
        mean_d_fake = torch.mean(d_fake)
        g_disc_loss = -torch.log(torch.mean((d_fake) ** (1.0 - (d_fake - mean_d_fake))))

        vgg_c = vgg(vgg_transform(content_images))
        vgg_g = vgg(vgg_transform(generator_images))
        
        g_loss = (torch.norm(vgg_c - vgg_g)**2) * LAMBDA / (BATCH_SIZE * C * H * W) + g_disc_loss
        g_loss.backward()
        g_optimizer.step()
    g_losses.append(g_loss.detach().cpu())
    
    if batch_num % args.print_every == 0:
        print('Epoch: {} Batch: {:4} Disc fake loss: {:.6f} Disc real loss: {:.6f} Disc total loss: {:.6f} Gen loss: {:.6f}'.format(epoch + 1, batch_num, d_fake_d.detach().cpu(), d_real_d.detach().cpu(), d_loss.detach().cpu(), g_loss.detach().cpu()))

if args.save_model_path[-4:] != '.pth':
    args.save_model_path = args.save_model_path + '.pth'
torch.save(gen.state_dict(), args.save_model_path)
print('Trained model saved to {}'.format(args.save_model_path))

if not args.no_loss_plot:
    plt.figure(figsize=(30, 10))
    plt.plot(range(BATCHES * EPOCHS), g_losses, label='Generator loss')
    plt.plot(range(BATCHES * EPOCHS), d_losses, label='Discriminator total loss')
    plt.legend(loc='best')
    plt.show()
