import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Transfer style from style image to content image using technique explained in paper A Neural Algorithm of Artistic Style by Gatys et al.')
parser.add_argument('-c', '--content-image', required=True, type=str, help='Path to content image')
parser.add_argument('-s', '--style-image', required=True, type=str, help='Path to style image')
parser.add_argument('-i', '--input-type', choices=['content', 'noise'], default='content', help='Use either content image or white noise image as input (default: content)')
parser.add_argument('--use-rgb', action='store_true', help='Use RBG images as input for VGG19 model instead of BGR images')
parser.add_argument('-d', '--dimensions', type=int, nargs=2, default=(512, 683), metavar=('H', 'W'), help='Dimensions of the output image, HxW (default: 512 683)')
parser.add_argument('--use-torchvision-model', action='store_true', help='Use pretrained VGG19 model from torchvision instad of paper authors\' pretrained VGG19 model')
parser.add_argument('--num-iters', type=int, default=500, help='Number of optimization iterations (default: 500)')
parser.add_argument('--print-every', metavar='N', type=int, default=50, help='Print current loss on every N iterations (default: 50)')
parser.add_argument('--alpha', type=float, default=1.e0, help='Style loss weight parameter (default: 1.0)')
parser.add_argument('--beta', type=float, default=1.e3, help='Content loss weight parameter (default: 1.0e3)')
parser.add_argument('--no-plot', action='store_true', help='Don\'t plot style transer result when finished')
parser.add_argument('--show-report', action='store_true', help='Plot final style transfer report')
parser.add_argument('-o', '--save-output', metavar='FILENAME', type=str, help='Save final output image to file')


args = parser.parse_args()

use_noise_image = (args.input_type == 'noise')
use_bgr = not args.use_rgb
use_authors_model = not args.use_torchvision_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dims = args.dimensions

num_iters = args.num_iters
print_every = args.print_every

alpha = args.alpha
beta = args.beta

def load_image(image_path, dimensions=(512, 683), bgr=False):
  image = Image.open(image_path)
  vgg_prepare = torchvision.transforms.Compose([torchvision.transforms.Resize(dimensions),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                                torchvision.transforms.Lambda(lambda x: x.mul_(255)),
                                                torchvision.transforms.Lambda(lambda x: x[[2, 1, 0], :, :] if bgr else x),
                                                ])
  image = vgg_prepare(image)
  return image.unsqueeze(0)

def postprocess_image(input_image, bgr=False, noise_image=False):
  inverse_vgg_prepare = torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda x: x[:, [2, 1, 0], :, :] if bgr else x),
                                                        torchvision.transforms.Lambda(lambda x: x.div_(255)),
                                                        torchvision.transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1./0.229, 1./0.224, 1./0.225]),
                                                        torchvision.transforms.Lambda(lambda x: x.mul_(255)),
                                                        ])
  return torch.clamp(inverse_vgg_prepare(input_image) if not noise_image else input_image, min=0, max=255).squeeze(0).permute(1, 2, 0).type(torch.uint8)

class NN(torch.nn.Module):
  def __init__(self, use_authors_model=False):
    super(NN, self).__init__()
    self.vgg = torchvision.models.vgg19(pretrained=True).features

    # Replace MaxPool with AvgPool as the paper suggests
    for name, child in self.vgg.named_children():
      if isinstance(child, torch.nn.MaxPool2d):
        self.vgg[int(name)] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
    
    self.vgg = self.vgg.eval()

    for param in self.vgg.parameters():
      param.requires_grad = False

    if use_authors_model:
      from torch.utils.model_zoo import load_url
      from collections import OrderedDict

      model = load_url('http://bethgelab.org/media/uploads/pytorch_models/vgg_conv.pth')
      state = OrderedDict()

      for i in range(len(self.vgg.state_dict().keys())):
        state.update({list(self.vgg.state_dict().keys())[i] : list(model.values())[i]})

      self.vgg.load_state_dict(state)


  def forward(self, x):
    return self.vgg(x)

  def get_content_features(self, x):
    return self.vgg[:23](x)

  def get_style_features(self, x):
    return [self.vgg[:2](x)] + [self.vgg[:7](x)] + [self.vgg[:12](x)] + [self.vgg[:21](x)] + [self.vgg[:30](x)]

def gram_matrix(x):
  (_, filters, height, width) = x.size()
  mat = x.view(filters, height * width).to(device)
  return mat.mm(mat.t()).div_(filters * height * width)

def total_loss(NN, input_image, content_outs, style_outs, alpha=1e0, beta = 1e3):
  input_content = NN.get_content_features(input_image)
  input_style = NN.get_style_features(input_image)

  content_loss = torch.nn.MSELoss()(input_content, content_outs)

  style_loss = 0.0
  for i in range(len(input_style)):
    style_loss += torch.nn.MSELoss()(gram_matrix(input_style[i]), style_outs[i])
  style_loss /= len(input_style)

  return alpha * content_loss + beta * style_loss

content_image = load_image(args.content_image, dimensions=dims, bgr=use_bgr).to(device)
style_image = load_image(args.style_image, dimensions=dims, bgr=use_bgr).to(device)

if use_noise_image:
  input_image = torch.autograd.Variable(torch.randn(content_image.size()).type_as(content_image.data).mul_(255), requires_grad=True)
else:
  input_image = torch.autograd.Variable(content_image.clone(), requires_grad=True).to(device)
input_image_start = postprocess_image(input_image.clone().cpu().detach(), bgr=use_bgr, noise_image=use_noise_image)

neural_net = NN(use_authors_model) 
neural_net.to(device)

content_outs = neural_net.get_content_features(content_image).detach()
style_outs = [gram_matrix(style_out).detach() for style_out in neural_net.get_style_features(style_image)]

optimizer = torch.optim.LBFGS([input_image], max_iter=10)
num_iter = 1

def closure():
  global num_iter
  optimizer.zero_grad()
  loss = total_loss(neural_net, input_image, content_outs, style_outs, alpha=alpha, beta=beta)
  if num_iter % print_every == 0:
    print("Iteration: {0: >4} Loss: {1}".format(num_iter, loss.item()))
  loss.backward()
  num_iter += 1
  return loss

while num_iter <= num_iters:
  optimizer.step(closure)

output_image = Image.fromarray(postprocess_image(input_image.cpu().detach(), bgr=use_bgr).numpy())

if not args.no_plot:
    plt.figure(figsize=(10, 10))
    plt.imshow(output_image)
    plt.title("Style transfer")
    plt.axis("off")
    plt.show()

if args.show_report:
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)

    ax1.imshow(postprocess_image(content_image.cpu().detach(), bgr=use_bgr))
    ax1.set_title("Content image")
    ax1.axis("off")
    ax2.imshow(postprocess_image(style_image.cpu().detach(), bgr=use_bgr))
    ax2.set_title("Style image")
    ax2.axis("off")
    ax3.imshow(input_image_start)
    ax3.set_title("Input image")
    ax3.axis("off")
    ax4.imshow(output_image)
    ax4.set_title("Output image")
    ax4.axis("off")
    fig.suptitle("alpha: {} beta: {} iterations: {} resolution: {}x{}".format(alpha, beta, num_iters, dims[1], dims[0]), y=0.07)
    plt.show()
    
if args.save_output:
    output_image.save(args.save_output)
