
from torchinfo import summary
from networks.skip import skip
import numpy as np
from networks.siren import *

def fcn(num_input_channels=200, num_output_channels=21, num_hidden=1000):
    model = nn.Sequential()
    model.add(nn.Linear(num_input_channels, num_hidden, bias=True))
    model.add(nn.ReLU6())

    model.add(nn.Linear(num_hidden, num_output_channels))

    model.add(nn.Softmax())
    return model




class AdaptiveCentralLayer(nn.Module):
    def __init__(self):
        super(AdaptiveCentralLayer, self).__init__()

    def forward(self, kernel):

        B, C, H, W = kernel.size()


        y_grid, x_grid = torch.meshgrid(torch.arange(H, device=kernel.device, dtype=torch.float32),
                                        torch.arange(W, device=kernel.device, dtype=torch.float32), indexing='ij')

        kernel_sum = kernel.sum(dim=(1, 2, 3), keepdim=True)
        if torch.any(kernel_sum == 0):
            raise ValueError("Kernel sum should not be zero to avoid division by zero error.")

        centroid_y = (kernel * y_grid).sum(dim=(2, 3)) / kernel_sum
        centroid_x = (kernel * x_grid).sum(dim=(2, 3)) / kernel_sum


        shift_y = (H / 2 - centroid_y).view(B, 1, 1, 1)
        shift_x = (W / 2 - centroid_x).view(B, 1, 1, 1)


        shifted_kernel = torch.roll(kernel, shifts=(shift_y.int(), shift_x.int()), dims=(2, 3))

        return shifted_kernel
class dinp_cen_s(nn.Module):
    def __init__(self, input_depth, channel):
        super(dinp_cen_s, self).__init__()
        self.skip = skip(input_depth, channel,
                         num_channels_down=[64, 64, 64, 64, 64],
                         num_channels_up=[64, 64, 64, 64, 64],
                         num_channels_skip=[16, 16, 16, 16, 16],
                         upsample_mode='bilinear',
                         need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')

        self.fcn1 = Siren(in_features=2, out_features=1, hidden_features=64,
                          hidden_layers=3, outermost_linear=True)

        self.central_layer = AdaptiveCentralLayer()
    def forward(self, skip_input, kernel_input,ker_size):
        skip_out = self.skip(skip_input)
        kernel, coords2 = self.fcn1(kernel_input)
        kernel = kernel.view(-1, 1, ker_size, ker_size)
        centered_kernel = self.central_layer(kernel)


        return skip_out, centered_kernel
class dinp_cen(nn.Module):
    def __init__(self, input_depth, channel):
        super(dinp_cen, self).__init__()
        self.skip = skip(input_depth, channel,
                         num_channels_down=[128, 128, 128, 128, 128],
                         num_channels_up=[128, 128, 128, 128, 128],
                         num_channels_skip=[16, 16, 16, 16, 16],
                         upsample_mode='bilinear',
                         need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')

        self.fcn1 = Siren(in_features=2, out_features=1, hidden_features=64,
                          hidden_layers=3, outermost_linear=True)

        self.central_layer = AdaptiveCentralLayer()
    def forward(self, skip_input, kernel_input,ker_size):
        skip_out = self.skip(skip_input)
        kernel, coords2 = self.fcn1(kernel_input)
        kernel = kernel.view(-1, 1, ker_size, ker_size)
        centered_kernel = self.central_layer(kernel)


        return skip_out, centered_kernel
