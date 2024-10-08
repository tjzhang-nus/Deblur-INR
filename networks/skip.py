import torch
import torch.nn as nn
from .common import *
from .normalizer import Mysoftmax
import torch.nn.functional as F
from networks.positional_encoding import get_input


# from dropblock import Dropblock2D, LinearScheduler
torch.cuda.set_device(0)
torch.set_num_threads(3)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

class DCGAN(nn.Module):
    def __init__(self, nz, ngf=64, output_size=(256, 256), nc=3, num_measurements=1000):
        super(DCGAN, self).__init__()
        self.nc = nc
        self.output_size = output_size

        self.conv1 = nn.ConvTranspose2d(nz, 8 * ngf, kernel_size=4, stride=1, padding=0,
                                        bias=False)  # kernel_size=4, stride=1, padding=0
        self.bn1 = nn.BatchNorm2d(8 * ngf)
        self.conv2 = nn.ConvTranspose2d(8 * ngf, 4 * ngf, 6, 2, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * ngf)
        self.conv3 = nn.ConvTranspose2d(4 * ngf, 4 * ngf, 6, 2, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(4 * ngf)
        self.conv4 = nn.ConvTranspose2d(4 * ngf, 4 * ngf, 6, 2, 2, bias=False)
        self.bn4 = nn.BatchNorm2d(4 * ngf)
        self.conv5 = nn.ConvTranspose2d(4 * ngf, 4 * ngf, 6, 2, 2, bias=False)
        self.bn5 = nn.BatchNorm2d(4 * ngf)
        self.conv6 = nn.ConvTranspose2d(4 * ngf, 4 * ngf, 6, 2, 2, bias=False)
        self.bn6 = nn.BatchNorm2d(4 * ngf)
        self.conv7 = nn.ConvTranspose2d(4 * ngf, 4 * ngf, 4, 2, 1, bias=False)  # output is image
        self.bn7 = nn.BatchNorm2d(4 * ngf)
        self.conv8 = nn.ConvTranspose2d(4 * ngf, nc, 4, 2, 1, bias=False)  # output is image

    def forward(self, z):
        input_size = z.size()
        x = F.relu(self.bn1(self.conv1(z)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.conv8(x)
        # print(x.shape)
        x = torch.nn.functional.sigmoid(x[:, :, 0:self.output_size[0], 0:self.output_size[1]])

        return x


def skip_ren(
        num_input_channels=2, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True,
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.
    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')
    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales

    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales

    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))

        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad,
                        downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        # if i > 1:
        #     deeper.add(NONLocalBlock2D(in_channels=num_channels_down[i]))
        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model


def mlp(encoder, input_depth, size_x, size_y, out_dim):
    class MLP(nn.Module):
        def __init__(self, encoder, input_depth, size_x, size_y,  out_dim,backbone='mlp'):
            super(MLP, self).__init__()
            self.res = size_x
            self.out = out_dim
            self.enc = True
            if backbone == 'mlp':
                n = 64
                self.backbone = nn.Sequential(
                    nn.Linear(32, 64),
                    nn.Sigmoid(),
                    nn.Linear(64, 128),
                    nn.Sigmoid(),
                    nn.Linear(128, 128),
                    nn.Sigmoid(),
                    nn.Linear(128, 64),
                    nn.Sigmoid(),
                    nn.Linear(64, 32),
                    nn.Sigmoid(),
                    nn.Linear(32, 16),
                    nn.Sigmoid(),
                    nn.Linear(16, self.out)
                )

        def forward(self, coords):
            coords = self.backbone(coords)
            img = coords.reshape(1, self.out, size_x, size_y)
            return img

    model= MLP(encoder=encoder, input_depth=input_depth, size_x=size_x, size_y=size_y, out_dim=out_dim, backbone='mlp')
    return model

def skip(
        num_input_channels=2, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True,
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True, dropout=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales

    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales

    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        if dropout:
            if i != 0:
                model_tmp.add(nn.Dropout(p=0.3))

        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad,
                        downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))

        deeper.add(act(act_fun))
        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))

    if need_sigmoid:
        model.add(nn.Sigmoid())
    return model


def skipkernel(
        num_input_channels=2, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True,
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True, dropout=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales

    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales

    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)
        if dropout:
            if i != 0:
                model_tmp.add(nn.Dropout(p=0.1))
        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad,
                        downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main
    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))  # kernel size
    if need_sigmoid:
        model.add(Mysoftmax())

    return model


class Mynet(nn.Module):
    def __init__(self, opt=None, num_input_channels=2, num_output_channels=3,
                 num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128],
                 num_channels_skip=[4, 4, 4, 4, 4],
                 filter_size_down=3, filter_size_up=3, filter_skip_size=1,
                 need_sigmoid=True, need_bias=True,
                 pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
                 need1x1_up=True):
        super(Mynet, self).__init__()
        self.prenet = skip(num_input_channels=num_input_channels, num_output_channels=num_output_channels,
                           num_channels_down=num_channels_down, num_channels_up=num_channels_up,
                           num_channels_skip=num_channels_skip,
                           filter_size_down=filter_size_down, filter_size_up=filter_size_up,
                           filter_skip_size=filter_skip_size,
                           need_sigmoid=need_sigmoid, need_bias=need_bias,
                           pad=pad, upsample_mode=upsample_mode, downsample_mode=downsample_mode, act_fun=act_fun,
                           need1x1_up=need1x1_up)
        # self.net_input = net_input_saved
        self.conv = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.opt = opt

    def forward(self, x):
        x = self.prenet(x)
        # below is the extended network
        x1 = nn.functional.upsample(x, size=(self.opt.img_size[0], self.opt.img_size[1]),
                                    mode='bilinear')  # scale_factor=2.0
        x2 = self.conv(x1)
        x = x1 + x2
        x = nn.functional.sigmoid(x)
        return x


class Mynetk(nn.Module):
    def __init__(self, opt=None, num_input_channels=2, num_output_channels=3,
                 num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128],
                 num_channels_skip=[4, 4, 4, 4, 4],
                 filter_size_down=3, filter_size_up=3, filter_skip_size=1,
                 need_sigmoid=True, need_bias=True,
                 pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
                 need1x1_up=True):
        super(Mynetk, self).__init__()
        self.prenet = skipkernel(num_input_channels=num_input_channels, num_output_channels=num_output_channels,
                                 num_channels_down=num_channels_down, num_channels_up=num_channels_up,
                                 num_channels_skip=num_channels_skip,
                                 filter_size_down=filter_size_down, filter_size_up=filter_size_up,
                                 filter_skip_size=filter_skip_size,
                                 need_sigmoid=need_sigmoid, need_bias=need_bias,
                                 pad=pad, upsample_mode=upsample_mode, downsample_mode=downsample_mode, act_fun=act_fun,
                                 need1x1_up=need1x1_up)
        # self.net_input = net_input_saved
        self.conv = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.opt = opt

    def forward(self, x):
        x = self.prenet(x)
        # below is the extended network
        x1 = nn.functional.upsample(x, size=(self.opt.kernel_size[0], self.opt.kernel_size[1]),
                                    mode='bilinear')  # scale_factor=2.0
        x2 = self.conv(x1)
        x = x1 + x2
        x = x.view(1, 1, 1, -1)
        x = torch.squeeze(x)
        output = nn.functional.softmax(x, dim=0)
        output = output.view(-1, 1, self.opt.kernel_size[0], self.opt.kernel_size[1])
        # x = nn.functional.sigmoid(x)
        return output