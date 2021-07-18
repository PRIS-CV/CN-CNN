import torch.nn as nn
import torch
from torch.autograd import Variable
import math
import time
import os
import numpy as np
import cv2
import random
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn import init


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False) #每个block里的conv1 conv2 conv3
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param




class SimpleFPA(nn.Module):
    def __init__(self, in_planes, out_planes):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(SimpleFPA, self).__init__()

        self.channels_cond = in_planes
        # Master branch
        self.conv_master = BasicConv(in_planes, out_planes, kernel_size=1, stride=1)

        # Global pooling branch
        self.conv_gpb = BasicConv(in_planes, out_planes, kernel_size=1, stride=1)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)

        out = x_master + x_gpb #？？

        return out

class PyramidFeatures(nn.Module):
    """Feature pyramid module with top-down feature pathway"""
    def __init__(self, B2_size, B3_size, B4_size, B5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        self.F5_1 = SimpleFPA(B5_size, feature_size)
        #self.F5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.F4_1 = nn.Conv2d(B4_size, feature_size, kernel_size=1, stride=1, padding=0)
        #self.F4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.F3_1 = nn.Conv2d(B3_size, feature_size, kernel_size=1, stride=1, padding=0)
        #self.F3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.convlstm = ConvLSTM(input_dim=feature_size,
                 hidden_dim=[feature_size],
                 kernel_size=(3, 3),
                 num_layers=1,
                 batch_first=True, bias=True, return_all_layers=True)


    def forward(self, inputs):
        B3, B4, B5 = inputs

        F5_upsampled_x = F.interpolate(B5, scale_factor=4)
        F5_x = self.F5_1(F5_upsampled_x) 

        F4_upsampled_x = F.interpolate(B4, scale_factor=2)
        F4_x = self.F4_1(F4_upsampled_x)

        F3_x = self.F3_1(B3)

        F5_x.unsqueeze(0)
        F4_x.unsqueeze(0)
        F3_x.unsqueeze(0)

        Fx = torch.cat((F5_x.unsqueeze(0),F4_x.unsqueeze(0),F3_x.unsqueeze(0)),dim=0)
        layer_output,_ = self.convlstm(Fx)

        F5_x = layer_output[-1][0,:,:,:]
        F4_x = layer_output[-1][1,:,:,:]
        F3_x = layer_output[-1][2,:,:,:]

        return [F3_x, F4_x, F5_x]

class PyramidAttentions(nn.Module):
    """Attention pyramid module with bottom-up attention pathway"""
    def __init__(self, channel_size=256):
        super(PyramidAttentions, self).__init__()

        self.A3_1 = SpatialGate(channel_size)
        self.A3_2 = ChannelGate(channel_size)

        self.A4_1 = SpatialGate(channel_size)
        self.A4_2 = ChannelGate(channel_size)

        self.A5_1 = SpatialGate(channel_size)
        self.A5_2 = ChannelGate(channel_size)

    def forward(self, inputs):
        F3, F4, F5 = inputs

        A3_spatial = self.A3_1(F3)
        A3_channel = self.A3_2(F3)
        A3 = A3_spatial*F3 + A3_channel*F3

        A4_spatial = self.A4_1(F4)
        A4_channel = self.A4_2(F4)
        A4_channel = (A4_channel + A3_channel) / 2
        A4 = A4_spatial*F4 + A4_channel*F4

        A5_spatial = self.A5_1(F5)
        A5_channel = self.A5_2(F5)
        A5_channel = (A5_channel + A4_channel) / 2
        A5 = A5_spatial*F5 + A5_channel*F5

        return [A3, A4, A5, A3_spatial, A4_spatial, A5_spatial]

class SpatialGate(nn.Module):
    """generation spatial attention mask"""
    def __init__(self, out_channels):
        super(SpatialGate, self).__init__()
        self.conv = nn.ConvTranspose2d(out_channels,1,kernel_size=3,stride=1,padding=1)
    def forward(self, x):
        x = self.conv(x)
        return torch.sigmoid(x)

class ChannelGate(nn.Module):
    """generation channel attention mask"""
    def __init__(self, out_channels):
        super(ChannelGate, self).__init__()
        self.conv1 = nn.Conv2d(out_channels,out_channels//16,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(out_channels//16,out_channels,kernel_size=1,stride=1,padding=0)
    def forward(self, x):
        x = nn.AdaptiveAvgPool2d(output_size=1)(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = torch.sigmoid(self.conv2(x))
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def generate_anchors_single_pyramid(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    box_centers = np.stack(
        [box_centers_x, box_centers_y], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_widths, box_heights], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (x1, y1, x2, y2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return torch.from_numpy(boxes).cuda()

class ResNet(nn.Module):
    """implementation of AP-CNN on ResNet"""
    def __init__(self, num_classes, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1) #layers[]代表每层有几个block
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) #stride=2 进行downsample
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if self.num_classes == 200:
            hidden_num = 512
        else:
            hidden_num = 256

        if block == BasicBlock:
            fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels, self.layer2[layers[1] - 1].conv2.out_channels,
                         self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer1[layers[0] - 1].conv3.out_channels, self.layer2[layers[1] - 1].conv3.out_channels,
                         self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels] #layer1[2].conv3.out_channels 每个layer的最后一个block的channel输出

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3])#?? 256 512 1024 2048
        self.apn = PyramidAttentions(channel_size=256)

        self.cls5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.BatchNorm1d(256),
            nn.Linear(256, hidden_num),
            nn.BatchNorm1d(hidden_num),
            nn.ELU(inplace=True),
            nn.Linear(hidden_num, self.num_classes)
        )

        self.cls4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.BatchNorm1d(256),
            nn.Linear(256, hidden_num),
            nn.BatchNorm1d(hidden_num),
            nn.ELU(inplace=True),
            nn.Linear(hidden_num, self.num_classes)
        )

        self.cls3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),            
            nn.BatchNorm1d(256),
            nn.Linear(256, hidden_num),
            nn.BatchNorm1d(hidden_num),
            nn.ELU(inplace=True),
            nn.Linear(hidden_num, self.num_classes)
        )

        self.cls_concate = nn.Sequential(
            Flatten(),
            nn.BatchNorm1d(256*3),
            nn.Linear(256*3, hidden_num),
            nn.BatchNorm1d(hidden_num),
            nn.ELU(inplace=True),
            nn.Linear(hidden_num, self.num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample)) 
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def Concate(self, f3, f4, f5):
        f3 = nn.AdaptiveAvgPool2d(output_size=1)(f3)
        f5 = nn.AdaptiveAvgPool2d(output_size=1)(f5)
        f4 = nn.AdaptiveAvgPool2d(output_size=1)(f4)
        f_concate = torch.cat([f3, f4, f5], dim=1)
        return f_concate

    def forward(self, inputs, targets):
        # ResNet backbone with FC removed
        n, c, img_h, img_w = inputs.size()
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # stage I
        f3, f4, f5 = self.fpn([x2, x3, x4])
        f3_att, f4_att, f5_att, a3, a4, a5 = self.apn([f3, f4, f5])

        # feature concat
        f_concate = self.Concate(f3, f4, f5) 
        out_concate = self.cls_concate(f_concate)
        loss_concate = self.criterion(out_concate, targets)

        out3 = self.cls3(f3_att)
        out4 = self.cls4(f4_att)
        out5 = self.cls5(f5_att)

        loss3 = self.criterion(out3, targets)
        loss4 = self.criterion(out4, targets)
        loss5 = self.criterion(out5, targets)

        loss = loss3 + loss4 + loss5 + loss_concate
        out = (out3 + out4 + out5 + out_concate) / 4
        _, predicted = torch.max(out.data, 1)
        correct = predicted.eq(targets.data).cpu().sum().item()

        # attetion masks for visualizaton
        mask_cat = torch.cat([a3,
            F.interpolate(a4, a3.size()[2:]),
            F.interpolate(a5, a3.size()[2:])], 1)
        
        # return loss_ret, acc_ret, mask_cat, roi_list
        return loss, correct, mask_cat


def resnet18(num_classes, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(num_classes, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(num_classes, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(num_classes, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(num_classes, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    return model



