import torch
import torch.nn as nn
from facemesh import FaceMeshBlock

class CoordToParam(nn.Module):
    def __init__(self):
        super(CoordToParam, self).__init__()

        self.numOfParams = 8
        self.lin = nn.Linear(1404, self.numOfParams)

    def load_weights(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def forward(self, input):
        input = input.reshape(-1, 1404)
        output = self.lin(input)

        return output



class FeatToParam(nn.Module):
    def __init__(self):
        super(FeatToParam, self).__init__()

        self.numOfParams = 8

        self.coord_head2 = nn.Sequential(
            FaceMeshBlockWithBatchnorm(32, 32),
            nn.Conv2d(32, self.numOfParams, 3)
        )
        self.norm = nn.BatchNorm2d(self.numOfParams)

    def forward(self, input):
        output = self.coord_head2(input)
        output = self.norm(output)
        output = output.reshape(-1, self.numOfParams) 

        return output

    def load_weights(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

class FaceMeshBlockWithBatchnorm(nn.Module):
    """This is the main building block for architecture
    which is just residual block with one dw-conv and max-pool/channel pad
    in the second branch if input channels doesn't match output channels"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super(FaceMeshBlockWithBatchnorm, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch 
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding, 
                      groups=in_channels, bias=True),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
        )


        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
        
        return self.act(self.convs(h) + x)