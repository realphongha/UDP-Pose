import torch
import torch.nn as nn

from .DUC import DUC


class PixelShuffleDecoder(nn.Module):
    def __init__(self, inplanes, start_channels=256, 
                 architecture=(512, 256, 128)):
        super(PixelShuffleDecoder, self).__init__()
        
        for num in architecture:
            assert num % 4 == 0, "Numbers in architecture should be divisible by 4"
        
        self.conv_compress = nn.Conv2d(inplanes, start_channels, 1, 1, 0, 
                                       bias=False)
        
        layers = list()
        for i in range(len(architecture)):
            if i == 0:
                layers.append(DUC(start_channels, architecture[0], upscale_factor=2))
            else:
                layers.append(DUC(architecture[i-1]//4, architecture[i], upscale_factor=2))
        
        self.duc = nn.Sequential(*layers)
        self.out_channels = architecture[-1]//4

    def forward(self, x):        
        x = self.conv_compress(x)
        x = self.duc(x)
        return x
    
if __name__ == "__main__":
    model = PixelShuffleDecoder(1280)
    input_test = torch.empty(16, 1280, 7, 7)
    output = model(input_test)
    print(output.shape)
    