import torch
import torch.nn as nn

class RecurrentBlock(nn.Module):
    def __init__(self, ch_out, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            RecurrentBlock(ch_out, t=t),
            RecurrentBlock(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class R2UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, t=2, features=[64, 128, 256, 512]):
        super(R2UNet, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder path
        self.encoder = nn.ModuleList()
        self.encoder.append(RRCNN_block(in_channels, features[0], t))
        
        for i in range(1, len(features)):
            self.encoder.append(RRCNN_block(features[i-1], features[i], t))
            
        # Decoder path
        self.decoder = nn.ModuleList()
        self.decoder.append(
            nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        )
        self.decoder.append(RRCNN_block(features[-1], features[-2], t))
        
        for i in range(len(features)-2, 0, -1):
            self.decoder.append(
                nn.ConvTranspose2d(features[i], features[i-1], kernel_size=2, stride=2)
            )
            self.decoder.append(RRCNN_block(features[i], features[i-1], t))
            
        # Final output layer
        self.Conv_1x1 = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # Encoder
        skip_connections = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            skip_connections.append(x)
            if i < len(self.encoder) - 1:  # No pooling after the last encoder block
                x = self.Maxpool(x)
                
        # Decoder
        skip_connections = skip_connections[::-1]  # Reverse for easier indexing
        
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)  # Upsample
            # Handle potential size mismatch due to odd dimensions
            if x.shape != skip_connections[i//2 + 1].shape:
                diff_y = skip_connections[i//2 + 1].size()[2] - x.size()[2]
                diff_x = skip_connections[i//2 + 1].size()[3] - x.size()[3]
                x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                                         diff_y // 2, diff_y - diff_y // 2])
            
            x = torch.cat([skip_connections[i//2 + 1], x], dim=1)  # Skip connection
            x = self.decoder[i+1](x)  # RRCNN block
            
        return self.Conv_1x1(x)