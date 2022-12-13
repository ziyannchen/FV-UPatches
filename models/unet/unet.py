import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
#         print(x.shape, skip.shape)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()
        l1=int(64/4); l2=int(128/4); l3=int(256/4); l4=int(512/4)
        layer1 = l1; layer2 = l2; layer3 = l3; layer4 = l4
        layer5 = layer3; layer6 = layer2
        out = layer1
        """ Encoder """
        self.e1 = encoder_block(1, layer1)
        self.e2 = encoder_block(layer1, layer2)
        self.e3 = encoder_block(layer2, layer3)
#         self.e4 = encoder_block(layer3, layer4)

        """ Bottleneck """
        self.b = conv_block(layer3, layer4)

        """ Decoder """
#         self.d1 = decoder_block(layer4, layer5)
        self.d2 = decoder_block(layer4, layer5)
        self.d3 = decoder_block(layer5, layer6)
        self.d4 = decoder_block(layer6, out)

        """ Classifier """
        self.outputs = nn.Conv2d(out, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
#         s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p3)

        """ Decoder """
#         d1 = self.d1(b, s4)
        d2 = self.d2(b, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)

        return outputs


from utils import seeding

class UNet:
    def __init__(self, weight_file=None, train=False,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        seeding(42)

        self.H = 128
        self.W = 256
        self.size = (self.W, self.H)

        self.model = build_unet().to(device)
        self.train = train
        if train is False:
            self.model.load_state_dict(torch.load(weight_file, map_location=device))
            self.eval()

    def eval(self):
        self.train = False
        self.model.eval()
        self.model.requires_grad_(False)

    def __call__(self, im):
        return self.model(im)
