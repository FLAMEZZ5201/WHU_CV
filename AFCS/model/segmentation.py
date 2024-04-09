import torch.nn as nn
class FCNs(nn.Module):
    def __init__(self, num_classes, backbone="vgg"):
        super(FCNs, self).__init__()
        self.num_classes = num_classes
        if backbone == "vgg":
            self.features = VGG()

        # deconv1 1/16
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU()

        # deconv1 1/8
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

        # deconv1 1/4
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        # deconv1 1/2
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        # deconv1 1/1
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.features(x)

        y = self.bn1(self.relu1(self.deconv1(features[4])) + features[3])

        y = self.bn2(self.relu2(self.deconv2(y)) + features[2])

        y = self.bn3(self.relu3(self.deconv3(y)) + features[1])

        y = self.bn4(self.relu4(self.deconv4(y)) + features[0])

        y = self.bn5(self.relu5(self.deconv5(y)))

        y = self.classifier(y)

        return y