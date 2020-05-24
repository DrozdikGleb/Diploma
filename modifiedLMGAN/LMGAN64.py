import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, features_size: int, instances_size: int, classes_size: int, meta_length: int, z_length: int):
        super(Generator, self).__init__()

        self.data_size = instances_size
        self.meta_length = meta_length
        self.z_length = z_length

        # in (?, z_length, 1, 1)
        # out (?, data_size * 2, 4, 4)
        self.fc_z = nn.ConvTranspose2d(in_channels=self.z_length,
                                       out_channels=self.data_size * 2, kernel_size=4, stride=1, padding=0)
        # in (?, meta_length, 1, 1)
        # out (?, data_size * 2, 4, 4)
        self.fc_meta = nn.ConvTranspose2d(in_channels=self.meta_length,
                                          out_channels=self.data_size * 2, kernel_size=4, stride=1, padding=0)
        # in (?, data_size * 4, 4, 4)
        # out (?, data_size * 2, 8, 8)
        self.deconv1 = nn.ConvTranspose2d(in_channels=self.data_size * 4,
                                          out_channels=self.data_size * 2, kernel_size=4, stride=2, padding=1)
        # out (?, data_size, 16, 16)
        self.deconv2 = nn.ConvTranspose2d(in_channels=self.data_size * 2,
                                          out_channels=self.data_size, kernel_size=4, stride=2, padding=1)
        # out (?, data_size / 2, 32, 16)
        self.deconv3 = nn.ConvTranspose2d(in_channels=self.data_size,
                                          out_channels=classes_size, kernel_size=(4, 1), stride=(2, 1),
                                          padding=(1, 0))
        # # out (?, data_size / 2, 64, 16)
        # self.deconv4 = nn.ConvTranspose2d(in_channels=self.data_size,
        #                                   out_channels=classes_size, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))

    def forward(self, z, meta):
        fc_z = F.leaky_relu(self.fc_z(z), 0.2)
        fc_meta = F.leaky_relu(self.fc_meta(meta), 0.2)

        fc = torch.cat((fc_z, fc_meta), 1)
        deconv1 = F.leaky_relu(self.deconv1(fc), 0.2)
        deconv2 = F.leaky_relu(self.deconv2(deconv1), 0.2)
        #deconv3 = F.leaky_relu(self.deconv3(deconv2), 0.2)
        deconv3 = F.sigmoid(self.deconv3(deconv2))
        return deconv3


class Discriminator(nn.Module):

    def __init__(self, features_size: int, instances_size: int, classes_size: int, meta_length: int,
                 lambda_length: int):
        super(Discriminator, self).__init__()

        self.data_size = instances_size
        self.meta_length = meta_length
        self.lambda_length = lambda_length

        # in (?, classes_size, instances_size, features_size)
        # out (?, data_size / 2, 16, 16)
        self.conv_1 = nn.Conv2d(in_channels=classes_size,
                                out_channels=self.data_size, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))

        # in (?, data_size, 16, 16)
        # out (?, data_size * 2, 8, 8)
        self.conv_2 = nn.Conv2d(in_channels=self.data_size,
                                out_channels=self.data_size * 2, kernel_size=4, stride=2, padding=1)
        # out (?, data_size * 4, 4, 4)
        self.conv_3 = nn.Conv2d(in_channels=self.data_size * 2,
                                out_channels=self.data_size * 4, kernel_size=4, stride=2, padding=1)
        # out (?, data_size * 8, 1, 1)
        self.conv_4 = nn.Conv2d(in_channels=self.data_size * 4,
                                out_channels=self.data_size * 8, kernel_size=4, stride=2, padding=0)
        # # out (?, self.data_size * 16, 1, 1)
        # self.conv_5 = nn.Conv2d(in_channels=self.data_size * 8,
        #                         out_channels=self.data_size * 16, kernel_size=4, stride=1, padding=0)

        # y = x*W^T + b in_features - x, out_features - y, W - (out_features, in_features),
        # b - features W and b - randomly
        self.fc = nn.Linear(in_features=self.data_size * 8 + self.meta_length, out_features=self.lambda_length + 1)

    def forward(self, data, meta):
        # print(data.shape)
        # print(meta.shape)
        conv1 = F.leaky_relu(self.conv_1(data), 0.2)
        # print(conv1.shape)
        conv2 = F.leaky_relu(self.conv_2(conv1), 0.2)
        #print(conv2.shape)
        conv3 = F.leaky_relu(self.conv_3(conv2), 0.2)
        #print(conv3.shape)
        conv4 = F.leaky_relu(self.conv_4(conv3), 0.2)
        #print(conv4.shape)
        # conv5 = F.leaky_relu(self.conv_5(conv4), 0.2)
        # print(conv5.shape)
        concat = torch.cat((conv4, meta), 1)
        result = self.fc(concat.squeeze())
        return result
