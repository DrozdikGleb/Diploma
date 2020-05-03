import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Discriminator(nn.Module):

    def __init__(self, features_size: int, instances_size: int, classes_size: int, meta_length: int,
                 lambda_length: int):
        super(Discriminator, self).__init__()
        #self.graph_builder = GraphBuilder()

        self.data_size = instances_size
        self.meta_length = meta_length
        self.lambda_length = lambda_length

        self.conv1 = GCNConv(features_size, 8)
        self.conv2 = GCNConv(8, 4)
        self.conv3 = GCNConv(4, 1)

        # y = x*W^T + b in_features - x, out_features - y, W - (out_features, in_features),
        # b - features W and b - randomly
        self.fc = nn.Linear(in_features=instances_size * 2 + self.meta_length, out_features=self.lambda_length + 1)

    def forward(self, graph1, graph2, meta):
        x_1, edge_index_1 = graph1.x, graph1.edge_index
        x_1 = self.conv1(x_1, edge_index_1)
        x_1 = self.conv2(x_1, edge_index_1)
        x_1 = self.conv3(x_1, edge_index_1)

        x_2, edge_index_2 = graph2.x, graph2.edge_index
        x_2 = self.conv1(x_2, edge_index_2)
        x_2 = self.conv2(x_2, edge_index_2)
        x_2 = self.conv3(x_2, edge_index_2)

        concat = torch.cat((x_1.squeeze(), x_2.squeeze(), meta.squeeze()))
        result = self.fc(concat.squeeze())
        return result

    def build_complete_graph(self, data):
        pass


class Generator(nn.Module):

    def __init__(self, features_size: int, instances_size: int, classes_size: int, meta_length: int, z_length: int):
        super(Generator, self).__init__()

        self.data_size = instances_size
        self.meta_length = meta_length
        self.z_length = z_length

        self.fc_z = nn.ConvTranspose2d(in_channels=self.z_length,
                                       out_channels=self.data_size * 4, kernel_size=4, stride=1, padding=0)
        self.fc_meta = nn.ConvTranspose2d(in_channels=self.meta_length,
                                          out_channels=self.data_size * 4, kernel_size=4, stride=1, padding=0)
        self.deconv1 = nn.ConvTranspose2d(in_channels=self.data_size * 8,
                                          out_channels=self.data_size * 4, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=self.data_size * 4,
                                          out_channels=self.data_size * 2, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=self.data_size * 2,
                                          out_channels=self.data_size, kernel_size=(4, 1), stride=(2, 1),
                                          padding=(1, 0))
        self.deconv4 = nn.ConvTranspose2d(in_channels=self.data_size,
                                          out_channels=classes_size, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))

    def forward(self, z, meta):
        fc_z = F.leaky_relu(self.fc_z(z), 0.2)
        fc_meta = F.leaky_relu(self.fc_meta(meta), 0.2)

        fc = torch.cat((fc_z, fc_meta), 1)
        deconv1 = F.leaky_relu(self.deconv1(fc), 0.2)
        deconv2 = F.leaky_relu(self.deconv2(deconv1), 0.2)
        deconv3 = F.leaky_relu(self.deconv3(deconv2), 0.2)
        deconv4 = F.sigmoid(self.deconv4(deconv3))
        return deconv4


