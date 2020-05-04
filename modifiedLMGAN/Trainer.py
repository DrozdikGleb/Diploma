import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
from scipy.spatial.distance import mahalanobis
from torch import optim
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, MSELoss
import numpy as np

from modifiedLMGAN.DatasetLoader import get_loader
from modifiedLMGAN.GraphBuilder import GraphBuilder
from modifiedLMGAN.ModifiedLMGAN import Generator, Discriminator
from feature_extraction.LambdaFeaturesCollector import LambdaFeaturesCollector
from feature_extraction.MetaFeaturesCollector import MetaFeaturesCollector


class Trainer:
    def __init__(self, num_epochs: int = 500, cuda: bool = False, models_path: str = "./models_graph"):
        self.features = 16
        self.instances = 64
        self.classes = 2
        self.z_size = 100
        self.batch_size = 1
        self.workers = 5
        self.num_epochs = num_epochs
        self.cuda = cuda
        self.log_step_print = 10
        self.save_period = 1
        self.graph_builder = GraphBuilder()

        self.models_path = models_path

        self.lambdas = LambdaFeaturesCollector(self.features, self.instances)
        self.metas = MetaFeaturesCollector(self.features, self.instances)
        self.data_loader = get_loader(
            f"../loader/datasets/dprocessed_{self.features}_{self.instances}_{self.classes}/",
            self.features, self.instances, self.classes, self.metas,
            self.lambdas, self.batch_size,
            self.workers)
        self.test_loader = get_loader("../loader/datasets/dtest8/", 16, 8, 2, self.metas,
                                      self.lambdas, 228, 5,
                                      train_meta=False)

        self.generator = Generator(self.features, self.instances, self.classes,
                                   self.metas.getLength(), self.z_size)
        self.discriminator = Discriminator(self.features, self.instances, self.classes,
                                           self.metas.getLength(), self.lambdas.getLength())

        self.lr = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999

        self.g_optimizer = optim.Adam(self.generator.parameters(),
                                      self.lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(self.discriminator.parameters(),
                                      self.lr, [self.beta1, self.beta2])

        self.cross_entropy = BCEWithLogitsLoss()
        self.mse = MSELoss()

    def getDistance(self, x: torch.Tensor, y: torch.Tensor) -> [float]:
        x_in = np.squeeze(x.cpu().detach().numpy())
        y_in = np.squeeze(y.cpu().detach().numpy())
        results = []
        for (xx, yy) in zip(x_in, y_in):
            try:
                V = np.cov(np.array([xx, yy]).T)
                V[np.diag_indices_from(V)] += 0.1
                IV = np.linalg.inv(V)
                D = mahalanobis(xx, yy, IV)
            except:
                D = 0.0
            results.append(D)
        return results

    def getMeta(self, data_in: torch.Tensor):
        meta_list = []
        for data in data_in:
            meta_list.append(self.metas.getShort(data.cpu().detach().numpy()))
        result = torch.stack(meta_list)
        return Variable(result.view((result.size(0), result.size(1), 1, 1)))

    def getLambda(self, data_in: torch.Tensor):
        lamba_list = []
        for data in data_in:
            lamba_list.append(self.lambdas.get(data.cpu().detach().numpy()))
        result = torch.stack(lamba_list)
        return Variable(result)

    def get_d_real(self, dataset, metas, lambdas, zeros):
        graph1, graph2 = self.graph_builder.build_complete_graph(dataset)
        real_outputs = self.discriminator(graph1, graph2, metas)
        # real_outputs = self.discriminator(dataset, metas)

        d_real_labels_loss = self.mse(real_outputs[1:], lambdas)

        d_real_rf_loss = self.mse(real_outputs[:1], zeros)  #
        return d_real_labels_loss + 0.7 * d_real_rf_loss, d_real_labels_loss, d_real_rf_loss

    def get_d_fake(self, dataset, noise, metas, ones):
        fake_data = self.generator(noise, metas)
        fake_data_metas = self.getMeta(fake_data)

        graph1, graph2 = self.graph_builder.build_complete_graph(dataset)
        fake_outputs = self.discriminator(graph1, graph2, fake_data_metas)
        # fake_outputs = self.discriminator(fake_data, fake_data_metas)
        fake_lambdas = self.getLambda(fake_data).squeeze()
        d_fake_labels_loss = self.cross_entropy(fake_outputs[1:], fake_lambdas)
        d_fake_rf_loss = self.mse(fake_outputs[:1], ones)
        return 0.7 * d_fake_rf_loss + 0.6 * d_fake_labels_loss, d_fake_labels_loss, d_fake_rf_loss

    def train(self):
        total_steps = len(self.data_loader)
        for epoch in range(self.num_epochs):
            loss = []
            print(len(self.data_loader))
            for i, data in enumerate(self.data_loader):
                dataset = Variable(data[0])
                metas = Variable(data[1])
                lambdas = Variable(data[2]).squeeze()
                batch_size = data[0].size(0)
                noise = torch.randn(batch_size, self.z_size)
                noise = noise.view((noise.size(0), noise.size(1), 1, 1))
                noise = Variable(noise)
                zeros = torch.zeros([batch_size, 1], dtype=torch.float32)
                zeros = Variable(zeros)
                ones = torch.ones([batch_size, 1], dtype=torch.float32)
                ones = Variable(ones)

                d_real_loss, d_real_labels_loss, d_real_rf_loss = \
                    self.get_d_real(dataset, metas, lambdas, zeros)

                d_fake_loss, d_fake_labels_loss, d_fake_rf_loss = \
                    self.get_d_fake(dataset, noise, metas, ones)

                d_loss = d_real_loss + 0.8 * d_fake_loss
                self.generator.zero_grad()
                self.discriminator.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                noise = torch.randn(batch_size, self.z_size)
                noise = noise.view(noise.size(0), noise.size(1), 1, 1)
                noise = Variable(noise)
                fake_data = self.generator(noise, metas)

                graph1, graph2 = self.graph_builder.build_complete_graph(dataset)
                fake_outputs = self.discriminator(graph1, graph2, metas)
                # fake_outputs = self.discriminator(fake_data, metas)
                g_fake_rf_loss = self.mse(fake_outputs[:1], zeros)
                fake_metas = self.getMeta(fake_data)
                g_fake_meta_loss = self.mse(fake_metas, metas)
                g_loss = 0.7 * g_fake_rf_loss + g_fake_meta_loss

                # minimize log(1 - D(G(z)))
                self.generator.zero_grad()
                self.discriminator.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                if (i + 1) % self.log_step_print == 0:
                    print((
                        f'[{datetime.now()}] Epoch[{epoch}/{self.num_epochs}], Step[{i}/{total_steps}],\n'
                        f' D_losses: [{d_real_rf_loss}|{d_real_labels_loss}|{d_fake_rf_loss}|{d_fake_labels_loss}],\n'
                        f'G_losses:[{g_fake_rf_loss}|{g_fake_meta_loss}]'
                    ))

            # saving
            if self.save_period == 1:
                done_data_str_path = Path(self.models_path)
                done_data_str_path.mkdir(parents=True, exist_ok=True)
                g_path = os.path.join(self.models_path,
                                      f'generator-{self.features}_{self.instances}_{self.classes}-{epoch + 1}.pkl')
                d_path = os.path.join(self.models_path,
                                      f'discriminator-{self.features}_{self.instances}_{self.classes}-{epoch + 1}.pkl')
                torch.save(self.generator.state_dict(), g_path)
                torch.save(self.discriminator.state_dict(), d_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./models_graph")
    args = parser.parse_args()
    trainer = Trainer(num_epochs=20, models_path=args.model_path)
    trainer.train()
