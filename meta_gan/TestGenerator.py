import torch
from scipy.spatial.distance import mahalanobis
from torch.autograd import Variable

from meta_gan.DatasetLoader import get_loader
from meta_gan.Models import Generator
from meta_gan.feature_extraction.LambdaFeaturesCollector import LambdaFeaturesCollector
from meta_gan.feature_extraction.MetaFeaturesCollector import MetaFeaturesCollector
import numpy as np


def to_variable(x):
    #x = x.cuda()
    return Variable(x)


def getMeta(data_in: torch.Tensor):
    meta_list = []
    for data in data_in:
        meta_list.append(metaCollector.getShort(data.cpu().detach().numpy()))
    result = torch.stack(meta_list)
    return to_variable(result.view((result.size(0), result.size(1), 1, 1)))


def getDistance(x: torch.Tensor, y: torch.Tensor) -> [float]:
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


if __name__ == '__main__':
    metaCollector = MetaFeaturesCollector(16, 64)
    metaCollector.train(f"../data-loader/datasets/dprocessed_16_64_2/")
    lambdas = LambdaFeaturesCollector(16, 64)
    loader = get_loader(f"../data-loader/datasets/dtest/", 16, 64, 2, metaCollector, lambdas, 100, 5, train_meta=False)
    generator = Generator(16, 64, 2, metaCollector.getLength(), 100)
    generator.load_state_dict(
        torch.load(
            f'./models_graph_1/generator-16_64_2-1.pkl'))
    generator.eval()
    results = []
    for i, data in enumerate(loader):
        print(i)
        metas = to_variable(data[1])
        batch_size = data[0].size(0)
        noise = torch.randn(batch_size, 100)
        noise = noise.view((noise.size(0), noise.size(1), 1, 1))
        noise = to_variable(noise)

        fake_data = generator(noise, metas)
        fake_metas = getMeta(fake_data)
        results.extend(getDistance(fake_metas, metas))

    print(results)
    print(max(results))
    print(min(results))
    print(np.mean(np.array(results)))
