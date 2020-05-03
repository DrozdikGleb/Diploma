import argparse

import torch
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable
import numpy as np

from modifiedLMGAN.DatasetLoader import get_loader
from modifiedLMGAN.ModifiedLMGAN import Generator
from feature_extraction.LambdaFeaturesCollector import LambdaFeaturesCollector
from feature_extraction.MetaFeaturesCollector import MetaFeaturesCollector


def get_meta(data_in: torch.Tensor):
    meta_list = []
    for data in data_in:
        meta_list.append(metaCollector.getShort(data.cpu().detach().numpy()))
    result = torch.stack(meta_list)
    return Variable(result.view((result.size(0), result.size(1), 1, 1)))


def get_mse(x: torch.Tensor, y: torch.Tensor) -> [float]:
    res = []
    x_in = np.squeeze(x.cpu().detach().numpy())
    y_in = np.squeeze(y.cpu().detach().numpy())
    for x, y in zip(x_in, y_in):
        res.append(mean_squared_error(x, y))
    return res


def test_generator(datasets, gen):
    results = []
    for data in datasets:
        metas = Variable(data[1])
        batch_size = data[0].size(0)
        noise = torch.randn(batch_size, 100)
        noise = noise.view((noise.size(0), noise.size(1), 1, 1))
        noise = Variable(noise)

        fake_data = gen(noise, metas)
        fake_metas = get_meta(fake_data)
        results.extend(get_mse(fake_metas, metas))

    return np.mean(np.array(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("-g", "--gen-location", required=True)
    args = parser.parse_args()
    metaCollector = MetaFeaturesCollector(16, 64)
    metaCollector.train(args.train_dir)
    lambdas = LambdaFeaturesCollector(16, 64)
    test_data = get_loader(args.test_dir, 16, 64, 2, metaCollector, lambdas, 100, 5,
                           train_meta=False)
    generator = Generator(16, 64, 2, metaCollector.getLength(), 100)
    generator.load_state_dict(torch.load(args.gen_location))
    generator.eval()
    print(test_generator(test_data, generator))
