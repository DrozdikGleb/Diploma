import os

import torch
from torch.utils import data
import numpy as np
from meta_gan.feature_extraction.MetaFeaturesCollector import MetaFeaturesCollector
from meta_gan.feature_extraction.LambdaFeaturesCollector import LambdaFeaturesCollector


class DatasetFolder(data.Dataset):

    def __init__(self, path: str, features_size: int, instances_size: int, classes_size: int,
                 meta: MetaFeaturesCollector, lambdas: LambdaFeaturesCollector, train_meta: bool):
        self.root = path
        self.features = features_size
        self.instances = instances_size
        self.classes = classes_size
        paths = []
        for fname in os.listdir(self.root):
            path = os.path.join(self.root, fname)
            if not os.path.isdir(path):
                paths.append(path)
        print(paths)
        self.data_paths = paths
        self.data_names = list(map(lambda x: x.split('/')[-1].split('.')[0], paths))
        self.meta_features = meta
        if train_meta:
            self.meta_features.train(self.root)
        self.lambda_features = lambdas

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        data_path = self.data_paths[index]
        data_name = self.data_names[index]
        data_np = np.load(data_path)
        dataset_tensor = torch.from_numpy(data_np).float().view((self.classes, self.instances, self.features))

        meta_tensor = self.meta_features.get(data_np, data_name)
        meta_tensor = meta_tensor.view(self.meta_features.getLength(), 1, 1)
        lambda_tensor = self.lambda_features.get(data_np, name_in=data_name)
        return dataset_tensor, meta_tensor, lambda_tensor

    def __len__(self):
        return len(self.data_paths)


def get_loader(path: str, features_size: int, instances_size: int, classes_size: int, meta: MetaFeaturesCollector,
               lambdas: LambdaFeaturesCollector, batch_size: int, num_workers: int, train_meta: bool = True):
    datasets_inner = DatasetFolder(path, features_size, instances_size, classes_size, meta, lambdas, train_meta)

    data_loader = data.DataLoader(
        dataset=datasets_inner,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return data_loader


if __name__ == '__main__':
    datasets = DatasetFolder("../processed_data/processed_16_64_2/", 16, 64, 2, MetaFeaturesCollector(16, 64),
                             LambdaFeaturesCollector(16, 64), True)
    for i in range(len(datasets)):
        print(datasets.__getitem__(i))
