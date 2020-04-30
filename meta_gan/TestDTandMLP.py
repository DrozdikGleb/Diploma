import torch
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from meta_gan.DatasetLoader import get_loader
from meta_gan.Models import Generator, Discriminator
from meta_gan.feature_extraction.LambdaFeaturesCollector import LambdaFeaturesCollector
from meta_gan.feature_extraction.MetaFeaturesCollector import MetaFeaturesCollector
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    datasize = 64
    z_size = 100
    batch_size = 1
    workers = 5
    lambdas = LambdaFeaturesCollector(16, 64)
    metas = MetaFeaturesCollector(16, 64)
    dataloader = get_loader(f"../processed_data/processed_16_64_2/", 16, 64, 2, metas, lambdas, batch_size, workers)
    datatest = get_loader(f"../processed_data/test/", 16, 64, 2, metas, lambdas, batch_size, workers, train_meta=False)

    meta_list = []
    lambdas_list = []
    for i, (data, meta, lambda_l) in enumerate(dataloader):
        print(i)
        meta_o = meta[:, :].numpy()
        meta_o = meta_o.ravel()
        meta_o = meta_o.tolist()
        meta_list.append(meta_o)
        lambdas_o = lambda_l[:, :].numpy().astype(int).ravel().tolist()
        lambdas_list.append(lambdas_o)

    meta_list_test = []
    lambdas_list_test = []
    for i, (data, meta, lambda_l) in enumerate(datatest):
        print(i)
        meta_o = meta[:, :].numpy()
        meta_o = meta_o.ravel()
        meta_o = meta_o.tolist()
        meta_list_test.append(meta_o)
        lambdas_o = lambda_l[:, :].numpy().astype(int).ravel().tolist()
        lambdas_list_test.append(lambdas_o)

    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(meta_list, lambdas_list)
    pred = dt.predict(meta_list_test)
    score = mean_squared_error(pred, lambdas_list_test)
    print(score)

    dt = KNeighborsClassifier(n_neighbors=25)
    dt.fit(meta_list, lambdas_list)
    pred = dt.predict(meta_list_test)
    score = mean_squared_error(pred, lambdas_list_test)
    print(score)

    dt = MLPClassifier(random_state=0)
    dt.fit(meta_list, lambdas_list)
    pred = dt.predict(meta_list_test)
    score = mean_squared_error(pred, lambdas_list_test)
    print(score)
