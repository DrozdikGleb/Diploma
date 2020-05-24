import argparse

import torch
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from torch.nn import MSELoss
from torch.autograd import Variable
from tqdm import tqdm

from modifiedLMGAN.DatasetLoader import get_loader
from modifiedLMGAN.GraphBuilder import GraphBuilder
from modifiedLMGAN.LMGAN64 import Discriminator
from feature_extraction.LambdaFeaturesCollector import LambdaFeaturesCollector
from feature_extraction.MetaFeaturesCollector import MetaFeaturesCollector


def get_metas_and_lambdas(data_loader):
    datas = []
    meta_list = []
    lambdas_list = []
    for (data, data_meta, lambda_l) in data_loader:
        meta = data_meta[:, :].numpy().ravel().tolist()
        meta_list.append(meta)
        lambdas = lambda_l[:, :].numpy().astype(int).ravel().tolist()
        lambdas_list.append(lambdas)
        datas.append(data)
    return datas, meta_list, lambdas_list


def test_classifier(clf, clf_name, train_metas, train_lambdas, test_metas, test_lambdas):
    clf.fit(train_metas, train_lambdas)
    preds = clf.predict(test_metas)
    res = mean_squared_error(preds, test_lambdas)
    print(clf_name, "result:", res)


def test_classifiers(train_metas, train_lambdas, test_metas, test_lambdas):
    clfs = [(DecisionTreeClassifier(random_state=0), "DecisionTree"),
            (KNeighborsClassifier(n_neighbors=25), "KNN"),
            (MLPClassifier(random_state=0), "MLP")]
    for (clf, clf_name) in clfs:
        test_classifier(clf, clf_name, train_metas, train_lambdas, test_metas, test_lambdas)


def test_discriminators(datatest, test_datasets, test_metas, test_lambdas, discriminator_location):
    discriminator = Discriminator(16, 32, 2, metas.getLength(),
                                  lambdas.getLength())
    discriminator.load_state_dict(
        torch.load(discriminator_location))
    discriminator.eval()
    loss = []
    mse = MSELoss()
    #graph_builder = GraphBuilder()
    for data in tqdm(datatest, total=len(datatest)):
        dataset = Variable(data[0])
        cur_meta = Variable(data[1])
        cur_lambda = Variable(data[2]).squeeze()
        real_outputs = discriminator(dataset, cur_meta)
        d_real_labels_loss = mse(real_outputs[1:], cur_lambda)
        loss.append(d_real_labels_loss.cpu().detach().numpy())
    # for i in range(len(test_metas)):
    #     dataset = Variable(test_datasets[i]).squeeze()
    #     cur_meta = Variable(test_metas[i])
    #     cur_lambda = Variable(test_lambdas[i]).squeeze()
    #     #graph1, graph2 = graph_builder.build_complete_graph(dataset)
    #
    #     real_outputs = discriminator(dataset, cur_meta)
    #     d_real_labels_loss = mse(real_outputs[1:], cur_lambda)
    #     loss.append(d_real_labels_loss.cpu().detach().numpy())
    print(np.mean(loss))


def start_test(train_dir, test_dir, discriminator_location):
    batch_size = 1
    workers = 5
    data_train = get_loader(train_dir, 16, datasize, 2, metas, lambdas, batch_size, workers)
    data_test = get_loader(test_dir, 16, datasize, 2, metas, lambdas, batch_size, workers,
                           train_meta=False)

    _, train_m, train_l = get_metas_and_lambdas(data_train)
    datasets, test_m, test_l = get_metas_and_lambdas(data_test)

    #test_classifiers(train_m, train_l, test_m, test_l)
    test_discriminators(data_test, datasets, test_m, test_l, discriminator_location)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="../loader/datasets/dprocessed_16_32_2/")
    parser.add_argument("--test-dir", default="../loader/datasets/dtest32/")
    parser.add_argument("-d", "--disc-location",
                        default="../modifiedLMGAN/models_LMGAN64/discriminator-16_32_2-20.pkl")
    datasize = 32
    lambdas = LambdaFeaturesCollector(16, datasize)
    metas = MetaFeaturesCollector(16, datasize)
    args = parser.parse_args()
    start_test(args.train_dir, args.test_dir, args.disc_location)
