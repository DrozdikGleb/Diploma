from openml import datasets as open_ml_dataset
import csv
import shutil
import os

from openml import OpenMLDataset


class DatasetInfo:
    def __init__(self, dataset_id, instances_number, features_number, class_number):
        self.id = dataset_id
        self.instances_number = instances_number
        self.features_number = features_number
        self.class_number = class_number


def get_datasets():
    datasets_list = open_ml_dataset.list_datasets()

    datasets = []
    for (dataset_id, dataset) in datasets_list.items():
        if 'NumberOfInstances' not in dataset or 'NumberOfClasses' not in dataset:
            continue
        if dataset['NumberOfInstances'] > 20000.0:
            continue
        datasets.append(DatasetInfo(dataset_id, dataset['NumberOfInstances'],
                                       dataset['NumberOfFeatures'], dataset['NumberOfClasses']))
    return datasets


def get_processed_datasets():
    processed = set()

    with open('./datasets/datasets.csv', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            processed.add(int(row[0]))
    return processed


def download_arff_files(datasets, processed_datasets):
    with open('./datasets/datasets.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for dataset in datasets:
            try:
                if dataset.id not in processed_datasets:
                    print("processing dataset %d" % dataset.id)
                    ds: OpenMLDataset = open_ml_dataset.get_dataset(dataset.id)
                    file = ds.data_file
                    filename, file_extension = os.path.splitext(file)
                    cur_dir = os.path.abspath(os.getcwd())
                    shutil.copyfile(file, f'{cur_dir}/datasets/arff/{dataset.id}{file_extension}')
                    features = ds.features
                    target_name = ds.default_target_attribute
                    if target_name is not None:
                        target_id = (-1, '')
                        for (_, f) in features.items():
                            if f.name == target_name:
                                target_id = (f.index, f.data_type)
                        writer.writerow(
                            [dataset.id, dataset.instances_number, dataset.features_number,
                             dataset.class_number, target_id[0], ds.url])
                        csvfile.flush()
            except :
                print("error")


if __name__ == '__main__':
    datasets = get_datasets()

    processed_datasets = get_processed_datasets()

    download_arff_files(datasets, processed_datasets)
