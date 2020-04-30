import openml
import csv
import shutil
import os

from openml import OpenMLDataset


def list_datasets():
    return openml.datasets.list_datasets()


def load_dataset(no):
    return openml.datasets.get_dataset(no)


if __name__ == '__main__':
    openml_list = list_datasets()
    #openml_list = {k: openml_list[k] for k in list(openml_list)[:5]}
    processed = set()
    with open('./datasets/datasets.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            processed.add(row[0])


    with open('./datasets/datasets.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        p = 0
        for (i, dsd) in openml_list.items():
            if p == 100:
                break
            p += 1
            try:
                did = dsd['did']
                if str(did) not in processed:
                    processed.add(str(did))
                    print(did)
                    ds: OpenMLDataset = load_dataset(did)
                    file = ds.data_file
                    filename, file_extension = os.path.splitext(file)
                    cur_dir = os.path.abspath(os.getcwd())
                    shutil.copyfile(file, f'{cur_dir}/datasets/arff/{did}{file_extension}')
                    features = ds.features
                    target_name = ds.default_target_attribute
                    if target_name is not None:
                        target_id = (-1, '')
                        for (_, f) in features.items():
                            if f.name == target_name:
                                target_id = (f.index, f.data_type)
                        writer.writerow(
                            [did, dsd['NumberOfInstances'], dsd['NumberOfFeatures'], dsd['NumberOfClasses'],
                             target_id[0],
                             ds.url])
                        csvfile.flush()

                    else:
                        print("Skipped", dsd['did'])
                else:
                    print("Already done")
            except:
                print("error")
