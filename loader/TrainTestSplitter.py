import argparse
from os import listdir
from os.path import isfile, join
import os


def make_split(train_path, test_path):
    count = 0
    files = [f for f in listdir(train_path) if isfile(join(train_path, f))]
    for i, name in enumerate(files):
        if i % 10 == 0:
            count += 1
            os.rename(f"{train_path}{name}", f"{test_path}{name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="datasets/dprocessed_16_32_2/")
    parser.add_argument("--test-dir", default="datasets/dtest32/")
    args = parser.parse_args()
    make_split(args.train_dir, args.test_dir)
