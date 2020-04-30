import numpy as np
from os import walk
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from pathlib import Path


def load(name: str) -> (np.ndarray, np.ndarray):
    name_path = f'{raw_data_str}{name}/'
    data_path = f'{name_path}{name}_data.npy'
    class_path = f'{name_path}{name}_class.npy'
    data_np = np.load(data_path)
    class_np = np.load(class_path)
    return data_np, class_np


def prepareLabels(labels_in: np.ndarray) -> np.ndarray:
    onehot = OneHotEncoder(sparse=False, categories='auto')
    result = onehot.fit_transform(labels_in)
    return result


def scaleData(zero_in: np.ndarray, one_in: np.ndarray) -> (np.ndarray, np.ndarray):
    data = np.append(zero_in, one_in, axis=0)
    data = data.astype(np.float64)
    min_max_scaler = MinMaxScaler()
    data_out = min_max_scaler.fit_transform(data)
    return data_out[:target_instances], data_out[target_instances:]


def shrinkFeatures(zero_in: np.ndarray, one_in: np.ndarray) -> (np.ndarray, np.ndarray):
    clf = ExtraTreesClassifier(n_estimators=50, random_state=239)
    data = np.append(zero_in, one_in, axis=0)
    labels = np.append(np.zeros(target_instances), np.ones(target_instances))
    clf.fit(data, labels)
    model = SelectFromModel(estimator=clf, threshold=-np.inf, prefit=True, max_features=target_features)
    new_data = model.transform(data)
    if new_data.shape[1] == target_features:
        return new_data[:target_instances], new_data[target_instances:]
    else:
        given_features = data.shape[1]
        new_data = data[:, np.random.choice(given_features.shape[1], target_features, replace=False)]
        assert new_data.shape[1] == target_features
        return new_data[:target_instances], new_data[target_instances:]


def swapCol(data_in: np.array, frm: int, to: int) -> np.ndarray:
    data_in[:, [frm, to]] = data_in[:, [to, frm]]
    return data_in


def swapRow(data_in: np.array, frm: int, to: int) -> np.ndarray:
    data_in[[frm, to], :] = data_in[[to, frm], :]
    return data_in


def swapRows(data_in: np.array, frm: np.ndarray, to: np.ndarray, col_num: int) -> np.ndarray:
    to_sorted = np.argsort(data_in[to, col_num])[::-1]
    for i, j in enumerate(to_sorted):
        frm_i = frm[i]
        to_i = to[j]
        data_in[[frm_i, to_i], :] = data_in[[to_i, frm_i], :]
    return data_in


def findMax(zero_in: np.ndarray, one_in: np.ndarray, step: int) -> (int, np.ndarray, np.ndarray):
    max_col_mean = []
    (rows, cols) = zero_in.shape
    for i in range(cols):
        zero_col = zero_in[:, i]
        zero_max_ind = np.argpartition(zero_col, -step)[-step:]
        if rows != step:
            zero_av = np.mean(np.delete(zero_col, zero_max_ind))
        else:
            zero_av = 0.0
        zero_val = np.mean(zero_col[zero_max_ind]) - zero_av
        one_col = one_in[:, i]
        one_max_ind = np.argpartition(one_col, -step)[-step:]
        if rows != step:
            one_av = np.mean(np.delete(one_col, one_max_ind))
        else:
            one_av = 0.0
        one_val = np.mean(one_col[one_max_ind]) - one_av
        max_col_mean.append((one_val + zero_val) / 2)

    max_col_mean = np.array(max_col_mean)
    max_col = np.argmax(max_col_mean).item(0)
    zero_max_ind = np.argpartition(zero_in[:, max_col], -step)[-step:]
    one_max_ind = np.argpartition(one_in[:, max_col], -step)[-step:]
    return max_col, zero_max_ind, one_max_ind


def sortData(zero_in: np.ndarray, one_in: np.ndarray) -> (np.ndarray, np.ndarray):
    zero_out = np.copy(zero_in)
    one_out = np.copy(one_in)
    step = target_instances // target_features
    for col_num in range(target_features):
        row_num = col_num * step
        col_max, zero_max, one_max = findMax(zero_out[row_num:, col_num:], one_out[row_num:, col_num:], step)
        col_max += col_num
        rows_from = np.array(range(row_num, row_num + 4))
        zero_max = zero_max + row_num
        zero_out = swapCol(zero_out, col_num, col_max)
        zero_out = swapRows(zero_out, rows_from, zero_max, col_num)
        one_max = one_max + row_num
        one_out = swapCol(one_out, col_num, col_max)
        one_out = swapRows(one_out, rows_from, one_max, col_num)

    return zero_out, one_out


def writeData(name_in: str, zero_num: int, one_num: int, zero_in: np.ndarray, one_in: np.ndarray) -> bool:
    zero_sorted, one_sorted = sortData(zero_in, one_in)
    #zero_sorted, one_sorted = (zero_in, one_in)
    raw_path = f'{done_data_str_raw}/{name_in}/'
    raw_name = f'{raw_path}{name_in}_{zero_num}_{one_num}'
    path = Path(raw_path)
    path.mkdir(parents=True, exist_ok=True)
    zero_raw_name = f'{raw_name}_zero'
    one_raw_name = f'{raw_name}_one'
    np.save(zero_raw_name, zero_sorted)
    np.save(one_raw_name, one_sorted)
    data_out = np.stack((zero_sorted, one_sorted))
    data_name = f'{done_data_str}{name_in}_{zero_num}_{one_num}'
    np.save(data_name, data_out)
    return True


def prepare(dataset_name: str, dataset_in: (np.ndarray, np.ndarray), pbar: tqdm):
    data_in, classes_in = dataset_in
    labels_in = prepareLabels(classes_in)
    (_, classes_count) = labels_in.shape
    (instances_size, features_size) = data_in.shape
    pbar.set_description("Processing %s:[%d, %d, %d]" % (dirname, instances_size, features_size, classes_count))

    if features_size < target_features:
        return False

    per_class_data = [[] for x in range(classes_count)]

    for k in range(classes_count):
        for j in range(instances_size):
            if labels_in[j][k] == 1.0:
                per_class_data[k].append(data_in[j])

    for j in range(classes_count - 1):
        for k in range(j + 1, classes_count):
            zero_data = per_class_data[j]
            zero_len = len(zero_data)
            one_data = per_class_data[k]
            one_len = len(one_data)
            if (zero_len >= target_instances) and (one_len >= target_instances):
                zero_data = np.array(zero_data)
                zero_data = np.array(zero_data[np.random.choice(zero_len, target_instances, replace=False)])
                one_data = np.array(one_data)
                one_data = np.array(one_data[np.random.choice(one_len, target_instances, replace=False)])
                scaled_zero_data, scaled_one_data = scaleData(zero_data, one_data)
                if features_size > target_features:
                    shrunk_zero_data, shrunk_one_data = shrinkFeatures(scaled_zero_data, scaled_one_data)
                else:
                    shrunk_zero_data = scaled_zero_data
                    shrunk_one_data = scaled_one_data
                writeData(dataset_name, j, k, shrunk_zero_data, shrunk_one_data)


if __name__ == '__main__':
    target_features = 16
    target_instances = 64
    target_classes = 2
    raw_data_str = "./datasets/np_raw/"
    done_data_str = f"./datasets/dprocessed_{target_features}_{target_instances}_{target_classes}/"
    done_data_str_raw = f"./datasets/dprocessed_{target_features}_{target_instances}_{target_classes}/raw"
    done_data_str_path = Path(f'{done_data_str}')
    done_data_str_path.mkdir(parents=True, exist_ok=True)
    done_data_str_raw_path = Path(f'{done_data_str_raw}')
    done_data_str_raw_path.mkdir(parents=True, exist_ok=True)

    start_from = 0

    non_processed = []
    for (dirpath, dirnames, filenames) in walk(raw_data_str):
        pbar = tqdm(enumerate(dirnames), total=len(dirnames))
        for cc, dirname in pbar:
            if cc >= start_from:
                prepare_status = prepare(dirname, load(dirname), pbar)
                if prepare_status is not None:
                    non_processed.append(dirname)

    print(non_processed)
