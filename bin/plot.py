__author__ = 'Ldaze'

import json
from core.data_processor import DatasetMaker
from core.model_builder import CustomizableModel

from matplotlib import pyplot as plt
from matplotlib import font_manager

import os


def main():
    configs = json.load(open('..\\configs\\config10.json', encoding='utf-8'))
    data = DatasetMaker(configs)
    try:
        x_train, y_train = data.get_dataset(data.x_train, data.y_train)

        if y_train.shape[1] == 1:
            y_train = y_train.reshape(y_train.shape[0], y_train.shape[2])
        else:
            y_train = data.tile(y_train)

    except ValueError:
        x_train = None

    x_test, y_test = data.get_dataset(data.x_test, data.y_test, True)

    if y_test.shape[1] == 1:
        y_test = y_test.reshape(y_test.shape[0], y_test.shape[2])
    else:
        y_test = data.tile(y_test)

    model = CustomizableModel(configs)
    model.load()
    try:
        train_pred = model.predict(x_train)
        train_pred = data.tile(train_pred)
    except ValueError:
        pass

    test_pred = model.predict(x_test)
    test_pred = data.tile(test_pred)

    y_test = data.converter(y_test)
    test_pred = data.converter(test_pred)

    font = font_manager.FontProperties(fname=configs["picture"]["font"])
    fig = plt.figure(figsize=(20, 10))
    ax = [0 for i in range(y_test.shape[1])]

    for i in range(len(ax)):
        ax[i] = fig.add_subplot(2, 2, i + 1)
        ax[i].set_title(configs['picture']['labels'][i])
        ax[i].plot(y_test[:, i], 'r', label='Original data')
        ax[i].plot(test_pred[:, i], 'b--', label='Fitting data')

    ax[i].legend(loc="best", prop=font)
    plt.rcParams.update({'font.size': 40})
    if configs['picture']['is_saved']:
        str_list = configs['model']['load_path'].split('\\')
        save_path = os.path.join(configs['picture']['save_dir'], str_list[-1].split('.')[0])
        plt.savefig(save_path)
    plt.show()

    model.show_r2_score(None, y_test, None, test_pred)
    model.show_mse_score(None, y_test, None, test_pred)
    model.show_euclidean_distance(None, y_test, None, test_pred)


if __name__ == "__main__":
    main()
