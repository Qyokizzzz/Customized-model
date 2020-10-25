__author__ = 'Ldaze'

import json
from core.data_processor import DatasetMaker
from core.model_builder import CustomizableModel


def main():
    configs = json.load(open('..\\configs\\config30.json', encoding='utf-8'))
    data = DatasetMaker(configs)

    model = CustomizableModel(configs)
    model.load()

    if configs['data']['split'] > 0:
        x_train, y_train = data.get_dataset(data.x_train, data.y_train)
        if y_train.shape[1] == 1:
            y_train = y_train.reshape(y_train.shape[0], y_train.shape[2])
        else:
            y_train = data.tile(y_train)

        train_pred = model.predict(x_train)
        train_pred = data.tile(train_pred)
        y_train = data.converter(y_train)
        train_pred = data.converter(train_pred)
    else:
        y_train, train_pred = None, None

    x_test, y_test = data.get_dataset(data.x_test, data.y_test, True)
    if y_test.shape[1] == 1:
        y_test = y_test.reshape(y_test.shape[0], y_test.shape[2])
    else:
        y_test = data.tile(y_test)

    test_pred = model.predict(x_test)
    test_pred = data.tile(test_pred)
    y_test = data.converter(y_test)
    test_pred = data.converter(test_pred)

    model.show_r2_score(y_train, y_test, train_pred, test_pred)
    model.show_mse_score(y_train, y_test, train_pred, test_pred)
    model.show_euclidean_distance(y_train, y_test, train_pred, test_pred)


if __name__ == "__main__":
    main()
