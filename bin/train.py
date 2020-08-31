__author__ = 'Ldaze'

import json
from core.data_processor import DatasetMaker
from core.model_builder import CustomizableModel


def main():
    configs = json.load(open('..\\configs\\config2.json', encoding='utf-8'))
    data = DatasetMaker(configs)
    x_train, y_train = data.get_dataset(data.x_train, data.y_train)
    x_test, y_test = data.get_dataset(data.x_test, data.y_test, False)
    if y_train.shape[1] == 1:
        y_train = y_train.reshape(y_train.shape[0], y_train.shape[2])
    if y_test.shape[1] == 1:
        y_test = y_test.reshape(y_test.shape[0], y_test.shape[2])

    x_test = x_test[0:960]
    y_test = y_test[0:960]
    model = CustomizableModel(configs)
    model.build()
    model.train(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
