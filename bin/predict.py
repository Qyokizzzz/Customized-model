__author__ = 'Ldaze'

import json
from core.data_processor import DatasetMaker
from core.model_builder import CustomizableModel


def main():
    configs = json.load(open('..\\configs\\config2.json', encoding='utf-8'))
    data = DatasetMaker(configs)
    x_train, y_train = data.get_dataset(data.x_train, data.y_train)
    x_test, y_test = data.get_dataset(data.x_test, data.y_test, True)
    if y_train.shape[1] == 1:
        y_train = y_train.reshape(y_train.shape[0], y_train.shape[2])
    else:
        y_train = data.tile(y_train)
    if y_test.shape[1] == 1:
        y_test = y_test.reshape(y_test.shape[0], y_test.shape[2])
    else:
        y_test = data.tile(y_test)

    model = CustomizableModel(configs)
    model.load()
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    train_pred = data.tile(train_pred)
    test_pred = data.tile(test_pred)
    model.show_score(y_train, y_test, train_pred, test_pred)


if __name__ == "__main__":
    main()
