__author__ = 'Ldaze'

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers \
    import LSTM, Dense, Activation, Dropout, Bidirectional,\
    Flatten, TimeDistributed, RepeatVector, Conv1D, Permute, Multiply, Lambda, Input

import json
import numpy as np
from core.data_processor import DatasetMaker
from core.model_builder import CustomizableModel
from matplotlib import pyplot as plt
from matplotlib import font_manager


def move_predict(sequence, model):
    y_pred = model.predict(sequence)
    return np.append(sequence, y_pred[-1].reshape(1, 1, 3), axis=0)


def r_predict(sequence, pre_steps, model):
    for i in range(pre_steps):
        sequence = move_predict(sequence, model)
    return sequence


def predict_more(x, pre_steps, model):
    temp = [x]
    for i in range(pre_steps):
        tmp = model.predict(temp[i].reshape(1, 1, 3))
        temp.append(tmp)
    return np.array(temp[1:])


def main():
    configs = json.load(open('..\\configs\\config4.json', encoding='utf-8'))
    data = DatasetMaker(configs)
    model = CustomizableModel(configs)
    model.load()
    x_test, y_test = data.get_dataset(data.x_test, data.y_test)
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[2])
    # print(x_test.shape, y_test.shape)
    print(x_test)
    print(y_test)

    pre_steps = 10
    test_s = x_test[x_test.shape[0] - pre_steps]
    y_s = y_test[x_test.shape[0] - pre_steps:]
    # y_seq = r_predict(test_s, pre_steps, model.model)
    y_true_pred = model.predict(x_test)
    y_true_pred = y_true_pred[y_true_pred.shape[0] - pre_steps:]
    y_pred = predict_more(test_s, pre_steps, model)

    # y_pred = y_seq[x_test.shape[0] - pre_steps: x_test.shape[0]]
    y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[2])
    y_s = data.converter(y_s)
    y_pred = data.converter(y_pred)
    y_true_pred = data.converter(y_true_pred)

    model.show_r2_score(None, y_s, None, y_pred)
    model.show_mse_score(None, y_s, None, y_pred)

    # print(y_s)
    # print(y_pred)
    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title(configs['picture']['labels'][0])
    ax1.plot(y_s[:, 0], 'r', label='Original data')
    ax1.plot(y_pred[:, 0], 'b--', label='Fitting data')
    ax1.plot(y_true_pred[:, 0], 'g', label=' True X fitting data')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title(configs['picture']['labels'][1])
    ax2.plot(y_s[:, 1], 'r', label='Original data')
    ax2.plot(y_pred[:, 1], 'b--', label='Fitting data')
    ax2.plot(y_true_pred[:, 1], 'g', label=' True X fitting data')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title(configs['picture']['labels'][2])
    ax3.plot(y_s[:, 2], 'r', label='Original data')
    ax3.plot(y_pred[:, 2], 'b--', label='Fitting data')
    ax3.plot(y_true_pred[:, 2], 'g', label=' True X fitting data')

    font = font_manager.FontProperties(fname=configs["picture"]["font"])
    ax3.legend(loc="best", prop=font)
    # plt.savefig("..\\saved_pictures\\move_predict")
    plt.show()


if __name__ == "__main__":
    main()
