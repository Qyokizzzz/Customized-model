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


# def read_dict_into_list(dic, lis):
#     for i in dic.keys():
#         if type(dic[i]) == dict:
#             read_dict_into_list(dic[i], lis)
#         else:
#             if i == 'type':
#                 lis.append(dic[i])
#                 lis.append('(')
#             else:
#                 lis.append(i)
#                 lis.append('=')
#                 lis.append(dic[i])
#                 lis.append(',')
#     temp = lis.pop()
#     if temp != ',':
#         lis.append(temp)
#     lis.append(')')
#
#
# def test():
#     command = []
#     inputs = Input(shape=(configs['data']['time_steps'], configs['data']['pre_steps']))
#     for layer in configs['model']['layers']:
#         tmp = []
#         read_dict_into_list(layer, tmp)
#         command.append(''.join('%s' % item for item in tmp))
#     return command

def move_predict(sequence, model):
    y_pred = model.predict(sequence)
    return np.append(sequence, y_pred[-1].reshape(1, 1, 3), axis=0)


def r_predict(sequence, pre_steps, model):
    for i in range(pre_steps):
        sequence = move_predict(sequence, model)
    return sequence


def main():
    configs = json.load(open('..\\configs\\config.json', encoding='utf-8'))
    data = DatasetMaker(configs)
    model = CustomizableModel(configs)
    model.load()
    x_test, y_test = data.get_dataset(data.x_test, data.y_test)
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[2])
    print(x_test.shape, y_test.shape)
    test_s = x_test[0: 1076]
    y_s = y_test[1076: 1081]
    y_seq = r_predict(test_s, 5, model.model)
    y_pred = y_seq[1076: 1081]
    y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[2])
    y_s = data.converter(y_s)
    y_pred = data.converter(y_pred)
    model.show_score(None, y_s, None, y_pred)

    print(y_s)
    print(y_pred)
    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title(configs['picture']['labels'][0])
    ax1.plot(y_s[:, 0], 'r', label='Original data')
    ax1.plot(y_pred[:, 0], 'b--', label='Fitting data')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title(configs['picture']['labels'][1])
    ax2.plot(y_s[:, 1], 'r', label='Original data')
    ax2.plot(y_pred[:, 1], 'b--', label='Fitting data')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title(configs['picture']['labels'][2])
    ax3.plot(y_s[:, 2], 'r', label='Original data')
    ax3.plot(y_pred[:, 2], 'b--', label='Fitting data')

    font = font_manager.FontProperties(fname=configs["picture"]["font"])
    ax3.legend(loc="best", prop=font)
    plt.savefig("..\\saved_pictures\\move_predict")
    plt.show()


if __name__ == "__main__":
    main()
