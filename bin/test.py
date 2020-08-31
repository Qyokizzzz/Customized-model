__author__ = 'Ldaze'

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers \
    import LSTM, Dense, Activation, Dropout, Bidirectional,\
    Flatten, TimeDistributed, RepeatVector, Conv1D, Permute, Multiply, Lambda, Input

import json
configs = json.load(open('..\\configs\\config.json', encoding='utf-8'))


def read_dict_into_list(dic, lis):
    for i in dic.keys():
        if type(dic[i]) == dict:
            read_dict_into_list(dic[i], lis)
        else:
            if i == 'type':
                lis.append(dic[i])
                lis.append('(')
            else:
                lis.append(i)
                lis.append('=')
                lis.append(dic[i])
                lis.append(',')
    temp = lis.pop()
    if temp != ',':
        lis.append(temp)
    lis.append(')')


def test():
    command = []
    inputs = Input(shape=(configs['data']['time_steps'], configs['data']['pre_steps']))
    for layer in configs['model']['layers']:
        tmp = []
        read_dict_into_list(layer, tmp)
        command.append(''.join('%s' % item for item in tmp))
    return command


if __name__ == "__main__":

    inputs = Input(shape=(1, 3))
    a = LSTM(120)(inputs)
    a = Dropout(0.2)(a)
    a = RepeatVector(10)(a)
    a = LSTM(360, return_sequences=True)(a)
    a = Dropout(0.2)(a)
    outputs = TimeDistributed(Dense(3))(a)
    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(loss='mse', optimizer='adam')

