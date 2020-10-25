__author__ = "Ldaze"

import numpy as np
import os

from core.utils import timer
from core.utils import Utils

from core.custom_layer import AttentionOnDims
from core.custom_layer import AttentionOnSteps
from tensorflow.keras import utils as ut
from tensorflow.keras import backend as bd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import BatchNormalization


class CustomizableModel(Utils):
    def __init__(self, configs):
        self.model = None
        super(CustomizableModel, self).__init__(configs)

    @timer
    def build(self):
        if self.configs['model']['need_batch']:
            inputs = Input(
                batch_input_shape=(
                    self.configs['training']['batch_size'],
                    self.configs['data']['time_steps'],
                    len(self.configs['data']['X_columns'])
                )
            )
        else:
            inputs = Input(shape=(self.configs['data']['time_steps'], len(self.configs['data']['X_columns'])))

        command = self.convert_list_to_cmd(self.configs['model']['layers'])
        # print(command)
        tmp = eval(command[0])(inputs)
        for i in range(1, len(command)):
            # print(command[i])
            tmp = eval(command[i])(tmp)

        self.model = Model(inputs=inputs, outputs=tmp)
        self.model.compile(loss=self.configs['model']['loss'], optimizer=self.configs['model']['optimizer'])

        print("model compiled")
        if self.configs['model']['save_structure']:
            self.__save_structure()

    def load(self):
        self.model = load_model(
            self.configs['model']['load_path'],
            custom_objects={'AttentionOnDims': AttentionOnDims, 'AttentionOnSteps': AttentionOnSteps}
        )

    def __save_structure(self):
        str_list = self.filename.split('\\')
        filename = str_list[-1].split('.')[0]
        filename = filename+'_structure.png'
        ut.plot_model(
            self.model,
            to_file=self.save_path(self.configs['model']['structure_dir'], filename),
            show_shapes=True,
            show_layer_names=True
        )

    @timer
    def train(self, x_train, y_train, x_test, y_test):
        callbacks = [self.early_stopping] if self.configs['training']['early_stopping']['is_enable'] else None

        self.model.fit(
            x_train,
            y_train,
            epochs=self.configs['training']['epochs'],
            batch_size=self.configs['training']['batch_size'],
            verbose=self.configs['training']['verbose'],
            shuffle=self.configs['training']['shuffle'],
            callbacks=callbacks,
            validation_data=(x_test, y_test)
        )
        save_path = self.save_path(self.configs['model']['save_dir'], self.filename)
        self.model.save(save_path)
        print('Training Completed. Model saved as %s' % save_path)

        if self.configs['training']['load_config']:
            load_path = os.path.join('..\\configs', self.configs['config']['filename'])
            self.modify_load(load_path, save_path)
            print('Loaded in config')

    def predict(self, x):
        return self.model.predict(x)

    def predict_multiple(self, sequence, start, pre_steps):
        for i in range(pre_steps):
            y_pred = self.model.predict(sequence)
            sequence = np.append(sequence, y_pred[-1].reshape(1, 1, 3), axis=0)

        tmp = sequence[start: start + pre_steps]
        return tmp.reshape(tmp.shape[0], tmp.shape[2])
