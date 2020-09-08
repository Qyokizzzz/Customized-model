__author__ = "Ldaze"

import json
import time
import os
import datetime as dt
from core.custom_early_stopping import EarlyStoppingByLossVal
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


def timer(func):
    def inside(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        stop_time = time.time()
        print('The %s function run time is %s second(s)' % (func.__name__, (stop_time - start_time)))
        return res
    return inside


def transpose_tensor(data):
    temp = data.reshape(data.shape[0], data.shape[2], data.shape[1])
    for i in range(data.shape[0]):
        temp[i] = data[i].T
    return temp


class Utils(object):
    def __init__(self, configs):
        self.configs = configs
        self.filename = self.get_filename()

    def read_dict_into_list(self, dic, lis):
        for i in dic.keys():
            if type(dic[i]) == dict:
                self.read_dict_into_list(dic[i], lis)
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

    def convert_list_to_cmd(self, layers):
        command = []
        for layer in layers:
            tmp = []
            self.read_dict_into_list(layer, tmp)
            command.append(''.join('%s' % item for item in tmp))
        return command

    def get_filename(self):
        filename = '%s-ts%s-id%s-ps%s-od%s.h5' % (
            dt.datetime.now().strftime('%Y%m%d-%H%M%S'),
            self.configs['data']['time_steps'],
            len(self.configs['data']['X_columns']),
            self.configs['data']['pre_steps'],
            len(self.configs['data']['y_columns'])
        )

        return filename

    @property
    def early_stopping(self):
        if self.configs['training']['stop_type'] == 'customized':
            early_stopping = EarlyStoppingByLossVal(
                monitor=self.configs['training']['early_stopping']['monitor'],
                value=self.configs['training']['early_stopping']['value'],
                verbose=self.configs['training']['early_stopping']['verbose'],
            )

        else:
            early_stopping = EarlyStopping(
                monitor=self.configs['training']['early_stopping']['monitor'],
                patience=self.configs['training']['early_stopping']['patience'],
                mode=self.configs['training']['early_stopping']['mode']
            )
        return early_stopping

    @staticmethod
    def modify_load(filename, new_dir):

        with open(filename, "r+", encoding='utf-8') as jsonFile:
            data = json.load(jsonFile)
            data["model"]["load_path"] = new_dir
            jsonFile.seek(0)
            json.dump(data, jsonFile, ensure_ascii=False, indent=4)
            jsonFile.truncate()

    @staticmethod
    def save_path(save_dir, f_name):
        save_path = os.path.join(
            save_dir,
            f_name
        )
        return save_path

    @staticmethod
    def adj_r2_score(y, y_pred):
        #  貌似没什么用，结果没区别
        n = len(y)
        f = y.shape[-1]
        return 1 - ((1 - r2_score(y, y_pred)) * (n - 1)) / (n - f - 1)

    @staticmethod
    def show_r2_score(y_train, y_test, train_predict, test_predict):
        if y_train is None:
            print("The R2 score on the Test set is:{:0.3f}".format(r2_score(y_test, test_predict)))
        elif y_test is None:
            print("The R2 score on the Train set is:{:0.3f}".format(r2_score(y_train, train_predict)))
        else:
            print("The R2 score on the Train set is:{:0.3f}".format(r2_score(y_train, train_predict)))
            print("The R2 score on the Test set is:{:0.3f}".format(r2_score(y_test, test_predict)))

    @staticmethod
    def show_mse_score(y_train, y_test, train_predict, test_predict):
        if y_train is None:
            print("The MSE score on the Test set is:{:0.3f}".format(mean_squared_error(y_test, test_predict)))
        elif y_test is None:
            print("The MSE score on the Train set is:{:0.3f}".format(mean_squared_error(y_train, train_predict)))
        else:
            print("The MSE score on the Train set is:{:0.3f}".format(mean_squared_error(y_train, train_predict)))
            print("The MSE score on the Test set is:{:0.3f}".format(mean_squared_error(y_test, test_predict)))
