# -*- coding: utf_8 -*-
''' データセットの取得、整形 ''' 

#----------------------------------------------------------
# Imports
#----------------------------------------------------------
from keras          import backend as K
from keras.datasets import mnist, fashion_mnist
from keras.utils    import np_utils, to_categorical


#----------------------------------------------------------
# classes
#----------------------------------------------------------
class DatasetBase:
    def __init__(self):
        self.x_train = None       # 訓練用 入力データ
        self.y_train = None       # 訓練用 正解データ
        self.x_test = None        # 評価用 入力データ
        self.y_test = None        # 評価用 正解データ
        self.input_shape = None   # 入力データの形式
        self.output_shape = None  # 出力データの形式


class AbstractDatasetMNIST(DatasetBase):
    def __init__(self):
        super().__init__()

    def __adjust(self):
        # 画像を1次元配列化
        self.x_train = self.x_train.reshape(60000, 784)
        self.x_test  = self.x_test.reshape(10000, 784)
        self.input_shape = 784
        
        # 0.0-1.0の範囲に変換
        self.x_train = self.x_train.astype('float32')
        self.x_test  = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test  /= 255
        
        # one-hot-encoding
        self.y_train = np_utils.to_categorical(self.y_train, 10)
        self.y_test  = np_utils.to_categorical(self.y_test, 10)
        self.output_shape = 10
    
    def __adjust_conv(self):
        img_rows = 28
        img_cols = 28
        if K.image_data_format() == 'channels_first':
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, img_rows, img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, img_rows, img_cols)
            self.input_shape = (1, img_rows, img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], img_rows, img_cols, 1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], img_rows, img_cols, 1)
            self.input_shape = (img_rows, img_cols, 1)

        # 0.0-1.0の範囲に変換
        self.x_train = self.x_train.astype('float32')
        self.x_test  = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test  /= 255
        
        # one-hot-encoding
        self.y_train = np_utils.to_categorical(self.y_train, 10)
        self.y_test  = np_utils.to_categorical(self.y_test, 10)
        self.output_shape = 10

    def load_data(self):
        raise NotImplementedError
    
    def create(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()
        self.__adjust()
    
    def create_conv(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_data()
        self.__adjust_conv()



class DatasetMNIST(AbstractDatasetMNIST):

    def __init__(self):
        super().__init__()
    
    def load_data(self):
        return mnist.load_data()


class DatasetFashionMNIST(AbstractDatasetMNIST):

    def __init__(self):
        super().__init__()
    
    def load_data(self):
        return fashion_mnist.load_data()
