# -*- coding: utf_8 -*-
''' Create Model ''' 

#----------------------------------------------------------
# Imports
#----------------------------------------------------------
from keras.models   import Model, Sequential
from keras.layers   import Input, Add, Dense, Dropout, Flatten, Activation, Lambda
from keras.layers   import Conv2D, MaxPooling2D
from keras.utils    import np_utils

from ex_dataset     import DatasetBase
from ex_util        import Util

import numpy as np

#----------------------------------------------------------
# Classes
#----------------------------------------------------------
class AbstractModel:
    def __init__(self):
        self.model = None   # モデル
        self.history = None # 学習履歴
        self.dataset = None # データセット

    def create_base(self):
        raise NotImplementedError

    def create(self):
        self.create_base()
        self.compile()

    def set_dataset(self, dataset):
        self.dataset = dataset
    
    def fit(self, batch_size = 32, epochs = 1):
        history = self.model.fit(
            self.dataset.x_train,
            self.dataset.y_train,
            batch_size=batch_size, 
            epochs=epochs,
            verbose=1, 
            validation_data=(self.dataset.x_test, self.dataset.y_test))
        self.history = history

    def show(self):
        self.model.summary()


    def result(self):
        if self.history != None:
            Util.plot_result(self.history)
    
    def evaluate(self):
        score = self.model.evaluate(self.dataset.x_test, self.dataset.y_test, verbose=0)
        print(score)


class AbstractModelDistillation(AbstractModel):

    def __init__(self):
        super().__init__()

    def compile(self):
        self.model.compile(
            optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])
  
    def set_softmax_temperature(self, temperature):
        ''' 温度付きソフトマックス層を追加する '''
        
        # softmax層を削除する
        self.model.layers.pop()
        
        # softed probabilities
        logits   = self.model.layers[-1].output
        logits_t = Lambda(lambda x: x/temperature)(logits)
        prob_t   = Activation('softmax')(logits_t)
        
        self.model = Model(self.model.input, prob_t)
        self.compile()

class ModelBasic(AbstractModel):

    def __init__(self):
        super().__init__()

    def create_base(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.dataset.input_shape))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.dataset.output_shape))
        model.add(Activation('softmax'))
        self.model = model


class ModelTeacher(AbstractModelDistillation):
    ''' 教師モデル'''

    def __init__(self):
        super().__init__()

    def create_base(self):
        '''教師モデルを生成'''
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=self.dataset.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        
        self.model = model


class ModelStudent(AbstractModelDistillation):
    ''' 生徒モデル'''

    def __init__(self):
        super().__init__()
        self.diff_model = None

    def create_base(self):
        ''' 生徒モデルを作成 '''
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=self.dataset.input_shape))
        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation('softmax'))

        self.model = model

    def set_teacher(self, teacher):

        for i in range(len(teacher.model.layers)):
            setattr(teacher.model.layers[i], 'trainable', False)

        # 教師モデルの出力層と生徒モデルの出力層の差分レイヤを作成
        layer = Activation(lambda x: -x)(self.model.output) 
        diff = Add()([teacher.model.output, layer])
        
        # 教師モデルの入力層、生徒モデルの入力層 → 差分レイヤ出力層となるモデルを作成
        self.diff_model = Model(inputs=[teacher.model.input, self.model.input], outputs=[diff])
        self.diff_model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['acc'])


    def fit_with_teacher(self, batch_size = 32, epochs = 1):

        # 生徒モデルの出力の期待値は教師モデルの出力(ソフトターゲット)のため
        # 差分を0に近づけるように訓練
        diff_y_train = np.zeros(self.dataset.y_train.shape)

        self.diff_model.fit(
            [self.dataset.x_train, self.dataset.x_train],
            [diff_y_train], 
            batch_size=batch_size,
            epochs=epochs,
            verbose=1)

    