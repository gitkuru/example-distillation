# -*- coding: utf_8 -*-
''' Main Functions ''' 

#----------------------------------------------------------
# Imports
#----------------------------------------------------------
from ex_model import ModelBasic
from ex_dataset import DatasetMNIST

from keras.layers import Dense
from sklearn.utils.extmath import randomized_svd

NUM_EPOCH = 3
NUM_BATCH_SIZE = 32

#----------------------------------------------------------
# Functions
#----------------------------------------------------------
def factorize(W: np.ndarray, k: int):
    """
    低ランク近似による行列分解
    W: (MxN)
    """

    # 特異値分解
    u, s, v = randomized_svd(W, k)

    # W = u x s x v
    # 二つの行列U:(MxK) V:(KxV)の積で表現するため
    # U=u x sqrt(s)  V=sqrt(s) x v となる行列を算出する

    # 対角行列を取得
    scale = np.diag(np.sqrt(s))
    U = u.dot(scale).astype(W.dtype)
    V = scale.dot(v).astype(W.dtype)
    return U, V



def training():
    
    # データ取得
    #--------------------------------------
    dataset = DatasetMNIST()
    dataset.create()

    # モデルを訓練
    #--------------------------------------
    print("training start")
    model = ModelBasic()
    model.set_dataset(dataset)
    model.create()
    model.show()
    model.fit(batch_size=NUM_BATCH_SIZE, epochs=NUM_EPOCH)
    model.result()
    model.evaluate()

    # モデルの行列パラメータを低ランク近似
    #---------------------------------------


#----------------------------------------------------------
# 実行スクリプト
#----------------------------------------------------------
if __name__ == '__main__':
    training()
