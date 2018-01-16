# -*- coding: utf_8 -*-
''' Main Functions ''' 

#----------------------------------------------------------
# Imports
#----------------------------------------------------------
from ex_model import ModelTeacher, ModelStudent
from ex_dataset import DatasetMNIST

NUM_EPOCH = 3
NUM_BATCH_SIZE = 32

#----------------------------------------------------------
# Functions
#----------------------------------------------------------
def training():
    
    # データ取得
    #--------------------------------------
    dataset = DatasetMNIST()
    dataset.create_conv()

    # 教師モデルをハードターゲットで訓練
    #--------------------------------------
    print("Teacher model with Hard target")
    teacher = ModelTeacher()
    teacher.set_dataset(dataset)
    teacher.create()
    teacher.show()
    teacher.fit(batch_size=NUM_BATCH_SIZE, epochs=NUM_EPOCH)
    teacher.result()
    teacher.evaluate()

    # 生徒モデルをハードターゲットで訓練
    #--------------------------------------
    print("Student model with Hard target")
    student = ModelStudent()
    student.set_dataset(dataset)
    student.create()
    student.show()
    student.fit(batch_size=NUM_BATCH_SIZE, epochs=NUM_EPOCH)
    student.result()
    student.evaluate()


    # 生徒モデルをソフトターゲットで訓練
    #--------------------------------------
    print("Student model with Soft target")
    student2 = ModelStudent()
    student2.set_dataset(dataset)
    student2.create()

    # 教師モデルのソフトマックス層を温度付きに変更
    teacher.set_softmax_temperature(10)
    student2.set_softmax_temperature(10)

    # 教師モデルの温度付き出力を教師として生徒モデルを訓練
    student2.set_teacher(teacher)
    student2.fit_with_teacher(batch_size=NUM_BATCH_SIZE, epochs=NUM_EPOCH)
    student2.result()

    # 評価前に温度を1に設定
    student2.set_softmax_temperature(1)
    student2.evaluate()

#----------------------------------------------------------
# 実行スクリプト
#----------------------------------------------------------
if __name__ == '__main__':
    training()
