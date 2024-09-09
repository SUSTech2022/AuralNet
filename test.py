import tensorflow as tf
from tensorflow import keras
import os
import re

from model import *
import time 

class Loss_Function(tf.keras.losses.Loss):
    def __init__(self,name="loss_function"):
        super(Loss_Function,self).__init__(name=name)
        self.loss_BinaryCrossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
    def call(self,y_true, y_pred):
        # loss weight
        loss_weight = [0.2,0.5,0.3]

        sounds = tf.transpose(y_pred[:,:,0])
        azis = tf.transpose(y_pred[:,:,1])
        eles = tf.transpose(y_pred[:,:,2]) 
        sounds_lable = tf.transpose(y_true[:,:,0])
        azis_lable = tf.transpose(y_true[:,:,1])
        eles_lable= tf.transpose(y_true[:,:,2])

        loss_sounds = self.loss_BinaryCrossentropy(sounds_lable, sounds)
        loss_sounds = tf.reduce_sum(loss_sounds)

        count_valid = tf.reduce_sum(sounds_lable,axis=-1)
        count_valid = tf.where(count_valid == 0, tf.ones_like(count_valid), count_valid)
        reciprocal_count_valid = tf.math.reciprocal(count_valid)

        loss_azis = tf.reduce_sum(tf.abs(tf.subtract(azis, azis_lable))*sounds_lable,axis=-1)*reciprocal_count_valid
        loss_azis = tf.reduce_sum(loss_azis)

        loss_eles = tf.reduce_sum(tf.abs(tf.subtract(eles, eles_lable))*sounds_lable,axis=-1)*reciprocal_count_valid
        loss_eles = tf.reduce_sum(loss_eles)

        loss = loss_weight[0]*loss_sounds + loss_weight[1] * loss_azis + loss_weight[2] * loss_eles
        
        return loss
    
class Model_Test():
    def __init__(self, model,test_names):
        self.model = model
        self.test_names = test_names
        self.test_num = len(self.test_names)
    def get_test_result(self):
        y_pred = []
        for i in range(self.test_num):
            y_pred.append(self.model.predict(x_test[i]))

        sounds_accuracys = [0]*self.test_num
        azis_abs_error_means = [0]*self.test_num
        eles_abs_error_means = [0]*self.test_num

        for i in range(self.test_num):
            sounds = y_pred[i][:,:,0]
            sounds_lable  = y_test[i][:,:,0]
            sounds_binary = tf.cast(sounds >= 0.5,dtype=tf.int32)
            correct_predictions = tf.reduce_sum(tf.cast((sounds_binary == sounds_lable),dtype=tf.int32)).numpy()
            correct_predictions_sounds = tf.reduce_sum(tf.cast(((sounds_binary == 1) & (sounds_lable == 1)),dtype=tf.int32)).numpy()

            total_predictions = sounds.shape[0]*sounds.shape[1]
            sounds_accuracys[i] = correct_predictions / total_predictions

            azis = y_pred[i][:,:,1]
            azis_lable = y_test[i][:,:,1]
            azis_abs_error = tf.abs(azis-azis_lable)*45
            azis_abs_error = tf.multiply(azis_abs_error,sounds_lable)
            azis_abs_error = tf.multiply(azis_abs_error,tf.cast(sounds_binary,dtype=tf.float32))
            azis_abs_error_mean = tf.reduce_sum(azis_abs_error)/correct_predictions_sounds
            azis_abs_error_means[i] = azis_abs_error_mean.numpy()

            eles = y_pred[i][:,:,2]
            eles_lable = y_test[i][:,:,2]
            eles_abs_error = tf.abs(eles-eles_lable)*75
            eles_abs_error = tf.multiply(eles_abs_error,sounds_lable)
            eles_abs_error = tf.multiply(eles_abs_error,tf.cast(sounds_binary,dtype=tf.float32))
            eles_abs_error_mean = tf.reduce_sum(eles_abs_error)/correct_predictions_sounds
            eles_abs_error_means[i] = eles_abs_error_mean.numpy()

        for i in range(self.test_num):
            print(f"| {self.test_names[i]} |{sounds_accuracys[i]*100:.2f}% | {azis_abs_error_means[i]:.2f} | {eles_abs_error_means[i]:.2f} |")
        
        print("sounds_accuracys:")
        for i, item in enumerate(sounds_accuracys):
            if i > 0:
                print(" & ", end="")
            print(f"{item*100:.2f}", end="")

        print("\n azis_abs_error_means:")
        for i, item in enumerate(azis_abs_error_means):
            if i > 0:
                print(" & ", end="")
            print(f"{item:.2f}", end="")

        print("\n eles_abs_error_means:")
        for i, item in enumerate(eles_abs_error_means):
            if i > 0:
                print(" & ", end="")
            print(f"{item:.2f}", end="")


USE_GPU_NUM = 0
if __name__ == '__main__':
    test_data_paths = [ "./dataset/test_anechoic_1src.h5",
                        "./dataset/test_20db_1src.h5",
                        "./dataset/test_10db_1src.h5",
                        "./dataset/test_0db_1src.h5",
                        "./dataset/test_anechoic_2src.h5",
                        "./dataset/test_20db_2src.h5",
                        "./dataset/test_10db_2src.h5",
                        "./dataset/test_0db_2src.h5",
                        "./dataset/test_anechoic_3src.h5",
                        "./dataset/test_20db_3src.h5",
                        "./dataset/test_10db_3src.h5",
                        "./dataset/test_0db_3src.h5",]
    x_test = []
    y_test = []
    test_names = [os.path.splitext(os.path.basename(path))[0] for path in test_data_paths]
    for i in range(len(test_data_paths)):
        x_test_i, y_test_i = load_preprocessed_data(save_path=test_data_paths[i])
        x_test.append(x_test_i)
        y_test.append(y_test_i)

    # check GPU
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # cpu_devices = tf.config.experimental.list_physical_devices('CPU')
    print("Num GPUs Available: ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[USE_GPU_NUM], True)
    tf.config.set_visible_devices(physical_devices[USE_GPU_NUM], 'GPU')

    deepear_model = AuralNet()
    model = deepear_model.build_model()

    loss_function = Loss_Function()
    model.compile(loss=loss_function)

    model.load_weights(os.path.join("./model_save", "MCT_weights.h5"))

    model_test = Model_Test(model=model,test_names=test_names)
    model_test.get_test_result()


