import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import argparse
from utils import loss_graph, shuffle_data, rotate_point_cloud, data_loader
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
TRAIN_FILES = [
"data/modelnet40_ply_hdf5_2048/ply_data_train0.h5",
"data/modelnet40_ply_hdf5_2048/ply_data_train1.h5",
"data/modelnet40_ply_hdf5_2048/ply_data_train2.h5",
"data/modelnet40_ply_hdf5_2048/ply_data_train3.h5",
# "data/modelnet40_ply_hdf5_2048/ply_data_train4.h5"
]

TEST_FILES = [
"data/modelnet40_ply_hdf5_2048/ply_data_test0.h5",
"data/modelnet40_ply_hdf5_2048/ply_data_test1.h5"
]

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--max_iteration', type=int, default=100, help='Iteration to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--dropout_rate', type=float, default=0.7, help='Dropout_rate [default: 0.7]')
args = parser.parse_args()
batch_size = args.batch_size
# 1번 cuda device 사용 : Use first cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"



# pointcloud 모델 신경망 : The pointcloud neural network architecture
class network(tf.keras.Model):
    def __init__(self):
        super().__init__()
        K=3
        self.input_conv1 = tf.keras.layers.Conv2D(64, (1, 3), (1, 1), padding='valid', activation='relu')
        self.input_conv2 = tf.keras.layers.Conv2D(128, (1, 1), (1, 1), padding='valid', activation='relu')
        self.input_conv3 = tf.keras.layers.Conv2D(1024, (1, 1), (1, 1), padding='valid', activation='relu')
        self.input_maxpooling2d = tf.keras.layers.MaxPooling2D(pool_size=(1024, 1), padding='valid')
        self.input_flatten = tf.keras.layers.Flatten()
        self.input_dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.input_dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.input_dense3 = tf.keras.layers.Dense(3*K, activation=None)
        self.model_1_conv1 = tf.keras.layers.Conv2D(64, (1, 3), (1, 1), padding='valid', activation='relu')
        self.model_1_conv2 = tf.keras.layers.Conv2D(64, (1, 1), (1, 1), padding='valid', activation='relu')
        K = 64
        self.feature_conv1 = tf.keras.layers.Conv2D(64, (1, 1), (1, 1), padding='valid', activation='relu')
        self.feature_conv2 = tf.keras.layers.Conv2D(128, (1, 1), (1, 1), padding='valid', activation='relu')
        self.feature_conv3 = tf.keras.layers.Conv2D(1024, (1, 1), (1, 1), padding='valid', activation='relu')
        self.feature_maxpooling2d = tf.keras.layers.MaxPooling2D(pool_size=(1024, 1), padding='valid')
        self.feature_flatten = tf.keras.layers.Flatten()
        self.feature_dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.feature_dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.feature_dense3 = tf.keras.layers.Dense(K*K, activation=None)
        self.model_2_conv1 = tf.keras.layers.Conv2D(64, (1, 1), (1, 1), padding='valid', activation='relu')
        self.model_2_conv2 = tf.keras.layers.Conv2D(128, (1, 1), (1, 1), padding='valid', activation='relu')
        self.model_2_conv3 = tf.keras.layers.Conv2D(1024, (1, 1), (1, 1), padding='valid', activation='relu')
        self.model_2_maxpooling2d = tf.keras.layers.MaxPooling2D(pool_size=(1024, 1), padding='valid')
        self.model_2_flatten = tf.keras.layers.Flatten()
        self.model_2_dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.model_2_dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.model_2_dense3 = tf.keras.layers.Dense(40, activation=None)
        self.dropout = tf.keras.layers.Dropout(args.dropout_rate)
    def call(self, inputs):
        y = self.input_conv1(inputs)
        y = self.input_conv2(y)
        y = self.input_conv3(y)
        y = self.input_maxpooling2d(y)
        y = self.input_flatten(y)
        y = self.input_dense1(y)
        y = self.input_dense2(y)
        y = self.input_dense3(y)
        transform = tf.reshape(y, (-1, 3, 3))
        x = tf.matmul(tf.squeeze(inputs, axis=[3]), transform)
        x = tf.expand_dims(x, -1)
        x = self.model_1_conv1(x)
        x = self.model_1_conv2(x)
        y = self.feature_conv1(x)
        y = self.feature_conv2(y)
        y = self.feature_conv3(y)
        y = self.feature_maxpooling2d(y)
        y = self.feature_flatten(y)
        y = self.feature_dense1(y)
        y = self.feature_dense2(y)
        y = self.feature_dense3(y)
        transform = tf.reshape(y, (-1, 64, 64))
        x = tf.matmul(tf.squeeze(x, axis=[2]), transform)
        x = tf.expand_dims(x, [2])
        x = self.model_2_conv1(x)
        x = self.model_2_conv2(x)
        x = self.model_2_conv3(x)
        x = self.model_2_maxpooling2d(x)
        x = self.model_2_flatten(x)
        x = self.model_2_dense1(x)
        x = self.model_2_dense2(x)
        x = self.model_2_dense3(x)
        return x
# 모델 구조 확인 : Building to see model construction
model = network()
model.build(input_shape=(None, 2048, 3, 1))
model.summary()
# 실험에 의해 확인한 learning_rate 0.0001 : The learning rate 0.0001 is chosen by experiments
optimizer = Adam(learning_rate=args.learning_rate)

def train(max_iteration):
    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []
    model_params = model.trainable_variables
    
    for iteration in range(max_iteration):
        total_loss = []
        total_accuracy = []
        
        num_batches = (2048 * 4) // batch_size
        with tqdm(total=num_batches, desc=f'iteration {iteration + 1}', unit='batch') as pbar:
            for data in TRAIN_FILES:
                data_point_cloud, data_labels = data_loader(data)
                for i in range(0, len(data_point_cloud), batch_size):
                    point_cloud_batch = data_point_cloud[i:i+batch_size]
                    labels_batch = data_labels[i:i+batch_size]
                    # 데이터 순서 셔플
                    shuffled_point_cloud_batch, shuffled_labels_batch = shuffle_data(point_cloud_batch, labels_batch)
                    # 데이터 회전
                    rotated_point_cloud_batch = rotate_point_cloud(shuffled_point_cloud_batch)
                    rotated_point_cloud_batch = np.expand_dims(rotated_point_cloud_batch, -1)
                    with tf.GradientTape() as tape:
                        tape.watch(model_params)
                        y = model(rotated_point_cloud_batch)
                        labels = tf.one_hot(shuffled_labels_batch, 40)
                        loss = tf.reduce_mean(tf.square(tf.squeeze(labels) - y))
                        y_pred = tf.argmax(y, axis=1)
                        labels_pred = tf.argmax(tf.squeeze(labels), axis=1)
                        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_pred, y_pred), dtype=tf.float32))
                    grads = tape.gradient(loss, model_params)
                    optimizer.apply_gradients(zip(grads, model_params))
                    pbar.update(1)
                    pbar.set_postfix({'Loss': loss.numpy()})
                    total_loss.append(loss)
                    total_accuracy.append(accuracy)
                validation_loss = []
                validation_accuracy = []
            for data in TEST_FILES:
                data_point_cloud, data_labels = data_loader(data)
                for i in range(0, len(data_point_cloud), batch_size):
                    point_cloud_batch = data_point_cloud[i:i+batch_size]
                    labels_batch = data_labels[i:i+batch_size]
                    # 데이터 순서 셔플
                    shuffled_point_cloud_batch, shuffled_labels_batch = shuffle_data(point_cloud_batch, labels_batch)
                    # 데이터 회전
                    rotated_point_cloud_batch = rotate_point_cloud(shuffled_point_cloud_batch)
                    rotated_point_cloud_batch = np.expand_dims(rotated_point_cloud_batch, -1)
                    y = model(rotated_point_cloud_batch)
                    labels = tf.one_hot(shuffled_labels_batch, 40)
                    loss = tf.reduce_mean(tf.square(tf.squeeze(labels) - y))
                    y_pred = tf.argmax(y, axis=1)
                    labels_pred = tf.argmax(tf.squeeze(labels), axis=1)
                    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_pred, y_pred), dtype=tf.float32))
                    validation_loss.append(loss)
                    validation_accuracy.append(accuracy)
            iteration_validation_loss = np.mean(np.array(validation_loss))
            iteration_validation_accuracy = np.mean(np.array(validation_accuracy))
            iteration_loss = np.mean(np.array(total_loss))
            iteration_accuracy = np.mean(np.array(total_accuracy))
            print(f'Iteration {iteration + 1}, Loss: {iteration_loss:.5f}, Validation Loss: {iteration_validation_loss:.5f}, Accuracy: {iteration_accuracy:.5f}, Validation Accuracy: {iteration_validation_accuracy:.5f}')
            
            train_losses.append(iteration_loss)
            validation_losses.append(iteration_validation_loss)

            train_accuracies.append(iteration_accuracy)
            validation_accuracies.append(iteration_validation_accuracy)
    return train_losses, validation_losses, train_accuracies, validation_accuracies
max_iteration = args.max_iteration
train_losses, validation_losses, train_accuracies, validation_accuracies = train(max_iteration)
# 훈련 곡선 : training curve

loss_graph(train_losses, validation_losses, train_accuracies, validation_accuracies)
