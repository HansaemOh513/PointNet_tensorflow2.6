import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
# 1번 cuda device 사용 : Use first cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
file_path = './data/modelnet40_ply_hdf5_2048/ply_data_train0.h5'
with h5py.File(file_path, 'r') as file:
    label = file['label']
    data_dataset = file['data']
    data_labels  = label[:]
    data_values = data_dataset[:]
# tensorflow 모델 투입을 위해서 차원확장 : Expandng demension to put in a tensorflow model 
point_cloud = np.expand_dims(data_values, -1)
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
# 실험에 의해 확인한 learning_rate : The learning rate is chosen by experiments
optimizer = Adam(learning_rate=0.00003)

def train(max_iteration):
    losses = []
    accuracies = []
    model_params = model.trainable_variables
    num_batches = len(point_cloud) // 32
    for iteration in range(max_iteration):
        total_loss = []
        total_accuracy = []
        with tqdm(total=num_batches, desc=f'iteration {iteration + 1}', unit='batch') as pbar:
            for i in range(0, len(point_cloud), 32):
                # batch size를 32로 조정 : The batch size of data is set as 32
                batch_size=32
                point_cloud_batch = point_cloud[i:i+batch_size]
                with tf.GradientTape() as tape:
                    tape.watch(model_params)
                    y = model(point_cloud_batch)
                    
                    labels = tf.one_hot([data_labels[i:i+batch_size]], 40)
                    loss = tf.reduce_mean(tf.square(tf.squeeze(labels, axis=[0, 2]) - y))
                    y_pred = tf.argmax(y, axis=1)
                    labels_pred = tf.argmax(tf.squeeze(labels, axis=[0, 2]), axis=1)
                    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_pred, y_pred), dtype=tf.float32))
                grads = tape.gradient(loss, model_params)
                optimizer.apply_gradients(zip(grads, model_params))
                pbar.update(1)
                pbar.set_postfix({'Loss': loss.numpy()})

                total_loss.append(loss)
                total_accuracy.append(accuracy)
        
        iteration_loss = np.mean(np.array(total_loss))
        iteration_accuracy = np.mean(np.array(total_accuracy))
        print(f'Iteration {iteration + 1}, Loss: {iteration_loss:.5f}, Accuracy: {iteration_accuracy:.5f}')
        losses.append(iteration_loss)
        accuracies.append(iteration_accuracy)
    return losses, accuracies
max_iteration = 1000
losses, accuracies = train(max_iteration)
# 훈련 곡선 : training curve
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracies, color='orange', label='Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
