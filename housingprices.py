import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import numpy as np

batch_size = 20
epochs = 200


dataset = pd.read_csv("data.csv")
X = dataset.iloc[:,2:14].values
y = dataset.iloc[:,1].values

def normalise(value,values):
    m = max(values)
    return value/m

for i in range(len(X)):
    index = 2
    for j in range(len(X[i])):
        X[i][j]=normalise(X[i][j],dataset.iloc[:,index])
        index+=1


input = tf.placeholder(tf.float32,[None,12])
target = tf.placeholder(tf.float32,[None,1])

W1 = tf.Variable(tf.random.normal([12,20]))
b1 = tf.Variable(tf.random.normal([20]))
layer1 = tf.add(tf.matmul(input,W1),b1)
layer1 = tf.nn.leaky_relu(layer1)

W2 = tf.Variable(tf.random.normal([20,20]))
b2 = tf.Variable(tf.random.normal([20]))
layer2 = tf.add(tf.matmul(layer1,W2),b2)
layer2 = tf.nn.leaky_relu(layer2)

W3 = tf.Variable(tf.random.normal([20,20]))
b3 = tf.Variable(tf.random.normal([20]))
layer3 = tf.add(tf.matmul(layer2,W3),b3)
layer3 = tf.nn.leaky_relu(layer3)

W4 = tf.Variable(tf.random.normal([20,20]))
b4 = tf.Variable(tf.random.normal([20]))
layer4 = tf.add(tf.matmul(layer3,W4),b4)
layer4 = tf.nn.leaky_relu(layer4)

W5 = tf.Variable(tf.random.normal([20,20]))
b5 = tf.Variable(tf.random.normal([20]))
layer5 = tf.add(tf.matmul(layer4,W5),b5)
layer5 = tf.nn.leaky_relu(layer5)

W6 = tf.Variable(tf.random.normal([20,20]))
b6 = tf.Variable(tf.random.normal([20]))
layer6 = tf.add(tf.matmul(layer5,W6),b6)
layer6 = tf.nn.leaky_relu(layer6)

W7 = tf.Variable(tf.random.normal([20,20]))
b7 = tf.Variable(tf.random.normal([20]))
layer7 = tf.add(tf.matmul(layer6,W7),b7)
layer7 = tf.nn.leaky_relu(layer7)

W8 = tf.Variable(tf.random.normal([20,20]))
b8 = tf.Variable(tf.random.normal([20]))
layer8 = tf.add(tf.matmul(layer7,W8),b8)
layer8 = tf.nn.leaky_relu(layer8)

W9 = tf.Variable(tf.random.normal([20,20]))
b9 = tf.Variable(tf.random.normal([20]))
layer9 = tf.add(tf.matmul(layer8,W9),b9)
layer9 = tf.nn.leaky_relu(layer9)

W10= tf.Variable(tf.random.normal([20,1]))
b10 = tf.Variable(tf.random.normal([1]))
output = tf.add(tf.matmul(layer9,W10),b10)

loss = tf.losses.mean_squared_error(predictions=output,labels = target)
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:

    batches = len(X) / batch_size
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(range(int(batches))):
            batchY = np.reshape(y[batch_size*batch:batch_size*batch+batch_size],(batch_size,-1))
            batchX = X[batch_size*batch:batch_size*batch+batch_size]
            _loss,_ = sess.run([loss,optimizer],feed_dict={input:batchX,target:batchY})
            total_loss+=_loss

        print("Epoch: {0}, loss: {1}".format(epoch,total_loss/batches))

    saver.save(sess,"housing_price_model/model")
