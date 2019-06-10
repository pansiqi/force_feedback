import tensorflow as tf
import glob
import os
from skimage import io
import numpy as np
import time

#  训练数据集
#  训练数据集的地址

path = '..\\DATA\\img\\'
model_path = 'model\\model.ckpt'


def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.png'):
            print('reading the images:%s' % (im))
            img = io.imread(im)
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


data, label = read_img(path)
data = data[:,:,:,np.newaxis]
print(data.shape)
print(label)

# 打乱数据顺序
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]

#  将数据集分成传说中的28定律
ratio = 0.8
s = np.int(num_example * ratio)
x_train = data[:s]
y_train = label[:s]
x_val = data[s:]
y_val = label[s:]

w = 16
h = 16
c = 1

#  ---------------------创建网络层-------------------------------------
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')


def inference(input_tensor, train, regularizer):
    # -----------------------第一层----------------------------
    with tf.variable_scope('layer1-conv1'):
        # 初始化权重conv1_weights为可保存变量，大小为5x5,3个通道（RGB），数量为32个
        conv1_weights = tf.get_variable("weight", [5, 5, 1, 32],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 初始化偏置conv1_biases，数量为32个
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        # 卷积计算，tf.nn.conv2d为tensorflow自带2维卷积函数，input_tensor为输入数据，
        # conv1_weights为权重，strides=[1, 1, 1, 1]表示左右上下滑动步长为1，padding='SAME'表示输入和输出大小一样，即补0
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        # 激励计算，调用tensorflow的relu函数
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        # 池化计算，调用tensorflow的max_pool函数，strides=[1,2,2,1]，表示池化边界，2个对一个生成，padding="VALID"表示不操作。
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    # -----------------------第二层----------------------------
    with tf.variable_scope("layer3-conv2"):
        # 同上，不过参数的有变化，根据卷积计算和通道数量的变化，设置对应的参数
        conv2_weights = tf.get_variable("weight", [5, 5, 32, 64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        nodes = 4 * 4 * 64
        reshaped = tf.reshape(pool2, [-1, nodes])
    # # -----------------------第五层---------------------------
    with tf.variable_scope('layer5-fc1'):
        # 初始化全连接层的参数，隐含节点为1024个
        fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))  # 正则化矩阵
        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))
        # 使用relu函数作为激活函数
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        # 采用dropout层，减少过拟合和欠拟合的程度，保存模型最好的预测效率
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
    # -----------------------第六层----------------------------
    with tf.variable_scope('layer6-fc2'):
        # 同上，不过参数的有变化，根据卷积计算和通道数量的变化，设置对应的参数
        fc2_weights = tf.get_variable("weight", [1024, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)
    # -----------------------第七层----------------------------
    with tf.variable_scope('layer11-fc3'):
        # 同上，不过参数的有变化，根据卷积计算和通道数量的变化，设置对应的参数
        fc3_weights = tf.get_variable("weight", [512, 5],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [5], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases  # matmul矩阵相乘
    # 返回最后的计算结果
    return logit


# ---------------------------网络结束---------------------------
# 设置正则化参数为0.0001
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
# 将上述构建网络结构引入
logits = inference(x, False, regularizer)

# (小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1, dtype=tf.float32)
logits_eval = tf.multiply(logits, b, name='logits_eval')  # b为1

# 设置损失函数，作为模型训练优化的参考标准，loss越小，模型越优
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
# 设置整体学习率为α为0.001
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
# 设置预测精度
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# 训练和测试数据，可将n_epoch设置更大一些


# 迭代次数
n_epoch = 1000
# 每次迭代输入的图片数据
batch_size = 1
saver = tf.train.Saver(max_to_keep=1)  # 可以指定保存的模型个数，利用max_to_keep=4，则最终会保存4个模型（
with tf.Session() as sess:
    # 初始化全局参数
    sess.run(tf.global_variables_initializer())
    # 开始迭代训练，调用的都是前面设置好的函数或变量
    for epoch in range(n_epoch):
        start_time = time.time()

        # training#训练集
        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err
            train_acc += ac
            n_batch += 1
            # print("   train loss: %f" % (np.sum(train_loss) / n_batch))
            # print("   train acc: %f" % (np.sum(train_acc) / n_batch))
            print("Train Epoch:", '%02d' % (epoch + 1),
                  "Loss=", "{:.9f}".format(np.sum(train_loss) / n_batch), " Accuracy=", (np.sum(train_acc) / n_batch))
        # validation#验证集
        val_loss, val_acc, n_batch = 0, 0, 0
        # for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x: x_val, y_: y_val})
        val_loss += err
        val_acc += ac
        n_batch += 1
        # print("   validation loss: %f" % (np.sum(val_loss)))
        # print("   validation acc: %f" % (np.sum(val_acc)))
        # 保存模型及模型参数
    saver.save(sess, model_path)
