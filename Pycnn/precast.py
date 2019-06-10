import numpy as np
import tensorflow as tf
from skimage import io

testpath1 = "..\\DATA\\test\\3.png"

force_type = {0: 'gun', 1: 'quantou', 2: '棍'}

w = 16
h = 16
c = 1


def read_one_image(path):
    img = io.imread(path)
    img = np.asarray(img)
    return img[:, :, np.newaxis]

def precast(filepath):
    with tf.Session() as sess:
        data = []
        data1 = read_one_image(filepath)
        data.append(data1)

        saver = tf.train.import_meta_graph('model\\model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('model\\'))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        feed_dict = {x: data}

        logits = graph.get_tensor_by_name("logits_eval:0")

        classification_result = sess.run(logits, feed_dict)

        # 打印出预测矩阵
        print(classification_result)
        # 打印出预测矩阵每一行最大值的索引
        output = tf.argmax(classification_result, 1).eval()
        print(force_type[output[0]])

precast(testpath1)