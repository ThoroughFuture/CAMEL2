from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import tensorflow as tf
import numpy as np
from PIL import Image
import string
from sklearn.metrics import roc_auc_score
from resnet import resnet_v2_34, resnet_arg_scope

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import mnist
mnist.temporary_dir = lambda: '/work/home/xugang/projects/tho/data/mnist/slide_mnist/data'

test_images = mnist.test_images() # (10000, 28, 28)
test_labels = mnist.test_labels() # (10000,)

test_images = np.reshape(test_images, (10000, 28, 28, 1))
test_labels = test_labels.astype(np.int32)

def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main(test_images, test_labels, SNAPSHOT_DIR):
    with tf.name_scope("create_inputs"):
        image = tf.placeholder(tf.float32, [10000, 28, 28, 1])

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        with tf.contrib.slim.arg_scope(resnet_arg_scope(use_batch_norm=False)):
            logits, _ = resnet_v2_34(image / 255.0,
                                     num_classes=2,
                                     is_training=False)
            
    sm = tf.nn.softmax(logits)
    prediction = sm[:, 1]
    
    sesss = []
    for restore_path in restore_paths:
        # Which variables to load.
        restore_var = tf.global_variables()
    
        # Set up tf session and initialize variables.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
    
        sess.run(init)
        sess.run(tf.local_variables_initializer())
    
        # Load weights.
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, restore_path)
        
        sesss.append(sess)
        
    predict_values = []
    image_x = test_images
    prediction_values = []
    for sess in sesss:
        prediction_value = sess.run(prediction, feed_dict={image: image_x})
        prediction_values.append(prediction_value)
    
    output = np.mean(prediction_values, 0)
    output = output.flatten().tolist()
    predict_values.extend(output)
    
    TN = 0.0
    TP = 0.0
    FP = 0.0
    FN = 0.0 
    for g,p in zip(test_labels, predict_values):
        
        p = 1 if p >= 0.5 else 0
        
        if g == 0 and p == 0:
            TN = TN + 1
        elif g == 0 and p == 1:
            FP = FP + 1
        elif g == 1 and p == 0:
            FN = FN + 1
        elif g == 1 and p == 1:
            TP = TP + 1
    
    assert (TP + TN + FP + FN) == len(test_labels) == len(predict_values) == 10000

    presicion = TP / (TP + FP + 1e-5) 
    recall = TP / (TP + FN + 1e-5)
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-5)
    specificity = TN / (FP + TN + 1e-5)
    f1 = 2*presicion*recall / (presicion + recall + 1e-5)
    
    print (TP, TN, FP, FN)
    print ("presicion:", presicion)
    print ("specificity:", specificity)
    print ("recall:", recall)
    print ("f1:", f1)
    print ("acc:", acc)
    print ("auc:", roc_auc_score(test_labels, predict_values))
    
    return recall, specificity, f1, roc_auc_score(test_labels, predict_values)
    
if __name__ == '__main__':
    
    MAX_SAMPLE = 5000
    # assume the bag with image 0 is positive
    i = 0
        
    import copy

    test_labels_ = copy.deepcopy(test_labels)
    
    test_labels_[np.where(test_labels_== i)] = -1
    test_labels_[np.where(test_labels_ > 0)] = 0
    test_labels_[np.where(test_labels_ == -1)] = 1
    
    print (i, test_labels_.shape, np.sum(test_labels_))
    test_labels_ = test_labels_.tolist()
    
    SNAPSHOT_DIR = './snapshots_' + str(MAX_SAMPLE) + "_" + str(i) + '/'
    restore_paths = [SNAPSHOT_DIR + "model.ckpt-800",
                     SNAPSHOT_DIR + "model.ckpt-1100",
                     SNAPSHOT_DIR + "model.ckpt-1400",
                     SNAPSHOT_DIR + "model.ckpt-1700",
                     SNAPSHOT_DIR + "model.ckpt-2000"] # 8, 11, 14, 17, 20        
    recall, specificity, f1, auc = main(test_images, test_labels_, SNAPSHOT_DIR)
    tf.reset_default_graph()
        
