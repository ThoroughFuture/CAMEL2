from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import tensorflow as tf
import numpy as np

from resnet import resnet_v2_34, resnet_arg_scope

TRAIN = True

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
GPU_NUM = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

EPOCH = 20
STEP_PER_EPOCH = 100

PATCH_SIZE = 28

BATCH_PER_GPU = 1
TOP_N = None

MAX_SAMPLE = 5000
RATIO = 0.1

RESTORE_FROM = None
LEARNING_RATE = 1e-3
    
START_STEP = 1
NUM_STEPS = EPOCH*STEP_PER_EPOCH + 1

NUM_CLASSES = 2
SAVE_PRED_EVERY = STEP_PER_EPOCH

import mnist
mnist.temporary_dir = lambda: '/work/home/xugang/projects/tho/data/mnist/slide_mnist/data'

train_images = mnist.train_images() # (60000, 28, 28)
train_labels = mnist.train_labels() # (60000,)

def data_gen(train_0, train_1, label_0, label_1):
    
    total_0 = train_0.shape[0]
    total_1 = train_1.shape[0]
    
    data1s = []
    data0s = []
    label1s = []
    label0s = []
    for _ in range(GPU_NUM):
        idx_0 = np.array(range(total_0))
        np.random.shuffle(idx_0)
    
        idx_1 = np.array(range(total_1))
        np.random.shuffle(idx_1)
        
        num_0_0 = np.random.randint(1, 1000)
        num_0_1 = MAX_SAMPLE - num_0_0
        
        num_1 = MAX_SAMPLE
        
        data00 = train_0[idx_0[:num_0_0]]
        data01 = train_1[idx_1[-num_0_1:]]
        
        label00 = label_0[idx_0[:num_0_0]]
        label01 = label_1[idx_1[-num_0_1:]]
        
        data1 = train_1[idx_1[:num_1]]
        label1 = label_1[idx_1[:num_1]]
        
        data0 = np.concatenate([data00, data01], 0)
        label0 = np.concatenate([label00, label01], 0)
        
        data0 = np.reshape(data0, (MAX_SAMPLE, 28, 28, 1))
        data1 = np.reshape(data1, (MAX_SAMPLE, 28, 28, 1))
    
        assert data0.shape == (MAX_SAMPLE, 28, 28, 1)
        assert data1.shape == (MAX_SAMPLE, 28, 28, 1)
        assert label0.shape == (MAX_SAMPLE,)
        assert label1.shape == (MAX_SAMPLE,)
        
        data1s.append(data1)
        data0s.append(data0)
        label1s.append(label1)
        label0s.append(label0)
    
    data1s = np.array(data1s)
    data0s = np.array(data0s)
    label1s = np.array(label1s)
    label0s = np.array(label0s)
    
    return data1s/255.0, data0s/255.0, label1s, label0s

def save(saver, sess, logdir, step):
   '''Save weights.

   Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
   '''
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')
   return checkpoint_path+'-'+str(step)


def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def get_lossid_and_labels(results_sm_0, results_sm_1):

    labels_onehot = []
    
    sm_onedim_1 = results_sm_1[:, 1]
    sm_onedim_1 = sm_onedim_1.flatten()
    
    num_1 = int(np.ceil(RATIO*MAX_SAMPLE)) - 1
    num_1 = max(1, num_1)
    
    idex_1 = np.argsort(sm_onedim_1).tolist()
    idex_1.reverse()
    result_id_1 = np.array(idex_1[:num_1])
    
    sm_onedim_0 = results_sm_0[:, 1]
    sm_onedim_0 = sm_onedim_0.flatten()

    num_0 = MAX_SAMPLE
    idex_0 = np.array(range(num_0))
    result_id_0 = np.array(idex_0)

    for i in range(num_0):
        labels_onehot.append([1,0])
    for i in range(num_1):
        labels_onehot.append([0,1])
        
    result_id_0 = np.array(result_id_0).astype(np.int32)
    result_id_1 = np.array(result_id_1).astype(np.int32)
    labels_onehot = np.array(labels_onehot).astype(np.float32)
    
    return result_id_0, result_id_1, labels_onehot

def main(train_0, train_1, label_0, label_1, SNAPSHOT_DIR):
    """Create the model and start the training."""
    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    print('Training: ' + str(TRAIN))
    print('LEARNING_RATE: ' + str(LEARNING_RATE))
    print('PATCH_SIZE: ' + str(PATCH_SIZE))
    print('MAX_SAMPLE: ' + str(MAX_SAMPLE))
    print('RATIO: ' + str(RATIO))
    print('GPU_NUM: ' + str(GPU_NUM))
    print('BATCH_PER_GPU: ' + str(BATCH_PER_GPU))
    print('TOP_N: ' + str(TOP_N))
    
    with tf.device("/cpu:0"):
        lr = tf.Variable(tf.constant(LEARNING_RATE), name='lr', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)               
        
        tower_grads = [[] for _ in range(GPU_NUM)]
        tower_losses = [0.0 for _ in range(GPU_NUM)]

    with tf.device("/cpu:0"):
        ph_inputs_0_imgs = tf.placeholder(tf.float32, [GPU_NUM, MAX_SAMPLE, PATCH_SIZE, PATCH_SIZE, 1])
        ph_inputs_1_imgs = tf.placeholder(tf.float32, [GPU_NUM, MAX_SAMPLE, PATCH_SIZE, PATCH_SIZE, 1])
            
    # Create network.
    for i in range(GPU_NUM):
        with tf.device("/cpu:0"):
            imgs_0 = ph_inputs_0_imgs[i]
            imgs_1 = ph_inputs_1_imgs[i]
                                
        with tf.device("/gpu:%d" % i):
            if i == 0:
                re_use = False
            else:
                re_use = True
            with tf.variable_scope(tf.get_variable_scope(), reuse=re_use):
                image_input = tf.concat([imgs_0, imgs_1], 0)
                # encoder
                with tf.contrib.slim.arg_scope(resnet_arg_scope(use_batch_norm=False)):
                    logits, _ = resnet_v2_34(image_input,
                                             num_classes=NUM_CLASSES,
                                             is_training=TRAIN)
                
                logits_0 = logits[:MAX_SAMPLE, :]
                logits_1 = logits[MAX_SAMPLE:, :]
                
                sm_0 = tf.nn.softmax(logits_0)
                sm_1 = tf.nn.softmax(logits_1)

                results_id_0, results_id_1, labels_onehot = tf.py_func(get_lossid_and_labels, [sm_0, sm_1], [tf.int32, tf.int32, tf.float32])
                
                logits_select = tf.concat([tf.gather(logits_0, results_id_0), tf.gather(logits_1, results_id_1)], 0)

                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_select, labels=labels_onehot))

                all_trainable = [v for v in tf.trainable_variables()]
                
                tower_losses[i] = loss
                tower_grads[i] = optimizer.compute_gradients(loss, var_list=all_trainable)                

    with tf.device("/cpu:0"):
        reduced_loss = tower_losses[0]
        for _ in range(1, GPU_NUM):
            reduced_loss = tf.add(reduced_loss, tower_losses[_])
            
        reduced_loss = reduced_loss / (GPU_NUM + 0.0)
        grads = average_gradients(tower_grads)

        train_op = optimizer.apply_gradients(grads)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    restore_var = [v for v in tf.global_variables()]
    
    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1000)

    # Load variables if the checkpoint is provided.
    if RESTORE_FROM is not None:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, RESTORE_FROM)
        
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    # Iterate over training steps.
    for step in range(START_STEP, NUM_STEPS):
        start_time = time.time()
        data0s, data1s, label0s, label1s = data_gen(train_0, train_1, label_0, label_1)
        
        loss_value, _, _  =  sess.run([reduced_loss, train_op, update_ops],
                                      feed_dict={ph_inputs_0_imgs: data0s,
                                                 ph_inputs_1_imgs: data1s})
        
        if step % (STEP_PER_EPOCH*5) == 0:
            sess.run(tf.assign(lr, lr/2))
            print('LEARNING_RATE:',sess.run(lr))

        if step in [800, 1100, 1400, 1700, 2000]:
            save(saver, sess, SNAPSHOT_DIR, step)
        
        duration = time.time() - start_time
        print('step {:d}, lr {:.3e}, loss = {:.3f}({:.3f} sec/step)'\
            .format(step, sess.run(lr), loss_value, duration))
    
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    
    # assume the bag with image 0 is positive
    i = 0
    
    index_0 = np.where(train_labels==i)
    index_1 = np.where(train_labels!=i)
    
    train_0 = train_images[index_0]
    train_1 = train_images[index_1]
    
    label_0 = train_labels[index_0]
    label_1 = train_labels[index_1]
    
    SNAPSHOT_DIR = './snapshots_' + str(MAX_SAMPLE) + "_" + str(i) + '/'
    if not os.path.exists(SNAPSHOT_DIR):
        os.mkdir(SNAPSHOT_DIR)
            
        main(train_0, train_1, label_0, label_1, SNAPSHOT_DIR)
        tf.reset_default_graph()
