from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.config import cfg
from ult.ult import Get_Next_Instance_HO_Neg_HICO
from ult.timer import Timer

import os
import numpy as np
import pickle as pkl
import tensorflow as tf


classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
           'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
           'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
           'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
           'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
           'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
           'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cellphone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
           'teddy bear', 'hair drier', 'toothbrush']
clsid2cls = dict()
for i in range(len(classes)):
    clsid2cls[i+1] = classes[i]


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def save_tfrecords(data, desfile):
    head_feature = data
    head_H = data.shape[1]
    head_W = data.shape[2]
    with tf.python_io.TFRecordWriter(desfile) as writer:
        features = tf.train.Features(
            feature={
                'head': _bytes_feature(head_feature.astype(np.float64).tostring()),
                'H': _int64_feature(head_H),
                'W': _int64_feature(head_W)
            }
        )
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)


def read_tfrecords(desfile):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([desfile])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'head': tf.FixedLenFeature([], tf.string),
            'H': tf.FixedLenFeature([], tf.int64),
            'W': tf.FixedLenFeature([], tf.int64)
        }
    )

    head = tf.decode_raw(features['head'], tf.float64)
    H = features['H']
    W = features['W']
    #head = tf.reshape(head, [1, H, W, 1024])
    return head, H, W


class SolverWrapper(object):
    """
    A wrapper class for the interactive training process
    """

    def __init__(self, sess, network, Trainval_GT, Trainval_N, output_dir, tbdir,
                 Pos_augment, Neg_select, Restore_flag, pretrained_model, posetype):
        self.net = network
        self.Trainval_GT = Trainval_GT
        self.Trainval_N = Trainval_N
        self.output_dir = output_dir
        self.tbdir = tbdir
        self.Pos_augment = Pos_augment
        self.Neg_select = Neg_select
        self.Restore_flag = Restore_flag
        self.pretrained_model = pretrained_model
        self.posetype = posetype

    def snapshot(self, sess, iter):

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Store the model snapshot
        filename = 'HOI' + '_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

    def construct_graph(self, sess):
        with sess.graph.as_default():
            # Set the random seed for tensorflow
            tf.set_random_seed(cfg.RNG_SEED)

            # Build the main computation graph
            layers = self.net.create_architecture(True)  # is_training_flag: True

            # Define the loss
            loss = layers['binary_cross_entropy']

            path_iter = self.pretrained_model.split('.ckpt')[0]
            iter_num = path_iter.split('_')[-1]

            if cfg.TRAIN_MODULE_CONTINUE == 1:
                global_step = tf.Variable(int(iter_num), trainable=False)
            elif cfg.TRAIN_MODULE_CONTINUE == 2:
                global_step = tf.Variable(0, trainable=False)
            else:
                raise NotImplementedError

            lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE * 10, global_step, cfg.TRAIN.STEPSIZE * 5,
                                            cfg.TRAIN.GAMMA, staircase=True)
            print('LR: {:06f}, stepsize: {:d}'.format(cfg.TRAIN.LEARNING_RATE * 10, cfg.TRAIN.STEPSIZE * 5))

            self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)

            grads_and_vars = self.optimizer.compute_gradients(loss, tf.trainable_variables())
            capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars]
            train_op = self.optimizer.apply_gradients(capped_gvs, global_step=global_step)

            self.saver = tf.train.Saver(max_to_keep=cfg.TRAIN.SNAPSHOT_KEPT)
            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)

        return lr, train_op

    def from_snapshot(self, sess):
        if self.Restore_flag == 0:
            saver_t = [var for var in tf.model_variables() if 'conv1' in var.name and 'conv1_sp' not in var.name]
            saver_t += [var for var in tf.model_variables() if 'conv2' in var.name and 'conv2_sp' not in var.name]
            saver_t += [var for var in tf.model_variables() if 'conv3' in var.name]
            saver_t += [var for var in tf.model_variables() if 'conv4' in var.name]
            saver_t += [var for var in tf.model_variables() if 'conv5' in var.name]
            saver_t += [var for var in tf.model_variables() if 'shortcut' in var.name]

            sess.run(tf.global_variables_initializer())

            print('Restoring model snapshots from {:s}'.format(self.pretrained_model))
            for var in tf.trainable_variables():
                print(var.name, var.eval().mean())

            self.saver_restore = tf.train.Saver(saver_t)
            self.saver_restore.restore(sess, self.pretrained_model)

        if self.Restore_flag == 5 or self.Restore_flag == 6 or self.Restore_flag == 7:
            sess.run(tf.global_variables_initializer())
            for var in tf.trainable_variables():
                print(var.name, var.eval().mean())
            print('Restoring model snapshots from {:s}'.format(self.pretrained_model))
            saver_t = {}

            # Add block0
            for ele in tf.model_variables():
                if 'resnet_v1_50/conv1/weights' in ele.name or 'resnet_v1_50/conv1/BatchNorm/beta' in ele.name or 'resnet_v1_50/conv1/BatchNorm/gamma' in ele.name or 'resnet_v1_50/conv1/BatchNorm/moving_mean' in ele.name or 'resnet_v1_50/conv1/BatchNorm/moving_variance' in ele.name:
                    saver_t[ele.name[:-2]] = ele
            # Add block1
            for ele in tf.model_variables():
                if 'block1' in ele.name:
                    saver_t[ele.name[:-2]] = ele

            # Add block2
            for ele in tf.model_variables():
                if 'block2' in ele.name:
                    saver_t[ele.name[:-2]] = ele

            # Add block3
            for ele in tf.model_variables():
                if 'block3' in ele.name:
                    saver_t[ele.name[:-2]] = ele

            # Add block4
            for ele in tf.model_variables():
                if 'block4' in ele.name:
                    saver_t[ele.name[:-2]] = ele

            self.saver_restore = tf.train.Saver(saver_t)
            self.saver_restore.restore(sess, self.pretrained_model)

            if self.Restore_flag >= 5:

                saver_t = {}
                # Add block5
                for ele in tf.model_variables():
                    if 'block4' in ele.name:
                        saver_t[ele.name[:-2]] = \
                        [var for var in tf.model_variables() if ele.name[:-2].replace('block4', 'block5') in var.name][
                            0]

                self.saver_restore = tf.train.Saver(saver_t)
                self.saver_restore.restore(sess, self.pretrained_model)

            if self.Restore_flag >= 6:
                saver_t = {}
                # Add block6
                for ele in tf.model_variables():
                    if 'block4' in ele.name:
                        saver_t[ele.name[:-2]] = \
                        [var for var in tf.model_variables() if ele.name[:-2].replace('block4', 'block6') in var.name][
                            0]

                self.saver_restore = tf.train.Saver(saver_t)
                self.saver_restore.restore(sess, self.pretrained_model)

            if self.Restore_flag >= 7:

                saver_t = {}
                # Add block7
                for ele in tf.model_variables():
                    if 'block4' in ele.name:
                        saver_t[ele.name[:-2]] = \
                        [var for var in tf.model_variables() if ele.name[:-2].replace('block4', 'block7') in var.name][
                            0]

                self.saver_restore = tf.train.Saver(saver_t)
                self.saver_restore.restore(sess, self.pretrained_model)

    def from_previous_ckpt(self, sess):
        for var in tf.trainable_variables():
            print(var.name, var.eval().mean())

        print('Restoring model snapshots from {:s}'.format(self.pretrained_model))

        # add saved variables here
        saver_t = [var for var in tf.model_variables()]
        self.saver_restore = tf.train.Saver(saver_t)
        self.saver_restore.restore(sess, self.pretrained_model)

    def train_model(self, sess, max_iters):
        lr, train_op = self.construct_graph(sess)

        if cfg.TRAIN_RES_RESTORE:
            self.from_snapshot(sess)
        else:
            sess.run(tf.global_variables_initializer())
        if cfg.TRAIN_MODULE_CONTINUE == 1:  # continue training
            self.from_previous_ckpt(sess)
        #print("the variables is being trained now \n")
        #for var in tf.trainable_variables():
        #    print(var.name, var.eval().mean())

        #sess.graph.finalize()

        objs_bert = pkl.load(open(cfg.DATA_DIR + '/' + 'objs_bert1024.pkl', 'rb'))
        Data_length = len(self.Trainval_GT)
        path_iter = self.pretrained_model.split('.ckpt')[0]
        iter_num = path_iter.split('_')[-1]
        iter = int(iter_num) if cfg.TRAIN_MODULE_CONTINUE == 1 else 0
        init_iter = iter
        timer = Timer()
        num_pose_total = 0
        total = 0
        flag = 0
        while iter < max_iters + 1:
            if iter >= Data_length:
                break
            timer.tic()
            blobs = Get_Next_Instance_HO_Neg_HICO(self.Trainval_GT, self.Trainval_N, iter, self.Pos_augment,
                                                  self.Neg_select, Data_length, clsid2cls, objs_bert, self.posetype)
            '''
            num_pose_total += blobs['num_pose']
            total += blobs['H_num']
            binary_labels = blobs['binary_label']
            for j in range(binary_labels.shape[0]):
                if binary_labels[j][0] == 1:
                    num_pose_total += 1
                total += 1
            print(self.Trainval_GT[iter % Data_length][0], num_pose_total, total)
            '''
            if not blobs['pose_none_flag']:
                iter += 1
                continue
            blobs_size = dict()
            for k, v in blobs.items():
                blobs_size[k] = v.shape if type(v) is not int else v

            total_loss, cls_binary, binary_label, head = self.net.train_step(sess, blobs, lr.eval(), train_op)

            tfid = iter % Data_length
            tfpath = 'Temp/train/'+str(tfid)+'.npy'
            np.save(tfpath, head)

            timer.toc()

            if iter % cfg.TRAIN.DISPLAY == 0:
                print('iter: %d / %d, im_id: %u, total loss: %.6f, lr: %f, speed: %.3f s/ iter' %
                      (iter, max_iters, self.Trainval_GT[iter % Data_length][0], total_loss, lr.eval(),
                       timer.average_time))
                #for i in range(binary_label.shape[0]):
                #    print(cls_binary[i], binary_label[i])
            # Snapshotting
            if (iter % cfg.TRAIN.SNAPSHOT_ITERS * 2 == 0 and iter != 0) or (iter == 10):
                if iter != init_iter:
                    self.snapshot(sess, iter)
            iter += 1
        self.writer.close()


def train_net(network, Trainval_GT, Trainval_N, output_dir, tb_dir, Pos_augment, Neg_select, Restore_flag, posetype,
              pretrained_model, max_iters=30000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    if cfg.TRAIN_MODULE_CONTINUE == 2:  # train from beginning
        # Remove previous events
        filelist = [f for f in os.listdir(tb_dir)]
        for f in filelist:
            os.remove(os.path.join(tb_dir, f))

        # Remove previous snapshots
        #filelist = [f for f in os.listdir(output_dir)]
        #for f in filelist:
        #    os.remove(os.path.join(output_dir, f))
    if not os.path.exists('Temp/train'):
        os.makedirs('Temp/train')
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        sw = SolverWrapper(sess, network, Trainval_GT, Trainval_N, output_dir, tb_dir, Pos_augment, Neg_select,
                           Restore_flag, pretrained_model, posetype)
        print('Solving..., Pos augment = ' + str(Pos_augment) + ', Neg augment = ' + str(Neg_select))
        sw.train_model(sess, max_iters)
        print('done solving')
