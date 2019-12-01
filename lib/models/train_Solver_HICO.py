from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.config import cfg
from utils.data_loader import Get_Next_Instance_HICO, Timer
import os
import pickle as pkl
import tensorflow as tf
import numpy as np

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


class SolverWrapper(object):
    """
    A wrapper class for the interactive training process
    """

    def __init__(self, network, Trainval_GT, Trainval_N, output_dir, Pos_augment, Neg_select, Restore_flag, pretrained_model):
        self.net = network
        self.interval_divide = 5
        self.Trainval_GT = self.changeForm(Trainval_GT, self.interval_divide)
        self.Trainval_N = Trainval_N
        self.output_dir = output_dir
        self.Pos_augment = Pos_augment
        self.Neg_select = Neg_select
        self.Restore_flag = Restore_flag
        self.pretrained_model = pretrained_model

    def snapshot(self, sess, iter):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # Store the model snapshot
        filename = 'HOI' + '_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

    def changeForm(self, Trainval_GT, interval_divide):
        GT_dict = {}
        for item in Trainval_GT:
            try:
                GT_dict[item[0]].append(item)
            except KeyError:
                GT_dict[item[0]] = [item]
        GT_new = []
        for image_id, value in GT_dict.items():
            count = 0
            length = len(value)
            while count < length:
                temp = value[count: min(count+interval_divide, length)]
                count += len(temp)
                GT_new.append(temp)
        return GT_new

    def construct_graph(self, sess):
        with sess.graph.as_default():
            # Set the random seed for tensorflow
            tf.set_random_seed(cfg.RNG_SEED)

            # Build the main computation graph
            layers = self.net.create_architecture(True)  # is_training_flag: True

            # Define the loss
            loss = layers['total_loss']

            path_iter = self.pretrained_model.split('.ckpt')[0]
            iter_num = path_iter.split('_')[-1]

            if cfg.TRAIN_MODULE_CONTINUE == 1:
                global_step = tf.Variable(int(iter_num), trainable=False)
            elif cfg.TRAIN_MODULE_CONTINUE == 2:
                global_step = tf.Variable(0, trainable=False)
            else:
                raise NotImplementedError

            #lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE * 10, global_step, cfg.TRAIN.STEPSIZE * 5,
            #                                cfg.TRAIN.GAMMA, staircase=True)
            lr = tf.train.piecewise_constant(global_step, boundaries=[250000], values=[0.04, 0.004])

            self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)

            grads_and_vars = self.optimizer.compute_gradients(loss, tf.trainable_variables())
            capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars]
            train_op = self.optimizer.apply_gradients(capped_gvs, global_step=global_step)
            self.saver = tf.train.Saver(max_to_keep=cfg.TRAIN.SNAPSHOT_KEPT)

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

            if self.Restore_flag > 5:
                saver_t = {}
                # Add block5
                for ele in tf.model_variables():
                    if 'block4' in ele.name:
                        saver_t[ele.name[:-2]] = \
                        [var for var in tf.model_variables() if ele.name[:-2].replace('block4', 'block5') in var.name][
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
        if cfg.TRAIN_MODULE_CONTINUE:  # continue training
            self.from_previous_ckpt(sess)

        sess.graph.finalize()
        objs_bert = pkl.load(open(cfg.DATA_DIR + '/' + 'objs_bert1024.pkl', 'rb'))
        Data_length = len(self.Trainval_GT)
        path_iter = self.pretrained_model.split('.ckpt')[0]
        iter_num = path_iter.split('_')[-1]
        iter = int(iter_num) if cfg.TRAIN_MODULE_CONTINUE == 1 else 0
        init_iter = iter
        timer = Timer()

        print('Data_length', Data_length)
        while iter < max_iters + 1:
            timer.tic()
            blobs = Get_Next_Instance_HICO(self.Trainval_GT, self.Trainval_N, iter,
                                           self.Pos_augment, self.Neg_select, Data_length, clsid2cls, objs_bert)

            blobs['head'] = np.load('Temp/train/' + str(blobs['image_id']) + '.npy')
            if not blobs['pose_none_flag']:
                iter += 1
                continue
            total_loss = self.net.train_step(sess, blobs, lr.eval(), train_op)
            timer.toc()
            if iter % cfg.TRAIN.DISPLAY == 0:
                print('iter: %d / %d, im_id: %u, total loss: %.6f, lr: %f, speed: %.3f s/ iter' %
                      (iter, max_iters, self.Trainval_GT[iter % Data_length][0][0], total_loss, lr.eval(),
                       timer.average_time))

            # Snapshotting
            if (iter % cfg.TRAIN.SNAPSHOT_ITERS * 2 == 0 and iter != 0) or (iter == 10):
                if iter != init_iter:
                    self.snapshot(sess, iter)
            iter += 1


def train_net(network, Trainval_GT, Trainval_N, output_dir, pretrained_model, Pos_augment, Neg_select,
              Restore_flag, max_iters=30000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        sw = SolverWrapper(network, Trainval_GT, Trainval_N, output_dir, Pos_augment, Neg_select,
                           Restore_flag, pretrained_model)
        print('Solving..., Pos augment = ' + str(Pos_augment) + ', Neg augment = ' + str(Neg_select))
        sw.train_model(sess, max_iters)
        print('done solving')
