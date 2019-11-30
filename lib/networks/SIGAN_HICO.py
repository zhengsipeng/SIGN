import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.config import cfg
from sp_gcn import Graph
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.python.framework import ops


def resnet_arg_scope(is_training=True,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': ops.GraphKeys.UPDATE_OPS
    }
    with arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_regularizer=tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
            weights_initializer=slim.variance_scaling_initializer(),
            biases_regularizer=tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
            biases_initializer=tf.constant_initializer(0.0),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


class SPGCN():
    def __init__(self,
                 in_channels=3,
                 num_classes=0,
                 num_nodes=17+2,
                 edge_importance_weighting=True,
                 is_training=True,
                 num_fc=1024,
                 posetype=1,
                 bi_posegraph=False,
                 bodypart=False,
                 binary=False,
                 posemap=False,
                 posegraph=False,
                 semantic=False,
                 data_bn=True):
        self.predictions = {}
        self.train_summaries = []
        self.losses = {}
        self.lr = tf.placeholder(tf.float32)
        self.num_binary = 1  # existence of HOI (0 or 1)
        self.num_classes = 600
        self.gt_binary_label = tf.placeholder(tf.float32, shape=[None, 1], name='gt_binary_label')
        self.gt_class_HO = tf.placeholder(tf.float32, shape=[None, 600], name='gt_class_HO')
        self.is_training = is_training
        if self.is_training:
            self.keep_prob = cfg.TRAIN_DROP_OUT_BINARY
            self.keep_prob_tail = .5
        else:
            self.keep_prob = 1
            self.keep_prob_tail = 1

        self.image = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='image')
        self.head = tf.placeholder(tf.float32, shape=[1, None, None, 1024], name='head')
        self.H_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='H_boxes')
        self.O_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='O_boxes')
        self.partboxes = tf.placeholder(tf.float32, shape=[None, 17, 5], name='part_boxes')
        self.semantic = tf.placeholder(tf.float32, shape=[None, 1024], name='semantic_feat')
        self.H_num = tf.placeholder(tf.int32)

        # Control the network architecture
        self.bodypart = bodypart
        self.binary = binary
        self.posemap = posemap
        self.posegraph = posegraph
        self.bi_posegraph = bi_posegraph
        self.posetype = posetype
        self.semantic_flag = semantic
        if self.posetype == 1:
            self.spatial = tf.placeholder(tf.float32, shape=[None, 64, 64, 2], name='sp')
        else:
            self.spatial = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='sp')

        # ResNet 50 Network
        self.scope = 'resnet_v1_50'
        self.num_fc = 1024
        self.num_fc2 = num_fc
        self.stride = [16, ]
        if tf.__version__ == '1.1.0':
            self.blocks = [resnet_utils.Block('block1', resnet_v1.bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
                           resnet_utils.Block('block2', resnet_v1.bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
                           resnet_utils.Block('block3', resnet_v1.bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 1)]),
                           resnet_utils.Block('block4', resnet_v1.bottleneck, [(2048, 512, 1)] * 3),
                           resnet_utils.Block('block5', resnet_v1.bottleneck, [(2048, 512, 1)] * 3)]
        else:
            from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
            self.blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                           resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                           resnet_v1_block('block3', base_depth=256, num_units=6, stride=1),
                           resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
                           resnet_v1_block('block5', base_depth=512, num_units=3, stride=1)]

        # GCN setting
        self.num_nodes = num_nodes
        self.c = in_channels
        self.data_bn = data_bn
        self.strategy = 'spatial'
        self.graph = Graph(strategy=self.strategy)
        self.A = tf.convert_to_tensor(self.graph.A.astype(np.float32))  # [None, num_nodes, num_nodes]
        self.spatial_kernel_size = self.A.shape[0]
        # [N, C, T, V, M] = [N, 3, 1, 19, 1]
        self.Gnodes = tf.placeholder(tf.float32, shape=[None, self.c, 1, self.num_nodes, 1], name='Gnodes')

        # ST_GCN
        self.depth_st_gcn_networks = 10
        #self.edge_importance_weighting = edge_importance_weighting
        #self.edge_importance = self.generate_edge_importance()

    #def generate_edge_importance(self):
    #    if self.edge_importance_weighting:
    #    else:
    #        return [1] * self.len_st_gcn_networks

    # ======================
    # Global Spatial Module
    # ======================
    def sp_to_head(self):
        with tf.variable_scope(self.scope, self.scope):
            conv1_sp = slim.conv2d(self.spatial[:, :, :, 0:2], 64, [5, 5], padding='VALID', scope='conv1_sp')
            pool1_sp = slim.max_pool2d(conv1_sp, [2, 2], scope='pool1_sp')
            conv2_sp = slim.conv2d(pool1_sp, 32, [5, 5], padding='VALID', scope='conv2_sp')
            pool2_sp = slim.max_pool2d(conv2_sp, [2, 2], scope='pool2_sp')
            pool2_flat_sp = slim.flatten(pool2_sp)
            fc_sp = slim.fully_connected(pool2_flat_sp, self.num_fc2, scope='fc_sp')
            fc_sp = slim.dropout(fc_sp, keep_prob=self.keep_prob, scope='dropout_fc_sp')
        return fc_sp  #pool2_flat_sp

    # ===============
    # ResNet Module
    # ===============
    def build_base(self):
        with tf.variable_scope(self.scope, self.scope):
            net = resnet_utils.conv2d_same(self.image, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')
        return net

    def image_to_head(self, is_training):
        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net = self.build_base()
            net, _ = resnet_v1.resnet_v1(net,
                                         self.blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                         global_pool=False,
                                         include_root_block=False,
                                         scope=self.scope)
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            head, _ = resnet_v1.resnet_v1(net,
                                          self.blocks[cfg.RESNET.FIXED_BLOCKS:-2],
                                          global_pool=False,
                                          include_root_block=False,
                                          scope=self.scope)
        return head

    def crop_pool_layer(self, bottom, rois, name, pooling_size):
        with tf.variable_scope(name) as scope:

            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self.stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self.stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height

            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            if cfg.RESNET.MAX_POOL:
                pre_pool_size = pooling_size * 2
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                                 name="crops")
                crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
            else:
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids),
                                                 [cfg.POOLING_SIZE, cfg.POOLING_SIZE], name="crops")
        return crops

    def attention_pool_layer_H(self, bottom, fc7_H, is_training, name):
        with tf.variable_scope(name) as scope:
            fc1 = slim.fully_connected(fc7_H, 512, scope='fc1_b')
            fc1 = slim.dropout(fc1, keep_prob=self.keep_prob, is_training=is_training, scope='dropout1_b')
            fc1 = tf.reshape(fc1, [tf.shape(fc1)[0], 1, 1, tf.shape(fc1)[1]])
            att = tf.reduce_mean(tf.multiply(bottom, fc1), 3, keepdims=True)
        return att

    def attention_norm_H(self, att, name):
        with tf.variable_scope(name) as scope:
            att = tf.transpose(att, [0, 3, 1, 2])
            att_shape = tf.shape(att)
            att = tf.reshape(att, [att_shape[0], att_shape[1], -1])
            att = tf.nn.softmax(att)
            att = tf.reshape(att, att_shape)
            att = tf.transpose(att, [0, 2, 3, 1])
        return att

    def attention_pool_layer_O(self, bottom, fc7_O, is_training, name):
        with tf.variable_scope(name) as scope:
            fc1 = slim.fully_connected(fc7_O, 512, scope='fc1_b')
            fc1 = slim.dropout(fc1, keep_prob=self.keep_prob, is_training=is_training, scope='dropout1_b')
            fc1 = tf.reshape(fc1, [tf.shape(fc1)[0], 1, 1, tf.shape(fc1)[1]])
            att = tf.reduce_mean(tf.multiply(bottom, fc1), 3, keepdims=True)
        return att

    def attention_norm_O(self, att, name):
        with tf.variable_scope(name) as scope:
            att = tf.transpose(att, [0, 3, 1, 2])
            att_shape = tf.shape(att)
            att = tf.reshape(att, [att_shape[0], att_shape[1], -1])
            att = tf.nn.softmax(att)
            att = tf.reshape(att, att_shape)
            att = tf.transpose(att, [0, 2, 3, 1])
        return att

    def res5(self, pool5_H, pool5_O, sp, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            fc7_H, _ = resnet_v1.resnet_v1(pool5_H,
                                           self.blocks[-2:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=False,
                                           scope=self.scope)

            fc7_H = tf.reduce_mean(fc7_H, axis=[1, 2])

            fc7_O, _ = resnet_v1.resnet_v1(pool5_O,
                                           self.blocks[-1:],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=False,
                                           scope=self.scope)

            fc7_O = tf.reduce_mean(fc7_O, axis=[1, 2])

        return fc7_H, fc7_O

    def res5_part(self, pool5_H, pool5_O, pool_part, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            fc7_H, _ = resnet_v1.resnet_v1(pool5_H,
                                           self.blocks[-2:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=False,
                                           scope=self.scope)

            fc7_H = tf.reduce_mean(fc7_H, axis=[1, 2])

            fc7_O, _ = resnet_v1.resnet_v1(pool5_O,
                                           self.blocks[-1:],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=False,
                                           scope=self.scope)

            fc7_part, _ = resnet_v1.resnet_v1(pool_part,
                                           self.blocks[-2:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=True,
                                           scope=self.scope)

            pool_part = tf.reduce_mean(fc7_part, axis=[1, 2])
            fc7_O = tf.reduce_mean(fc7_O, axis=[1, 2])

        return fc7_H, fc7_O, pool_part

    def head_to_tail(self, fc7_H, fc7_O, pool5_SH, pool5_SO, sp, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            fc7_SH = tf.reduce_mean(pool5_SH, axis=[1, 2])
            fc7_SO = tf.reduce_mean(pool5_SO, axis=[1, 2])

            Concat_SH = tf.concat([fc7_H, fc7_SH], 1)
            fc8_SH = slim.fully_connected(Concat_SH, self.num_fc, scope='fc8_SH')
            fc8_SH = slim.dropout(fc8_SH, keep_prob=self.keep_prob_tail, is_training=is_training, scope='dropout8_SH')
            fc9_SH = slim.fully_connected(fc8_SH, self.num_fc, scope='fc9_SH')
            fc9_SH = slim.dropout(fc9_SH, keep_prob=self.keep_prob_tail, is_training=is_training, scope='dropout9_SH')

            Concat_SO = tf.concat([fc7_O, fc7_SO], 1)
            fc8_SO = slim.fully_connected(Concat_SO, self.num_fc, scope='fc8_SO')
            fc8_SO = slim.dropout(fc8_SO, keep_prob=self.keep_prob_tail, is_training=is_training, scope='dropout8_SO')
            fc9_SO = slim.fully_connected(fc8_SO, self.num_fc, scope='fc9_SO')
            fc9_SO = slim.dropout(fc9_SO, keep_prob=self.keep_prob_tail, is_training=is_training, scope='dropout9_SO')
            #if not self.posegraph:
            Concat_SHsp = tf.concat([fc7_H, sp], 1)
            #else:
            #    Concat_SHsp = tf.concat([fc7_H, sp, self.predictions['all_nodes']], axis=1)
            Concat_SHsp = slim.fully_connected(Concat_SHsp, self.num_fc, scope='Concat_SHsp')
            Concat_SHsp = slim.dropout(Concat_SHsp, keep_prob=0.5, is_training=is_training, scope='dropout6_SHsp')
            fc7_SHsp = slim.fully_connected(Concat_SHsp, self.num_fc, scope='fc7_SHsp')
            fc7_SHsp = slim.dropout(fc7_SHsp, keep_prob=0.5, is_training=is_training, scope='dropout7_SHsp')

        return fc9_SH, fc9_SO, fc7_SHsp

    def bottleneck(self, bottom, is_training, name, reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            head_bottleneck = slim.conv2d(bottom, 1024, [1, 1], scope=name)

        return head_bottleneck

    # ======================
    # Binary Classification
    # ======================
    # binary discriminator for 0/1 classification of interaction, fc7_H, fc7_SH, fc7_O, fc7_SO, sp
    def binary_discriminator(self, fc7_H, fc7_O, fc7_SH, fc7_SO, sp, is_training, name):
        with tf.variable_scope(name) as scope:
            conv1_pose_map = slim.conv2d(self.spatial[:, :, :, 2:], 32, [5, 5], padding='VALID', scope='conv1_pose_map')
            pool1_pose_map = slim.max_pool2d(conv1_pose_map, [2, 2], scope='pool1_pose_map')
            conv2_pose_map = slim.conv2d(pool1_pose_map, 16, [5, 5], padding='VALID', scope='conv2_pose_map')
            pool2_pose_map = slim.max_pool2d(conv2_pose_map, [2, 2], scope='pool2_pose_map')
            pool2_flat_pose_map = slim.flatten(pool2_pose_map)

            # fc7_H + fc7_SH + sp---fc1024---fc8_binary_1
            if not self.bi_posegraph:
                fc_binary_1 = tf.concat([fc7_H, fc7_SH], 1)  # [pos + neg, 3072]
                fc_binary_2 = tf.concat([fc7_O, fc7_SO], 1)  # [pos, 3072]
            else:
                fc_binary_1 = tf.concat([fc7_H, fc7_SH, self.predictions['hnodes']], axis=1)
                fc_binary_2 = tf.concat([fc7_O, fc7_SO, self.predictions['onodes']], 1)  # [pos, 3072]
                pool2_flat_pose_map = slim.fully_connected(pool2_flat_pose_map, 1024, scope='pose_map_fc')
                pool2_flat_pose_map = slim.dropout(pool2_flat_pose_map, keep_prob=self.keep_prob, scope='dropout_pose_map_fc')
            fc_binary_1 = tf.concat([fc_binary_1, sp, pool2_flat_pose_map], 1)  # [pos + neg, 8480]
            fc8_binary_1 = slim.fully_connected(fc_binary_1, 1024, scope='fc8_binary_1')
            fc8_binary_1 = slim.dropout(fc8_binary_1, keep_prob=cfg.TRAIN_DROP_OUT_BINARY, is_training=is_training,
                                        scope='dropout8_binary_1')  # [pos + neg,1024]

            # fc7_O + fc7_SO---fc1024---fc8_binary_2
            fc8_binary_2 = slim.fully_connected(fc_binary_2, 1024, scope='fc8_binary_2')
            fc8_binary_2 = slim.dropout(fc8_binary_2, keep_prob=cfg.TRAIN_DROP_OUT_BINARY, is_training=is_training,
                                        scope='dropout8_binary_2')  # [pos,1024]
            fc8_binary_2 = tf.concat([fc8_binary_2, fc8_binary_1[self.H_num:, :]], 0)

            # fc8_binary_1 + fc8_binary_2---fc1024---fc9_binary
            fc8_binary = tf.concat([fc8_binary_1, fc8_binary_2], 1)
            fc9_binary = slim.fully_connected(fc8_binary, 1024, scope='fc9_binary')
            fc9_binary = slim.dropout(fc9_binary, keep_prob=cfg.TRAIN_DROP_OUT_BINARY, is_training=is_training,
                                      scope='dropout9_binary')
        return fc9_binary

    # biary classification
    def binary_classification(self, fc9_binary, is_training, initializer, name):
        with tf.variable_scope(name) as scope:
            cls_score_binary = slim.fully_connected(fc9_binary, self.num_binary,
                                                    weights_initializer=initializer,
                                                    trainable=is_training,
                                                    activation_fn=None, scope='cls_score_binary')
            cls_prob_binary = tf.nn.sigmoid(cls_score_binary, name='cls_prob_binary')
            tf.reshape(cls_prob_binary, [1, self.num_binary])

            self.predictions["cls_score_binary"] = cls_score_binary
            self.predictions["cls_prob_binary"] = cls_prob_binary

    # ======================
    # Region Classification
    # ======================
    def region_classification_iCAN(self, fc7_H, fc7_O, fc7_SHsp, is_training,
                                   initializer, name):
        with tf.variable_scope(name) as scope:
            if self.posemap:
                conv1_pose_map = slim.conv2d(self.spatial[:, :, :, 2:], 32, [5, 5], padding='VALID',
                                             scope='conv1_pose_map')
                pool1_pose_map = slim.max_pool2d(conv1_pose_map, [2, 2], scope='pool1_pose_map')
                conv2_pose_map = slim.conv2d(pool1_pose_map, 16, [5, 5], padding='VALID', scope='conv2_pose_map')
                pool2_pose_map = slim.max_pool2d(conv2_pose_map, [2, 2], scope='pool2_pose_map')
                pool2_flat_pose_map = slim.flatten(pool2_pose_map)
                posemap_fc = slim.fully_connected(pool2_flat_pose_map, self.num_fc2, scope='posemap_fc')
                posemap_fc = slim.dropout(posemap_fc, keep_prob=self.keep_prob, is_training=is_training,
                                          scope='dropout_posemap')

                fc7_H = tf.concat([fc7_H, posemap_fc], axis=1)
                #fc7_O = tf.concat([fc7_O, posemap_fc], axis=1)
                fc7_SHsp = tf.concat([fc7_SHsp, posemap_fc], axis=1)

            # ---------------
            # Body Part
            # ----------------
            if self.bodypart:
                fc1_bd = slim.fully_connected(self.predictions['fc7_part'], 1024, scope='fc1_bd')
                fc1_bd = slim.dropout(fc1_bd, keep_prob=self.keep_prob, scope='dropout1_bd')
                fc2_bd = slim.fully_connected(fc1_bd, 1024, scope='fc2_bd')
                fc2_bd = slim.dropout(fc2_bd, keep_prob=self.keep_prob, scope='dropout2_bd')
                fc7_H = tf.concat([fc7_H, fc2_bd], axis=1)
                #fc7_SHsp = tf.concat([fc7_SHsp, fc2_bd], axis=1)

            # ----------------
            # Semantic Module
            # ----------------
            if self.semantic_flag:
                fc_semantic = slim.fully_connected(self.semantic, self.num_fc2, scope='fc_semantic')
                fc_semantic = slim.dropout(fc_semantic, keep_prob=self.keep_prob, is_training=is_training,
                                           scope='dropout_semantic')
                fc7_H = tf.concat([fc7_H, fc_semantic], axis=1)
                fc7_O = tf.concat([fc7_O, fc_semantic], axis=1)
                fc7_SHsp = tf.concat([fc7_SHsp, fc_semantic], axis=1)

            cls_score_H = slim.fully_connected(fc7_H, self.num_classes,
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_H')
            cls_prob_H = tf.nn.sigmoid(cls_score_H, name='cls_prob_H')
            tf.reshape(cls_prob_H, [1, self.num_classes])

            cls_score_O = slim.fully_connected(fc7_O, self.num_classes,
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_O')
            cls_prob_O = tf.nn.sigmoid(cls_score_O, name='cls_prob_O')
            tf.reshape(cls_prob_O, [1, self.num_classes])
            self.predictions["cls_score_H"] = cls_score_H
            self.predictions["cls_prob_H"] = cls_prob_H
            self.predictions["cls_score_O"] = cls_score_O
            self.predictions["cls_prob_O"] = cls_prob_O

            cls_score_sp = slim.fully_connected(fc7_SHsp, self.num_classes,
                                                weights_initializer=initializer,
                                                trainable=is_training,
                                                activation_fn=None, scope='cls_score_sp')
            cls_prob_sp = tf.nn.sigmoid(cls_score_sp, name='cls_prob_sp')
            tf.reshape(cls_prob_sp, [1, self.num_classes])
            self.predictions["cls_score_sp"] = cls_score_sp
            self.predictions["cls_prob_sp"] = cls_prob_sp
            self.predictions["cls_prob_HO"] = cls_prob_sp * (cls_prob_O + cls_prob_H)

            #if self.binary:
            #    self.predictions['cls_prob_HO'] = self.predictions["cls_prob_binary"] * \
            #                                      self.predictions['cls_prob_HO']

    def region_classification(self, fc7_H, fc7_O, fc7_SHsp, allnodes, hnodes, onodes,
                              is_training, initializer, name):
        with tf.variable_scope(name) as scope:
            if self.posemap:
                conv1_pose_map = slim.conv2d(self.spatial[:, :, :, 2:], 32, [5, 5], padding='VALID',
                                             scope='conv1_pose_map')
                pool1_pose_map = slim.max_pool2d(conv1_pose_map, [2, 2], scope='pool1_pose_map')
                conv2_pose_map = slim.conv2d(pool1_pose_map, 16, [5, 5], padding='VALID', scope='conv2_pose_map')
                pool2_pose_map = slim.max_pool2d(conv2_pose_map, [2, 2], scope='pool2_pose_map')
                pool2_flat_pose_map = slim.flatten(pool2_pose_map)
                posemap_fc = slim.fully_connected(pool2_flat_pose_map, self.num_fc2, scope='posemap_fc')
                posemap_fc = slim.dropout(posemap_fc, keep_prob=self.keep_prob, is_training=is_training,
                                          scope='dropout_posemap')

                fc7_H = tf.concat([fc7_H, posemap_fc], axis=1)
                fc7_SHsp = tf.concat([fc7_SHsp, posemap_fc], axis=1)

            # ----------------
            # Semantic Module
            # ----------------
            if self.semantic_flag:
                fc_semantic = slim.fully_connected(self.semantic, self.num_fc2, scope='fc_semantic')
                fc_semantic = slim.dropout(fc_semantic, keep_prob=self.keep_prob, is_training=is_training,
                                           scope='dropout_semantic')
                fc7_H = tf.concat([fc7_H, fc_semantic], axis=1)
                fc7_O = tf.concat([fc7_O, fc_semantic], axis=1)
                fc7_SHsp = tf.concat([fc7_SHsp, fc_semantic], axis=1)

            fc7_H = tf.concat([fc7_H, hnodes], axis=1)
            cls_score_H = slim.fully_connected(fc7_H, self.num_classes,
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_H')
            cls_prob_H = tf.nn.sigmoid(cls_score_H, name='cls_prob_H')
            tf.reshape(cls_prob_H, [1, self.num_classes])

            fc7_O = tf.concat([fc7_O, onodes[:self.H_num]], axis=1)
            cls_score_O = slim.fully_connected(fc7_O, self.num_classes,
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_O')
            cls_prob_O = tf.nn.sigmoid(cls_score_O, name='cls_prob_O')
            tf.reshape(cls_prob_O, [1, self.num_classes])

            cls_score_sp = slim.fully_connected(fc7_SHsp, self.num_classes,
                                                weights_initializer=initializer,
                                                trainable=is_training,
                                                activation_fn=None, scope='cls_score_sp')
            cls_prob_sp = tf.nn.sigmoid(cls_score_sp, name='cls_prob_sp')
            tf.reshape(cls_prob_sp, [1, self.num_classes])

            self.predictions["cls_score_H"] = cls_score_H
            self.predictions["cls_prob_H"] = cls_prob_H
            self.predictions["cls_score_O"] = cls_score_O
            self.predictions["cls_prob_O"] = cls_prob_O
            self.predictions["cls_score_sp"] = cls_score_sp
            self.predictions["cls_prob_sp"] = cls_prob_sp

            self.predictions["cls_prob_HO"] = cls_prob_sp * (cls_prob_O + cls_prob_H)

    # ===============
    # SP_GCN Module
    # ===============
    def data_bn_layer(self, x, in_channels):
        if self.data_bn is True:
            return slim.batch_norm(x)
        else:
            return x

    def st_gconv(self, input, A, in_channels, out_channels, kernel_size, initializer):
        """
        Args:
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int): Size of the  graph convolving kernel
            stride (int, optional): Stride of the temporal convolution. Default: 1
        Shape:
            - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}=1, V)` format
            - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
            - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}=1, V)` format
            - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

            where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
        Note:
            for image we don't consider temporal dimension
            the size of input is nkcv, and the size of A is ncw
        """
        t_kernel_size = 1
        t_stride = 1
        assert A.shape[0] == kernel_size  # kernel_size is the spatial kernel size, equals 3 when 'spatial'
        x = slim.conv2d(input,
                        out_channels*kernel_size,
                        kernel_size=(t_kernel_size, 1), stride=t_stride, padding='VALID',
                        activation_fn=tf.nn.relu
                        )

        if in_channels == out_channels:
            residual_x = input
        else:
            residual_x = slim.conv2d(input,
                                 out_channels,
                                 kernel_size=(t_kernel_size, 1), stride=t_stride, padding='VALID',
                                 activation_fn=tf.nn.relu)

        x = tf.reshape(x, [-1, 1, self.num_nodes, self.spatial_kernel_size, out_channels])  # ntvkc
        x = tf.transpose(x, perm=[0, 3, 4, 1, 2])  # nkctv
        x = tf.einsum('nkctv, kvw->nctw', x, A)
        x = tf.transpose(x, perm=[0, 2, 3, 1])  # ntwc

        x = x + residual_x

        return x

    # ======================
    # Pose Pooling Module
    # ======================
    def pose_pooling(self, x):
        # ====== pooling strategy 1: global pooling ========
        # '''
        human = x[:, :, :17, :]
        obj = x[:, :, 17:, :]
        '''
        x = slim.avg_pool2d(x, [1, self.num_nodes])  # [N, 1, 1, 256]
        x = tf.reshape(x, [-1, 256])
        x = slim.fully_connected(x, self.num_fc2, scope='fc_spgcn')
        x = slim.dropout(x, keep_prob=self.keep_prob, is_training=self.is_training, scope='drop_x')
        '''
        hnodes = tf.reshape(slim.avg_pool2d(human, [1, 17]), [-1, 256])
        hnodes = slim.fully_connected(hnodes, self.num_fc2, scope='fc_hnodes')
        hnodes = slim.dropout(hnodes, keep_prob=self.keep_prob, is_training=self.is_training, scope='drop_hnodes')

        onodes = tf.reshape(slim.avg_pool2d(obj, [1, 2]), [-1, 256])
        onodes = slim.fully_connected(onodes, self.num_fc2, scope='fc_onodes')
        onodes = slim.dropout(onodes, keep_prob=self.keep_prob, is_training=self.is_training, scope='drop_onodes')
        # '''

        # ==== human-object dividion strategy 2 =========
        '''
        human = x[:, :, :17, :]
        object = x[:, :, 17:, :]
        human_x = tf.reshape(slim.avg_pool2d(human, [1, 17]), [-1, 256])
        object_x = tf.reshape(slim.avg_pool2d(object, [1, 2]), [-1, 256])
        human_x = slim.fully_connected(human_x, 1024, scope='fc_huamn')
        object_x = slim.fully_connected(object_x, 1024, scope='fc_object')
        fc_human = slim.dropout(human_x, keep_prob=self.keep_prob, is_training=self.is_training,
                                scope='dropout_human')
        fc_object = slim.dropout(object_x, keep_prob=self.keep_prob, is_training=self.is_training,
                                 scope='dropout_object')
        hp = tf.concat([fc_human, fc_object], axis=1)  # 1024 * 2
        '''

        # ======= body-part division strategy 3  =======
        # N, 1, 19, 256
        '''
        head = x[:, :, :5, :]
        object = x[:, :, 17:, :]
        x = tf.transpose(x, perm=[2, 0, 1, 3])  # 19, N, 1, 256
        left_hand = tf.transpose(tf.gather(x, [5, 7, 9]), perm=[1, 2, 0, 3])   # N, 1, 5, 256
        right_hand = tf.transpose(tf.gather(x, [6, 8, 10]), perm=[1, 2, 0, 3])
        left_foot = tf.transpose(tf.gather(x, [11, 13, 15]), perm=[1, 2, 0, 3])
        right_foot = tf.transpose(tf.gather(x, [12, 14, 16]), perm=[1, 2, 0, 3])
        assert left_hand.shape[-1] == 256 and left_hand.shape[1] == 1

        # N, 256
        part1 = tf.reshape(slim.avg_pool2d(tf.concat([head, object], axis=2), [1, 5+2]), [-1, 256])
        part2 = tf.reshape(slim.avg_pool2d(tf.concat([left_hand, object], axis=2), [1, 3+2]), [-1, 256])
        part3 = tf.reshape(slim.avg_pool2d(tf.concat([right_hand, object], axis=2), [1, 3+2]), [-1, 256])
        part4 = tf.reshape(slim.avg_pool2d(tf.concat([left_foot, object], axis=2), [1, 3+2]), [-1, 256])
        part5 = tf.reshape(slim.avg_pool2d(tf.concat([right_foot, object], axis=2), [1, 3+2]), [-1, 256])
        x = tf.concat([part1, part2, part3, part4, part5], axis=1)  # N, 256 * 5
        hp = slim.fully_connected(x, 1024, scope='fc_spgcn')
        '''

        # ====== object division strategy 4 ================
        '''
        human = x[:, :, :17, :]
        objectNodeA = tf.reshape(x[:, :, 17, :], [-1, 1, 1, 256])
        objectNodeB = tf.reshape(x[:, :, 18, :], [-1, 1, 1, 256])
        x1 = tf.concat([human, objectNodeA], axis=2)  # N, 1, 18, 256
        x1 = tf.reshape(slim.avg_pool2d(x1, [1, 18]), [-1, 256])  # N, 256
        fc_x1 = slim.fully_connected(x1, 512, scope='fc_x1')
        fc_x1 = slim.dropout(fc_x1, keep_prob=self.keep_prob, scope='fc1_dropout')
        x2 = tf.concat([human, objectNodeB], axis=2)  # N, 1, 18, 256
        x2 = tf.reshape(slim.avg_pool2d(x2, [1, 18]), [-1, 256])
        fc_x2 = slim.fully_connected(x2, 512, scope='fc_x2')
        fc_x2 = slim.dropout(fc_x2, keep_prob=self.keep_prob, scope='fc2_dropout')
        hp = tf.concat([fc_x1, fc_x2], axis=1)
        '''

        return x, hnodes, onodes

    # ======================
    # Pose Attention Module
    # ======================
    def pose_attention(self, nodes_feat, semantic):
        """
        :param pose_feat:  [N, 1, 19, 256]
        :param semantic: [N, 1024]
        :return:
        """
        pose = nodes_feat[:, :, :17, :]
        object = nodes_feat[:, :, 17:, :]
        Ws = slim.fully_connected(semantic, 256, scope='fc_ws')  # N, 256
        score = tf.einsum('ntkc, nc->ntk', pose, Ws)  # [N, 1, 17]
        att_weights = tf.nn.softmax(score, axis=2)
        att_pose = tf.einsum('ntkc, ntk->ntkc', pose, att_weights)  # N, 1, 17, 256
        new_nodes_feat = tf.concat([att_pose, object], axis=2)
        return new_nodes_feat

    # ==================
    # Body Part Module
    # ==================
    def part_attention(self, bodypart):
        # input: N, 17, 1024
        att_part_fc1 = slim.fully_connected(self.semantic, 256, scope='att_part_fc1')
        att_part_fc2 = slim.fully_connected(att_part_fc1, 17, scope='att_part_fc2')  # N, 17
        att = tf.nn.softmax(att_part_fc2, axis=1)
        att_part = tf.einsum('nkc, nk->nkc', bodypart, att)  # N, 17, 1024
        att_part = tf.reduce_sum(att_part, axis=1)  # N, 1024
        return att_part
    
    # ======================
    # Network Build Modules
    # ======================
    def build_network(self, is_training):
        """
        Args:
            self.edge_importance_weighting (bool):
                If ``True``, adds a learnable importance weighting to the edges of the graph
        Shape:
            - Input: (N, in_channels, T_{in}=1, V_{in}, M_{in}=1)
            - Output: the same iwth input
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence(=1 for the image)
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.(=1)
        """
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

        # batch normalization
        # x = self.data_bn_layer(x, C)  # pair_num, 3, 19

        # self.Gnodes.shape = N, C, T, V, M
        x = tf.transpose(self.Gnodes, perm=[0, 4, 2, 3, 1])  # N, M, T, V, C
        x = tf.reshape(x, [-1, 1, self.num_nodes, self.c])

        spatial_kernel_size = self.A.shape[0]
        kernel_size = spatial_kernel_size

        # ----------------
        # SP-GCN_Network
        # ----------------
        if self.posegraph or self.bi_posegraph:
            with tf.variable_scope('sp_gcn') as scope:
                x = self.st_gconv(x, self.A, 3, 64, kernel_size,
                                  initializer)  # input_channel -> 64, input=input_channel*spatial_channel
                x = self.st_gconv(x, self.A, 64, 64, kernel_size, initializer)  # 64 -> 64
                x = self.st_gconv(x, self.A, 64, 64, kernel_size, initializer)  # 64 -> 64
                x = self.st_gconv(x, self.A, 64, 64, kernel_size, initializer)  # 64 -> 64
                x = self.st_gconv(x, self.A, 64, 128, kernel_size, initializer)  # 64 -> 128
                x = self.st_gconv(x, self.A, 128, 128, kernel_size, initializer)  # 128 -> 128
                x = self.st_gconv(x, self.A, 128, 128, kernel_size, initializer)  # 128 -> 128
                x = self.st_gconv(x, self.A, 128, 256, kernel_size, initializer)  # 128 -> 256
                x = self.st_gconv(x, self.A, 256, 256, kernel_size, initializer)  # 256 -> 256
                x = self.st_gconv(x, self.A, 256, 256, kernel_size, initializer)  # 256 -> 256
                # x = self.pose_attention(x, self.semantic)
                # [N, 1, 19, 256]

                # --------------------
                # Pose Pooling Module
                # --------------------
                allnodes, hnodes, onodes = self.pose_pooling(x)
                self.predictions['all_nodes'] = allnodes
                self.predictions['hnodes'] = hnodes
                self.predictions['onodes'] = onodes

        # -----------------------
        # ResNet Backbone Module
        # =----------------------
        head = self.head
        sp = self.sp_to_head()

        pool5_H = self.crop_pool_layer(head, self.H_boxes, 'Crop_H', cfg.POOLING_SIZE)
        pool5_O = self.crop_pool_layer(head, self.O_boxes[:self.H_num, :], 'Crop_O', cfg.POOLING_SIZE)
        self.predictions['pool5_H'] = pool5_H
        self.predictions['pool5_O'] = pool5_O

        # ------------
        # Body Part
        # ------------
        if self.bodypart:
            partboxes = tf.reshape(self.partboxes, [-1, 5])  # N, 17, 5 -> N*17, 5

            pool_part = self.crop_pool_layer(head, partboxes, 'Crop_H', 5)
            fc7_H, fc7_O, fc7_part = self.res5_part(pool5_H, pool5_O, pool_part, is_training, 'res5')  # N, 1024
            fc7_part = tf.reshape(fc7_part, [-1, 17, 2048])  # N, 17, 1024
            #fc7_att_part = self.part_attention(fc7_part)
            self.predictions['fc7_part'] = tf.reduce_sum(fc7_part, axis=1)  # N, 1024
        else:
            fc7_H, fc7_O = self.res5(pool5_H, pool5_O, sp, is_training, 'res5')

        # Phi
        head_phi = slim.conv2d(head, 512, [1, 1], scope='head_phi')
        # g
        head_g = slim.conv2d(head, 512, [1, 1], scope='head_g')

        Att_H = self.attention_pool_layer_H(head_phi, fc7_H, is_training, 'Att_H')
        Att_H = self.attention_norm_H(Att_H, 'Norm_Att_H')
        att_head_H = tf.multiply(head_g, Att_H)

        Att_O = self.attention_pool_layer_O(head_phi, fc7_O, is_training, 'Att_O')
        Att_O = self.attention_norm_O(Att_O, 'Norm_Att_O')
        att_head_O = tf.multiply(head_g, Att_O)

        pool5_SH = self.bottleneck(att_head_H, is_training, 'bottleneck', False)
        pool5_SO = self.bottleneck(att_head_O, is_training, 'bottleneck', True)

        # fc7_SH = 1024, fc7_SO=1024
        fc7_SH, fc7_SO, fc7_SHsp = self.head_to_tail(fc7_H, fc7_O, pool5_SH, pool5_SO, sp, is_training, 'fc_HO')

        # --------------
        # Classification
        # --------------
        if self.binary:
            fc9_binary = self.binary_discriminator(fc7_H, fc7_O, fc7_SH, fc7_SO, sp, is_training, 'fc_binary')
            self.binary_classification(fc9_binary, is_training, initializer, 'binary_classification')

        if not self.posegraph:
            self.region_classification_iCAN(fc7_SH, fc7_SO, fc7_SHsp,
                                            is_training, initializer, 'classification')
        else:
            self.region_classification(fc7_SH, fc7_SO, fc7_SHsp, allnodes, hnodes, onodes,
                                       is_training, initializer, 'classification')

    def create_architecture(self, is_training):
        self.build_network(is_training)
        for var in tf.trainable_variables():
            self.train_summaries.append(var)
        self.add_loss()
        layers_to_output = {}
        layers_to_output.update(self.losses)
        return layers_to_output

    def add_loss(self):

        with tf.variable_scope('LOSS') as scope:
            cls_score_sp = self.predictions["cls_score_sp"]
            label_HO = self.gt_class_HO
            label_binary = self.gt_binary_label

            sp_cross_entropy = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=label_HO, logits=cls_score_sp))
            self.losses['sp_cross_entropy'] = sp_cross_entropy

            if self.binary:
                cls_score_binary = self.predictions["cls_score_binary"]
                #binary_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_binary, logits=cls_score_binary))
                binary_cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=label_binary, logits=cls_score_binary,
                                                                                               pos_weight=2))
                self.losses['binary_cross_entropy'] = binary_cross_entropy
                loss = sp_cross_entropy + binary_cross_entropy
            else:
                loss = sp_cross_entropy

            cls_score_H = self.predictions["cls_score_H"]
            cls_score_O = self.predictions["cls_score_O"]
            H_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_HO[:self.H_num, :],
                                                                                     logits=cls_score_H[:self.H_num,
                                                                                            :]))
            O_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_HO[:self.H_num, :],
                                                                                     logits=cls_score_O[:self.H_num,
                                                                                            :]))

            self.losses['H_cross_entropy'] = H_cross_entropy
            self.losses['O_cross_entropy'] = O_cross_entropy
            loss += (H_cross_entropy + O_cross_entropy)

            self.losses['total_loss'] = loss

        return loss

    def train_step(self, sess, blobs, lr, train_op):
        gnodes = blobs['Gnodes']  # [N, 51+6]
        batchnum, vc = gnodes.shape
        c, v = 3, 17 + 2
        assert vc == v * c
        gnodes = gnodes.reshape(batchnum, c, 1, v, 1)

        feed_dict = {self.image: blobs['image'], self.head: blobs['head'],
                     self.partboxes: blobs['partboxes'],
                     self.H_boxes: blobs['H_boxes'], self.O_boxes: blobs['O_boxes'],
                     self.spatial: blobs['sp'],
                     self.gt_class_HO: blobs['gt_class_HO'],
                     self.H_num: blobs['H_num'],
                     self.Gnodes: gnodes,
                     self.semantic: blobs['semantic'], self.lr: lr,
                     self.gt_binary_label: blobs['binary_label']}
        loss,  _ = sess.run([self.losses['total_loss'],
                             #self.predictions,
                             #self.predictions['pool5_O'],
                                    #self.losses['H_cross_entropy'],
                                    #self.losses['O_cross_entropy'],
                                    #self.losses['sp_cross_entropy'],
                                    train_op], feed_dict=feed_dict)
        #print predictions['pool5_H'].shape
        #print predictions['pool_part'].shape
        #print pool5_O.shape
        return loss

    def test_image_HO(self, sess, image, blobs):
        gnodes = blobs['Gnodes']  # [1, 51+6]
        batchnum, vc = gnodes.shape
        c, v = 3, 17 + 2
        assert vc == v * c
        gnodes = gnodes.reshape(batchnum, c, 1, v, 1)
        feed_dict = {self.image: image, self.head: blobs['head'],
                     #self.partboxes: blobs['partboxes'],
                     self.H_boxes: blobs['H_boxes'], self.O_boxes: blobs['O_boxes'],
                     self.spatial: blobs['sp'], self.H_num: blobs['H_num'],
                     self.semantic: blobs['semantic'],
                     self.Gnodes: gnodes}
        cls_prob_HO = sess.run([self.predictions['cls_prob_HO']], feed_dict=feed_dict)
        return cls_prob_HO