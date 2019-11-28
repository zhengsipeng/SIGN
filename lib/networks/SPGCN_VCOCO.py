import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from ult.config import cfg
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
                 num_nodes=17+2,
                 is_training=True,
                 num_fc=1024,
                 posetype=1,
                 bi_posegraph=False,
                 bodypoint=False,
                 bodypart=False,
                 binary=False,
                 posemap=False,
                 posegraph=False,
                 posegraph_att=False,
                 semantic=False,
                 latefusion=False,
                 H=False,
                 data_bn=True):
        self.predictions = {}
        self.train_summaries = []
        self.losses = {}
        self.lr = tf.placeholder(tf.float32)
        self.num_binary = 1  # existence of HOI (0 or 1)
        self.num_classes = 29

        self.gt_binary_label = tf.placeholder(tf.float32, shape=[None, 1], name='gt_binary_label')
        self.gt_class_H = tf.placeholder(tf.float32, shape=[None, 29], name='gt_class_H')
        self.gt_class_HO = tf.placeholder(tf.float32, shape=[None, 29], name='gt_class_HO')
        self.gt_class_sp = tf.placeholder(tf.float32, shape=[None, 29], name='gt_class_sp')
        self.Mask_HO = tf.placeholder(tf.float32, shape=[None, 29], name='HO_mask')
        self.Mask_H = tf.placeholder(tf.float32, shape=[None, 29], name='H_mask')
        self.Mask_sp = tf.placeholder(tf.float32, shape=[None, 29], name='sp_mask')
        self.is_training = is_training
        if self.is_training:
            self.keep_prob = cfg.TRAIN_DROP_OUT_BINARY
            self.keep_prob_tail = .5
        else:
            self.keep_prob = 1
            self.keep_prob_tail = 1

        self.image = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='image')
        self.head = tf.placeholder(tf.float32, shape=[1, None, None, 1024], name='head')
        self.Hsp_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='Hsp_boxes')  # num_pos 5
        self.O_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='O_boxes')  # num_pos 5
        self.U_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='U_boxes')
        self.pointboxes = tf.placeholder(tf.float32, shape=[None, 17, 5], name='part_boxes')
        self.partboxes = tf.placeholder(tf.float32, shape=[None, 6, 5], name='bodypart_boxes')
        self.semantic = tf.placeholder(tf.float32, shape=[None, 1024], name='semantic_feat')
        self.H_num = tf.placeholder(tf.int32)

        # Control the network architecture
        self.bodypoint = bodypoint
        self.bodypart = bodypart
        self.binary = binary
        self.posemap = posemap
        self.posegraph = posegraph
        self.posegraph_att = posegraph_att
        self.bi_posezhihugraph = bi_posegraph
        self.posetype = posetype
        self.latefusion = latefusion
        self.H = H
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
        self.graph = Graph(num_node=self.num_nodes, strategy=self.strategy)
        self.A = tf.convert_to_tensor(self.graph.A.astype(np.float32))  # [None, num_nodes, num_nodes]
        self.ori_A = tf.convert_to_tensor(self.graph.ori_A.astype(np.float32))
        self.spatial_kernel_size = self.A.shape[0]
        # [N, C, T, V, M] = [N, 3, 1, 19, 1]
        self.Gnodes = tf.placeholder(tf.float32, shape=[None, self.c, 1, self.num_nodes, 1], name='Gnodes')

        # ST_GCN
        self.depth_st_gcn_networks = 10

    def sp_to_head(self):
        with tf.variable_scope(self.scope, self.scope):
            conv1_sp = slim.conv2d(self.spatial[:, :, :, 0:2], 64, [5, 5], padding='VALID', scope='conv1_sp')
            pool1_sp = slim.max_pool2d(conv1_sp, [2, 2], scope='pool1_sp')
            conv2_sp = slim.conv2d(pool1_sp, 32, [5, 5], padding='VALID', scope='conv2_sp')
            pool2_sp = slim.max_pool2d(conv2_sp, [2, 2], scope='pool2_sp')
            pool2_flat_sp = slim.flatten(pool2_sp)
            fc_sp = slim.fully_connected(pool2_flat_sp, self.num_fc2, scope='fc_sp')
            fc_sp = slim.dropout(fc_sp, keep_prob=self.keep_prob, scope='dropout_fc_sp')
        return fc_sp

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

    def res5(self, pool5_H, pool5_O, is_training):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            fc7_H, _ = resnet_v1.resnet_v1(pool5_H,
                                           self.blocks[-1:],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=False,
                                           scope=self.scope)

            fc7_H = tf.reduce_mean(fc7_H, axis=[1, 2])

            fc7_O, _ = resnet_v1.resnet_v1(pool5_O,
                                           self.blocks[-2:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=False,
                                           scope=self.scope)

            fc7_O = tf.reduce_mean(fc7_O, axis=[1, 2])

        return fc7_H, fc7_O

    def res5_part(self, pool5_bp, is_training):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            fc7_bp, _ = resnet_v1.resnet_v1(pool5_bp,
                                           self.blocks[-2:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=True,
                                           scope=self.scope)

            fc7_bp = tf.reduce_mean(fc7_bp, axis=[1, 2])

        return fc7_bp

    def head_to_tail(self, fc7_H, fc7_O, pool5_SH, pool5_SO, is_training):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            fc7_SH = tf.reduce_mean(pool5_SH, axis=[1, 2])
            fc7_SO = tf.reduce_mean(pool5_SO, axis=[1, 2])

            Concat_SH = tf.concat([fc7_H[: self.H_num], fc7_SH], 1)
            fc8_SH = slim.fully_connected(Concat_SHsp, self.num_fc, scope='fc8_SH')
            fc8_SH = slim.dropout(fc8_SH, keep_prob=self.keep_prob_tail, is_training=is_training, scope='dropout8_SH')
            fc9_SH = slim.fully_connected(fc8_SH, self.num_fc, scope='fc9_SH')
            fc9_SH = slim.dropout(fc9_SH, keep_prob=self.keep_prob_tail, is_training=is_training, scope='dropout9_SH')

            Concat_SO = tf.concat([fc7_O, fc7_SO], 1)
            fc8_SO = slim.fully_connected(Concat_SO, self.num_fc, scope='fc8_SO')
            fc8_SO = slim.dropout(fc8_SO, keep_prob=0.5, is_training=is_training, scope='dropout8_SO')
            fc9_SO = slim.fully_connected(fc8_SO, self.num_fc, scope='fc9_SO')
            fc9_SO = slim.dropout(fc9_SO, keep_prob=0.5, is_training=is_training, scope='dropout9_SO')

            fc9_SHsp = tf.concat([Concat_SH, Concat_SO], 1)
        return fc9_SH, fc9_SO, fc9_SHsp

    def bottleneck(self, bottom, name, reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            head_bottleneck = slim.conv2d(bottom, 1024, [1, 1], scope=name)

        return head_bottleneck

    # ======================
    # Binary Classification
    # ======================
    def binary_discriminator(self, fc7_H, fc7_O, fc7_SH, fc7_SO, sp, is_training, name):
        with tf.variable_scope(name) as scope:
            conv1_pose_map = slim.conv2d(self.spatial[:, :, :, 2:], 32, [5, 5], padding='VALID', scope='conv1_pose_map')
            pool1_pose_map = slim.max_pool2d(conv1_pose_map, [2, 2], scope='pool1_pose_map')
            conv2_pose_map = slim.conv2d(pool1_pose_map, 16, [5, 5], padding='VALID', scope='conv2_pose_map')
            pool2_pose_map = slim.max_pool2d(conv2_pose_map, [2, 2], scope='pool2_pose_map')
            pool2_flat_pose_map = slim.flatten(pool2_pose_map)

            # fc7_H + fc7_SH + sp---fc1024---fc8_binary_1
            fc_binary_1 = tf.concat([fc7_H, fc7_SH], 1)  # [pos + neg, 3072]
            fc_binary_2 = tf.concat([fc7_O, fc7_SO], 1)  # [pos, 3072]

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

    def region_classification(self, fc7_H, fc7_O, fc_HOsp, sp, is_training, initializer, name):
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
                fc_HOsp = tf.concat([fc_HOsp, posemap_fc[:self.H_num]], axis=1)
            if self.posegraph:
                fc_HOsp = tf.concat([fc_HOsp, self.predictions['all_nodes'][:self.H_num]], axis=1)
            if self.bodypoint:
                fc8_bp = slim.fully_connected(self.predictions['fc7_bp'], 1024, scope='fc8_bp')
                fc8_bp = slim.dropout(fc8_bp, keep_prob=self.keep_prob, scope='dropout8_bp')
                fc9_bp = slim.fully_connected(fc8_bp, 1024, scope='fc9_bp')
                fc9_bp = slim.dropout(fc9_bp, keep_prob=self.keep_prob, scope='dropout9_bp')
                fc_HOsp = tf.concat([fc_HOsp, fc9_bp], axis=1)
            if self.bodypart:
                fc8_part = slim.fully_connected(self.predictions['fc7_part'], 1024, scope='fc8_part')
                fc8_part = slim.dropout(fc8_part, keep_prob=self.keep_prob, scope='dropout8_part')
                fc9_part = slim.fully_connected(fc8_part, 1024, scope='fc9_part')
                fc9_part = slim.dropout(fc9_part, keep_prob=self.keep_prob, scope='dropout9_part')
                fc_HOsp = tf.concat([fc_HOsp, fc9_part], axis=1)

            cls_score = slim.fully_connected(fc_HOsp, self.num_classes,
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_O')
            cls_prob = tf.nn.sigmoid(cls_score, name='cls_prob_O')
            tf.reshape(cls_prob, [1, self.num_classes])
            self.predictions["cls_score"] = cls_score
            self.predictions["cls_prob"] = cls_prob

            if self.latefusion:
                Concat_SHsp = tf.concat([fc7_H, fc7_O, sp], 1)
                Concat_SHsp = slim.fully_connected(Concat_SHsp, self.num_fc, scope='Concat_SHsp')
                Concat_SHsp = slim.dropout(Concat_SHsp, keep_prob=0.5, is_training=is_training, scope='dropout6_SHsp')
                fc7_SHsp = slim.fully_connected(Concat_SHsp, self.num_fc, scope='fc7_SHsp')
                fc7_SHsp = slim.dropout(fc7_SHsp, keep_prob=0.5, is_training=is_training, scope='dropout7_SHsp')
                cls_score_sp = slim.fully_connected(fc7_SHsp, self.num_classes,
                                                    weights_initializer=initializer,
                                                    trainable=is_training,
                                                    activation_fn=None, scope='cls_score_sp')
                cls_prob_sp = tf.nn.sigmoid(cls_score_sp, name='cls_prob_sp')
                tf.reshape(cls_prob_sp, [1, self.num_classes])
                self.predictions["cls_score_sp"] = cls_score_sp
                self.predictions["cls_prob_sp"] = cls_prob_sp
                self.predictions['cls_prob_HO_final'] = cls_prob * cls_prob_sp
            else:
                self.predictions["cls_prob_HO_final"] = cls_prob

    def st_gconv(self, input, A, in_channels, out_channels, kernel_size, name):
        with tf.variable_scope(name) as scope:
            t_kernel_size = 1
            t_stride = 1
            assert A.shape[0] == kernel_size  # kernel_size is the spatial kernel size, equals 3 when 'spatial'
            x = slim.conv2d(input,
                            out_channels * kernel_size,
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
            if not self.posegraph_att:
                x = tf.einsum('nkctv, kvw->nctw', x, A)
            else:
                # input N, 1, k, c
                attx = tf.reshape(input, [-1, self.num_nodes, in_channels])  # N, K, C
                h = slim.fully_connected(attx, 256, activation_fn=None, scope='att_fc1')  # N, K, attC
                a_input = tf.concat([
                    tf.reshape(tf.tile(h, [1, 1, self.num_nodes]), [-1, self.num_nodes * self.num_nodes, 256]),
                    tf.tile(h, [1, self.num_nodes, 1])], axis=2)
                a_input = tf.reshape(a_input, [-1, self.num_nodes, self.num_nodes, 2 * 256])
                attention = tf.squeeze(slim.fully_connected(a_input, 1, activation_fn=None, scope='att_fc2'), axis=3)
                attention = tf.nn.leaky_relu(attention)
                attention = tf.nn.softmax(attention, axis=2)  # N, K, K
                ori_A = tf.tile(tf.expand_dims(self.ori_A, 0), (self.H_num, 1, 1))
                norm_att = tf.multiply(attention, ori_A)  # N, k, k
                attention = tf.nn.softmax(norm_att, axis=2)  # N, k, k

                attention = tf.tile(tf.expand_dims(attention, 1), (1, kernel_size, 1, 1))  # N, 1, k, k -> N, 3, k, k
                att_A = tf.einsum('nkvw, kvw->nkvw', attention, A)
                x = tf.einsum('nkctv, nkvw->nctw', x, att_A)
                x = tf.nn.elu(x)
            x = tf.transpose(x, perm=[0, 2, 3, 1])  # ntwc
            x = x + residual_x
            return x

    def pose_pooling(self, x):
        x = tf.reshape(x, [-1, self.num_nodes*256])
        x = slim.fully_connected(x, 256, scope='posepool_fc')
        allnodes = slim.dropout(x, keep_prob=self.keep_prob, scope='posepoll_dropout')
        return allnodes

    def build_network(self, is_training):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        # self.Gnodes.shape = N, C, T, V, M
        x = tf.transpose(self.Gnodes, perm=[0, 4, 2, 3, 1])  # N, M, T, V, C
        x = tf.reshape(x, [-1, 1, self.num_nodes, self.c])

        spatial_kernel_size = self.A.shape[0]
        kernel_size = spatial_kernel_size

        if self.posegraph:
            with tf.variable_scope('sp_gcn') as scope:
                x = self.st_gconv(x[:self.H_num], self.A, 3, 64, kernel_size, 'blobck1')
                x = self.st_gconv(x, self.A, 64, 64, kernel_size, 'blobck2')  # 64 -> 64
                x = self.st_gconv(x, self.A, 64, 64, kernel_size, 'blobck3')  # 64 -> 64
                x = self.st_gconv(x, self.A, 64, 64, kernel_size, 'blobck4')  # 64 -> 64
                x = self.st_gconv(x, self.A, 64, 128, kernel_size, 'blobck5')  # 64 -> 128
                x = self.st_gconv(x, self.A, 128, 128, kernel_size, 'blobck6')  # 128 -> 128
                x = self.st_gconv(x, self.A, 128, 128, kernel_size, 'blobck7')  # 128 -> 128
                x = self.st_gconv(x, self.A, 128, 256, kernel_size, 'blobck8')  # 128 -> 256
                x = self.st_gconv(x, self.A, 256, 256, kernel_size, 'blobck9')  # 256 -> 256
                x = self.st_gconv(x, self.A, 256, 256, kernel_size, 'blobck10')  # 256 -> 256
                allnodes = self.pose_pooling(x)
                self.predictions['all_nodes'] = allnodes

        head = self.head
        sp = self.sp_to_head()

        pool5_H = self.crop_pool_layer(head, self.Hsp_boxes, 'Crop_H', cfg.POOLING_SIZE)
        pool5_O = self.crop_pool_layer(head, self.O_boxes, 'Crop_O', cfg.POOLING_SIZE)

        fc7_H, fc7_O = self.res5(pool5_H, pool5_O, is_training)

        if self.bodypoint:
            pointboxes = tf.reshape(self.pointboxes[:self.H_num], [-1, 5])  # N*17, 5
            pool5_bp = self.crop_pool_layer(head, pointboxes, 'Crop_BP', 5)
            fc7_bp = self.res5_part(pool5_bp, is_training)  # N*17, channel
            fc7_bp = tf.reshape(fc7_bp, [-1, 1, 17, 2048])
            fc7_bp = tf.reshape(slim.avg_pool2d(fc7_bp, [1, 17]), [-1, 2048])
            self.predictions['fc7_bp'] = fc7_bp

        if self.bodypart:
            partboxes = tf.reshape(self.partboxes[:self.H_num], [-1, 5])
            pool5_part = self.crop_pool_layer(head, partboxes, 'Crop_part', 5)
            fc7_p = self.res5_part(pool5_part, is_training)
            fc7_p = tf.reshape(fc7_p, [-1, 1, 6, 2048])
            fc7_p = tf.reshape(slim.avg_pool2d(fc7_p, [1, 6]), [-1, 2048])
            self.predictions['fc7_part'] = fc7_p

        head_phi = slim.conv2d(head, 512, [1, 1], scope='head_phi')
        head_g = slim.conv2d(head, 512, [1, 1], scope='head_g')
        Att_H = self.attention_pool_layer_H(head_phi, fc7_H[:self.H_num, :], is_training, 'Att_H')
        Att_H = self.attention_norm_H(Att_H, 'Norm_Att_H')
        att_head_H = tf.multiply(head_g, Att_H)
        Att_O = self.attention_pool_layer_O(head_phi, fc7_O, is_training, 'Att_O')
        Att_O = self.attention_norm_O(Att_O, 'Norm_Att_O')
        att_head_O = tf.multiply(head_g, Att_O)
        pool5_SH = self.bottleneck(att_head_H, 'bottleneck', False)
        pool5_SO = self.bottleneck(att_head_O, 'bottleneck', True)

        fc7_SH, fc7_SO, fc9_SHsp = self.head_to_tail(fc7_H, fc7_O, pool5_SH, pool5_SO, is_training)

        if self.binary:
            fc9_binary = self.binary_discriminator(fc7_H, fc7_O, fc7_SH, fc7_SO, sp, is_training, 'fc_binary')
            self.binary_classification(fc9_binary, is_training, initializer, 'binary_classification')

        self.region_classification(fc7_H, fc7_O, fc9_SHsp, sp, is_training, initializer, 'classification')

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
            cls_score = self.predictions["cls_score"]

            label_HO = self.gt_class_HO
            label_sp = self.gt_class_sp
            label_binary = self.gt_binary_label

            HO_mask = self.Mask_HO
            sp_mask = self.Mask_sp
            HO_cross_entropy = tf.reduce_mean(
                tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_HO, logits=cls_score[:self.H_num]),
                            HO_mask))
            self.losses['HO_cross_entropy'] = HO_cross_entropy
            loss = 2 * HO_cross_entropy

            if self.latefusion:
                cls_score_sp = self.predictions["cls_score_sp"]
                sp_cross_entropy = tf.reduce_mean(
                    tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_sp, logits=cls_score_sp), sp_mask))
                self.losses['sp_cross_entropy'] = sp_cross_entropy
                loss += sp_cross_entropy

            if self.binary:
                cls_score_binary = self.predictions["cls_score_binary"]
                binary_cross_entropy = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=label_binary, logits=cls_score_binary))
                self.losses['binary_cross_entropy'] = binary_cross_entropy
                loss += binary_cross_entropy

            self.losses['total_loss'] = loss

        return loss

    def train_step(self, sess, blobs, lr, train_op):
        gnodes = blobs['Gnodes']  # [N, 51+6]
        batchnum, vc = gnodes.shape
        c, v = 3, 17+2
        assert vc == v * c
        gnodes = gnodes.reshape(batchnum, c, 1, v, 1)

        if self.num_nodes != 19:
            gnodes = gnodes[:, :, :, :self.num_nodes, :]

        feed_dict = {self.image: blobs['image'],
                     self.Gnodes: gnodes,
                     self.head: blobs['head'],
                     self.Hsp_boxes: blobs['Hsp_boxes'],
                     self.O_boxes: blobs['O_boxes'], self.U_boxes: blobs['U_boxes'],
                     self.gt_class_H: blobs['gt_class_H'],
                     self.gt_class_HO: blobs['gt_class_HO'], self.Mask_H: blobs['Mask_H'],
                     self.Mask_HO: blobs['Mask_HO'], self.spatial: blobs['sp'],
                     self.lr: lr, self.Mask_sp: blobs['Mask_sp'],
                     self.pointboxes: blobs['pointboxes'], self.partboxes: blobs['partboxes'],
                     self.gt_class_sp: blobs['gt_class_sp'], self.H_num: blobs['H_num'],
                     self.gt_binary_label: blobs['binary_label']}

        loss_cls_HO, loss, _ = sess.run([self.losses['HO_cross_entropy'],
                                         self.losses['total_loss'],
                                         train_op],
                                         feed_dict=feed_dict)
        return loss_cls_HO, loss

    def test_image_HO(self, sess, image, blobs):
        gnodes = blobs['Gnodes']  # [N, 51+6]
        batchnum, vc = gnodes.shape
        c, v = 3, 17 + 2
        assert vc == v * c
        gnodes = gnodes.reshape(batchnum, c, 1, v, 1)

        if self.num_nodes != 19:
            gnodes = gnodes[:, :, :, :self.num_nodes, :]
        feed_dict = {self.image: image, self.head: blobs['head'],
                     self.pointboxes: blobs['pointboxes'], self.partboxes: blobs['partboxes'],
                     self.Gnodes: gnodes,
                     self.Hsp_boxes: blobs['H_boxes'], self.O_boxes: blobs['O_boxes'], self.U_boxes: blobs['U_boxes'],
                     self.spatial: blobs['sp'], self.H_num: blobs['H_num']}
        cls_prob_HO = sess.run([self.predictions["cls_prob_HO_final"]], feed_dict=feed_dict)

        return cls_prob_HO
