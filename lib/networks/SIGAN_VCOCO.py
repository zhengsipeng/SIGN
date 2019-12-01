import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from utils.config import cfg
from Graph_builder import Graph

def resnet_arg_scope(is_training=True, batch_norm_decay=0.997, batch_norm_epsilon=1e-5, batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': ops.GraphKeys.UPDATE_OPS}
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


class SIGAN():
    def __init__(self,
                 num_sgnodes=2+17,
                 num_agnodes=6+17,
                 is_training=True,
                 use_skebox=False,
                 use_bodypart=False,
                 use_pm=False,
                 use_u=False,
                 use_sg=False,
                 use_sg_att=False,
                 use_ag=False,
                 use_ag_att=False,
                 use_binary=False,
                 use_Hsolo=False):

        # Control the network architecture
        self.use_skebox = use_skebox  # whether use skeleton box
        self.use_bp = use_bodypart    # whether use body part
        self.use_pm = use_pm          # whether use pose map
        self.use_u = use_u            # whether use union box
        self.use_sg = use_sg          # whether use spatial graph
        self.use_sg_att = use_sg_att  # whether use spatial graph attention
        self.use_ag = use_ag          # whether use appearance graph attention
        self.use_ag_att = use_ag_att  # whether use appearance attention
        self.use_binary = use_binary      # whether train binary module
        self.use_Hsolo = use_Hsolo

        # Annotation feed
        self.gt_binary_label = tf.placeholder(tf.float32, shape=[None, 1], name='gt_binary_label')
        self.gt_class_H = tf.placeholder(tf.float32, shape=[None, 29], name='gt_class_H')
        self.gt_class_HO = tf.placeholder(tf.float32, shape=[None, 29], name='gt_class_HO')
        self.gt_class_sp = tf.placeholder(tf.float32, shape=[None, 29], name='gt_class_sp')
        self.Mask_HO = tf.placeholder(tf.float32, shape=[None, 29], name='HO_mask')
        self.Mask_H = tf.placeholder(tf.float32, shape=[None, 29], name='H_mask')
        self.Mask_sp = tf.placeholder(tf.float32, shape=[None, 29], name='sp_mask')

        # Training utils
        self.predictions = {}
        self.losses = {}
        self.lr = tf.placeholder(tf.float32)
        self.num_binary = 1  # existence of HOI (0 or 1)
        self.num_classes = 29
        self.is_training = is_training
        #self.keep_prob = cfg.TRAIN_DROP_OUT_BINARY if self.is_training else 1
        self.keep_prob = 0.8 if self.is_training else 1

        # Training data feed
        self.image = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='image')
        self.head = tf.placeholder(tf.float32, shape=[1, None, None, 1024], name='head')  # extract head directly
        self.H_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='Hsp_boxes')
        self.O_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='O_boxes')
        self.U_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='U_boxes')
        self.skeboxes = tf.placeholder(tf.float32, shape=[None, 17, 5], name='part_boxes')
        self.bodyparts = tf.placeholder(tf.float32, shape=[None, 6, 5], name='bodypart_boxes')
        self.spatial = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='sp')
        self.H_num = tf.placeholder(tf.int32)

        # ResNet backbone
        self.scope = 'resnet_v1_50'
        self.num_fc = 1024
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
        # Spatial GCN
        self.num_SGnodes = num_sgnodes
        self.SGraph = Graph(num_node=self.num_SGnodes)
        self.ori_As = tf.convert_to_tensor(self.SGraph.ori_A.astype(np.float32))
        self.As = tf.convert_to_tensor(self.SGraph.A.astype(np.float32))  # list for partition, N, V, V
        self.spatial_kernel_size = self.As.shape[0]
        self.SGinput = tf.placeholder(tf.float32,  shape=[None, 3, 1, self.num_SGnodes, 1],
                                      name='Gnodes')  # [N, C, T, V, M] = [N, 3, 1, 19, 1]
        # Appearance GCN
        self.num_AGnodes = num_agnodes
        self.AGraph = Graph(num_node=self.num_AGnodes)
        self.ori_Aa = tf.convert_to_tensor(self.AGraph.ori_A.astype(np.float32))
        self.Aa = tf.convert_to_tensor(self.AGraph.A.astype(np.float32))  # list for partition, N, V, V

    def sp_to_head(self):
        with tf.variable_scope(self.scope, self.scope):
            conv1_sp = slim.conv2d(self.spatial[:, :, :, :2], 64, [5, 5], padding='VALID', scope='conv1_sp')
            pool1_sp = slim.max_pool2d(conv1_sp, [2, 2], scope='pool1_sp')
            conv2_sp = slim.conv2d(pool1_sp, 32, [5, 5], padding='VALID', scope='conv2_sp')
            pool2_sp = slim.max_pool2d(conv2_sp, [2, 2], scope='pool2_sp')
            pool2_flat_sp = slim.flatten(pool2_sp)  # 8194
            fc_sp = slim.fully_connected(pool2_flat_sp, 1024, scope='fc_sp')
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

    def res5(self, pool5, is_training, reuse):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            fc7, _ = resnet_v1.resnet_v1(pool5, self.blocks[-2:-1],
                                         global_pool=False,
                                         include_root_block=False,
                                         reuse=reuse,
                                         scope=self.scope)
            fc7 = tf.reduce_mean(fc7, axis=[1, 2])
        return fc7

    def head_to_tail(self, fc7_H, fc7_O, pool5_SH, pool5_SO, is_training):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            fc7_SH = tf.reduce_mean(pool5_SH, axis=[1, 2])
            fc7_SO = tf.reduce_mean(pool5_SO, axis=[1, 2])

            Concat_SH = tf.concat([fc7_H[: self.H_num], fc7_SH], 1)
            fc8_SH = slim.fully_connected(Concat_SH, self.num_fc, scope='fc8_SH')
            fc8_SH = slim.dropout(fc8_SH, keep_prob=self.keep_prob, is_training=is_training, scope='dropout8_SH')
            fc9_SH = slim.fully_connected(fc8_SH, self.num_fc, scope='fc9_SH')
            fc9_SH = slim.dropout(fc9_SH, keep_prob=self.keep_prob, is_training=is_training, scope='dropout9_SH')

            Concat_SO = tf.concat([fc7_O, fc7_SO], 1)
            fc8_SO = slim.fully_connected(Concat_SO, self.num_fc, scope='fc8_SO')
            fc8_SO = slim.dropout(fc8_SO, keep_prob=0.5, is_training=is_training, scope='dropout8_SO')
            fc9_SO = slim.fully_connected(fc8_SO, self.num_fc, scope='fc9_SO')
            fc9_SO = slim.dropout(fc9_SO, keep_prob=0.5, is_training=is_training, scope='dropout9_SO')
        return fc9_SH, fc9_SO

    def bottleneck(self, bottom, name, reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            head_bottleneck = slim.conv2d(bottom, 1024, [1, 1], scope=name)
        return head_bottleneck

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
            self.predictions['cls_prob'] = cls_prob_binary * self.predictions['cls_prob']

    def region_classification(self, fc_HOsp, is_training, initializer, name):
        with tf.variable_scope(name) as scope:
            if self.use_u:
                fc_HOsp = tf.concat([fc_HOsp, self.predictions['fc_U']], axis=1)
            if self.use_sg:
                fc_HOsp = tf.concat([fc_HOsp, self.predictions['sgcn_out']], axis=1)  # + 512
            if self.use_ag:
                fc_HOsp = tf.concat([fc_HOsp, self.predictions['agcn_out']], axis=1)  # + 1024
            if self.use_pm:
                fc_HOsp = tf.concat([fc_HOsp, self.predictions['pm_out']], axis=1)  # + 1024
            if self.use_skebox:
                fc_HOsp = tf.concat([fc_HOsp, self.predictions['skbox_out']], axis=1)  # + 1024
            if self.use_bp:
                fc_HOsp = tf.concat([fc_HOsp, self.predictions['bp_out']], axis=1)  # + 1024

            fc_HOsp = slim.fully_connected(fc_HOsp, 1024, scope='fc_HOsp1')
            fc_HOsp = slim.dropout(fc_HOsp, keep_prob=self.keep_prob, scope='dropout_HOsp1')
            fc_HOsp = slim.fully_connected(fc_HOsp, 1024, scope='fc_HOsp2')
            fc_HOsp = slim.dropout(fc_HOsp, keep_prob=self.keep_prob, scope='dropout_HOsp2')

            cls_score = slim.fully_connected(fc_HOsp, self.num_classes, weights_initializer=initializer,
                                             trainable=is_training, activation_fn=None, scope='cls_score')
            cls_prob = tf.nn.sigmoid(cls_score, name='cls_prob')
            tf.reshape(cls_prob, [1, self.num_classes])
            self.predictions["cls_score"] = cls_score
            self.predictions["cls_prob"] = cls_prob

            if self.use_Hsolo:
                cls_score_H = slim.fully_connected(self.predictions['H_solo'], self.num_classes,
                                                   weights_initializer=initializer,
                                                   trainable=is_training,
                                                   activation_fn=None, scope='cls_score_H')
                cls_prob_H = tf.nn.sigmoid(cls_score_H, name='cls_prob_H')
                tf.reshape(cls_prob_H, [1, self.num_classes])
                self.predictions["cls_score_H"] = cls_score_H
                self.predictions["cls_prob_H"] = cls_prob_H

    def gconv(self, input, A, in_channels, out_channels, kernel_size, name, alen=256, use_att=True, SA='spatial'):
        if SA == 'spatial':
            num_nodes = self.num_SGnodes
            ori_A = self.ori_As
        else:
            num_nodes = self.num_AGnodes
            ori_A = self.ori_Aa
        with tf.variable_scope(name) as scope:
            assert A.shape[0] == kernel_size  # kernel_size is the spatial kernel size, equals 3
            x = slim.conv2d(input, out_channels * kernel_size,
                            kernel_size=(1, 1), stride=1, padding='VALID',
                            activation_fn=tf.nn.relu)
            x = tf.reshape(x, [-1, 1, num_nodes, self.spatial_kernel_size, out_channels])  # ntvkc
            x = tf.transpose(x, perm=[0, 3, 4, 1, 2])  # nkctv

            if use_att:
                # input N, 1, k, c
                attx = tf.reshape(input, [-1, num_nodes, in_channels])  # N, K, C
                h = slim.fully_connected(attx, alen, activation_fn=None, scope='att_fc1')  # N, K, attC
                a_input = tf.concat([
                    tf.reshape(tf.tile(h, [1, 1, num_nodes]), [-1, num_nodes * num_nodes, alen]),
                    tf.tile(h, [1, num_nodes, 1])], axis=2)
                a_input = tf.reshape(a_input, [-1, num_nodes, num_nodes, 2 * alen])
                attention = tf.squeeze(slim.fully_connected(a_input, 1, activation_fn=None, scope='att_fc2'), axis=3)
                attention = tf.nn.leaky_relu(attention)
                attention = tf.nn.softmax(attention, axis=2)  # N, K, K
                ori_A = tf.tile(tf.expand_dims(ori_A, 0), (self.H_num, 1, 1))
                norm_att = tf.multiply(attention, ori_A)  # N, k, k
                attention = tf.nn.softmax(norm_att, axis=2)  # N, k, k

                attention = tf.tile(tf.expand_dims(attention, 1), (1, kernel_size, 1, 1))  # N, 1, k, k -> N, 3, k, k
                att_A = tf.einsum('nkvw, kvw->nkvw', attention, A)
                x = tf.einsum('nkctv, nkvw->nctw', x, att_A)
                x = tf.nn.elu(x)
            else:
                x = tf.einsum('nkctv, kvw->nctw', x, A)
            x = tf.transpose(x, perm=[0, 2, 3, 1])  # ntwc

            if in_channels == out_channels:
                residual_x = input
            else:
                residual_x = slim.conv2d(input, out_channels,
                                         kernel_size=(1, 1), stride=1, padding='VALID', activation_fn=tf.nn.relu)
            x = x + residual_x
            return x

    def spatial_GCN(self, input, use_att=True):
        with tf.variable_scope('sp_gcn') as scope:
            x = tf.transpose(input, perm=[0, 4, 2, 3, 1])  # N, M, T, V, C
            x = tf.reshape(x, [-1, 1, self.num_SGnodes, 3])
            spatial_kernel_size = self.As.shape[0]
            kernel_size = spatial_kernel_size
            x = self.gconv(x, self.As, 3, 64, kernel_size, 'block1', use_att)
            x = self.gconv(x, self.As, 64, 64, kernel_size, 'block2', use_att)  # 64 -> 64
            x = self.gconv(x, self.As, 64, 64, kernel_size, 'block3', use_att)  # 64 -> 64
            x = self.gconv(x, self.As, 64, 64, kernel_size, 'block4', use_att)  # 64 -> 64
            x = self.gconv(x, self.As, 64, 128, kernel_size, 'block5', use_att)  # 64 -> 128
            x = self.gconv(x, self.As, 128, 128, kernel_size, 'block6', use_att)  # 128 -> 128
            x = self.gconv(x, self.As, 128, 128, kernel_size, 'block7', use_att)  # 128 -> 128
            x = self.gconv(x, self.As, 128, 256, kernel_size, 'block8', use_att)  # 128 -> 256
            x = self.gconv(x, self.As, 256, 256, kernel_size, 'block9', use_att)  # 256 -> 256
            x = self.gconv(x, self.As, 256, 256, kernel_size, 'block10', use_att)  # 256 -> 256
            x = tf.reshape(x, [-1, self.num_SGnodes*256])
            x = slim.fully_connected(x, 512, scope='fc_sg')  # concat 256 * 19 -> 512
            x = slim.dropout(x, keep_prob=self.keep_prob, scope='dropout_sg')
            return x

    def appearance_GCN(self, input, use_att=True):
        with tf.variable_scope('a_gcn') as scope:
            x = tf.transpose(input, perm=[0, 4, 2, 3, 1])  # N, M, T, V, C
            x = tf.reshape(x, [-1, 1, self.num_AGnodes, 256])
            spatial_kernel_size = self.Aa.shape[0]
            kernel_size = spatial_kernel_size
            x = self.gconv(x, self.Aa, 256, 256, kernel_size, 'block1', use_att)
            x = self.gconv(x, self.Aa, 256, 512, kernel_size, 'block2', use_att)
            x = self.gconv(x, self.Aa, 512, 512, kernel_size, 'block3', use_att)
            x = self.gconv(x, self.Aa, 512, 512, kernel_size, 'block4', use_att)
            x = tf.reshape(x, [-1, 17+6+1, 512])
            x = tf.reshape(x[:, 17:17+6, :], [-1, self.num_AGnodes*512])
            x = slim.fully_connected(x, 1024, scope='fc_ag')  # concat 512 * 6 -> 1024
            x = slim.dropout(x, keep_prob=self.keep_prob, scope='dropout_ag')
        return x

    def build_network(self, is_training):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        head = self.head  # directly extract head features of backbone
        sp = self.sp_to_head()

        pool_H = self.crop_pool_layer(head, self.H_boxes, 'Crop_H', cfg.POOLING_SIZE)
        pool_O = self.crop_pool_layer(head, self.O_boxes, 'Crop_O', cfg.POOLING_SIZE)
        fc_H = self.res5(pool_H, is_training, False)  # 2048
        fc_O = self.res5(pool_O, is_training, True)   # 2048

        fc1_H = slim.fully_connected(fc_H, self.num_fc, scope='fc1_H')
        fc1_H = slim.dropout(fc1_H, keep_prob=self.keep_prob, scope='drop1_H')
        fc2_H = slim.fully_connected(fc1_H, self.num_fc, scope='fc2_H')
        fc2_H = slim.dropout(fc2_H, keep_prob=self.keep_prob, scope='drop2_H')
        fc1_O = slim.fully_connected(fc_O, self.num_fc, scope='fc1_O')
        fc1_O = slim.dropout(fc1_O, keep_prob=self.keep_prob, scope='drop1_O')
        fc2_O = slim.fully_connected(fc1_O, self.num_fc, scope='fc2_O')
        fc2_O = slim.dropout(fc2_O, keep_prob=self.keep_prob, scope='drop2_O')

        if self.use_u:
            pool_U = self.crop_pool_layer(head, self.U_boxes, 'Crop_U', cfg.POOLING_SIZE)
            fc_U = self.res5(pool_U, is_training, True)
            fc1_U = slim.fully_connected(fc_U, self.num_fc, scope='fc1_U')
            fc1_U = slim.dropout(fc1_U, keep_prob=self.keep_prob, scope='drop1_U')
            fc2_U = slim.fully_connected(fc1_U, self.num_fc, scope='fc2_U')
            fc2_U = slim.dropout(fc2_U, keep_prob=self.keep_prob, scope='drop2_U')
            self.predictions['fc_U'] = fc2_U
        # whether use spatial GCN
        if self.use_sg:
            x = self.spatial_GCN(self.SGinput, self.use_sg_att)  # input N, C, T, V, M
            self.predictions['sgcn_out'] = x
        # whether use appearance GCN
        if self.use_ag:
            # body node input
            bodyparts = tf.reshape(self.bodyparts, [-1, 5])  # N, 6, 5 -> N*6, 5
            pool_bp = self.crop_pool_layer(head, bodyparts, 'Crop_part', 5)
            fc1_bp = slim.fully_connected(pool_bp, 256, scope='fc1_bp')
            # skeleton input
            skeboxes = tf.reshape(self.skeboxes[:self.H_num], [-1, 5])  # N*17, 5
            pool5_skbox = self.crop_pool_layer(head, skeboxes, 'Crop_SK', 5)
            fc_skbox = self.res5(pool5_skbox, is_training, True)  # N*17, channel
            fc1_sk = slim.fully_connected(fc_skbox, 256, scope='fc1_sk')

            bp_in = tf.reshape(fc1_bp, [-1, 6, 256])
            ske_in = tf.reshape(fc1_sk, [-1, 17, 256])
            o_in = tf.reshape(slim.fully_connected(fc_O, 256, scope='ag_o_fc'), [-1, 1, 256])
            agraph_x = tf.concat([ske_in, bp_in, o_in], axis=1)
            x = self.appearance_GCN(agraph_x, self.use_ag_att)
            self.predictions['agcn_out'] = x

            # whether use bodypart
            if self.use_bp:
                fc1_bp = slim.dropout(fc1_bp, keep_prob=self.keep_prob, scope='dropout1_bp')
                fc2_bp = tf.reshape(fc1_bp, [-1, 6 * 256])
                fc2_bp = slim.fully_connected(fc2_bp, 1024, scope='fc2_bp')
                fc2_bp = slim.dropout(fc2_bp, keep_prob=self.keep_prob, scope='dropout2_bp')
                self.predictions['bp_out'] = fc2_bp
            # whether use skeleton box
            if self.use_skebox:
                fc1_sk = slim.dropout(fc1_sk, keep_prob=self.keep_prob, scope='dropout1_sk')
                fc1_sk = tf.reshape(fc1_sk, [-1, 17 * 256])
                fc2_sk = slim.fully_connected(fc1_sk, 1024, scope='fc2_sk')
                fc2_sk = slim.dropout(fc2_sk, keep_prob=self.keep_prob, scope='dropout2_sk')
                fc_skbox = tf.reshape(slim.avg_pool2d(fc2_sk, [1, 17]), [-1, 2048])
                self.predictions['skbox_out'] = fc_skbox
        # whether use pose map
        if self.use_pm:
            conv1_pm = slim.conv2d(self.spatial[:, :, :, 2:], 32, [5, 5], padding='VALID', scope='conv1_pm')
            pool1_pm = slim.max_pool2d(conv1_pm, [2, 2], scope='pool1_pm')
            conv2_pm = slim.conv2d(pool1_pm, 16, [5, 5], padding='VALID', scope='conv2_pm')
            pool2_pm = slim.max_pool2d(conv2_pm, [2, 2], scope='pool2_pm')
            pool2_flat_pm = slim.flatten(pool2_pm)
            pm_fc = slim.fully_connected(pool2_flat_pm, 1024, scope='pm_fc')
            pm_fc = slim.dropout(pm_fc, keep_prob=self.keep_prob, is_training=is_training, scope='dropout_pm')
            self.predictions['pm_out'] = pm_fc

        if self.use_Hsolo:
            head_phi = slim.conv2d(head, 512, [1, 1], scope='head_phi')
            head_g = slim.conv2d(head, 512, [1, 1], scope='head_g')
            Att_H = self.attention_pool_layer_H(head_phi, fc_H[:self.H_num, :], is_training, 'Att_H')
            Att_H = self.attention_norm_H(Att_H, 'Norm_Att_H')
            att_head_H = tf.multiply(head_g, Att_H)
            pool5_SH = self.bottleneck(att_head_H, 'bottleneck', False)
            fc7_SH = tf.reduce_mean(pool5_SH, axis=[1, 2])
            Concat_SH = tf.concat([fc_H[:self.H_num, :], fc7_SH[:self.H_num, :]], 1)
            fc8_SH = slim.fully_connected(Concat_SH, self.num_fc, scope='fc8_SH')
            fc8_SH = slim.dropout(fc8_SH, keep_prob=0.5, is_training=is_training, scope='dropout8_SH')
            fc9_SH = slim.fully_connected(fc8_SH, self.num_fc, scope='fc9_SH')
            self.predictions['H_solo'] = slim.dropout(fc9_SH, keep_prob=0.5, is_training=is_training, scope='dropout9_SH')

        fc_HOsp = tf.concat([fc2_H, fc2_O, sp], 1)
        self.region_classification(fc_HOsp, is_training, initializer, 'classification')

        if self.use_binary:
            # Remain iCAN components for TIN binary
            head_phi = slim.conv2d(head, 512, [1, 1], scope='head_phi')
            head_g = slim.conv2d(head, 512, [1, 1], scope='head_g')
            Att_H = self.attention_pool_layer_H(head_phi, fc_H[:self.H_num, :], is_training, 'Att_H')
            Att_H = self.attention_norm_H(Att_H, 'Norm_Att_H')
            att_head_H = tf.multiply(head_g, Att_H)
            Att_O = self.attention_pool_layer_O(head_phi, fc_O, is_training, 'Att_O')
            Att_O = self.attention_norm_O(Att_O, 'Norm_Att_O')
            att_head_O = tf.multiply(head_g, Att_O)
            pool5_SH = self.bottleneck(att_head_H, 'bottleneck', False)
            pool5_SO = self.bottleneck(att_head_O, 'bottleneck', True)
            fc7_SH, fc7_SO = self.head_to_tail(fc_H, fc_O, pool5_SH, pool5_SO, is_training)
            fc9_binary = self.binary_discriminator(fc_H, fc_O, fc7_SH, fc7_SO, sp, is_training, 'fc_binary')
            self.binary_classification(fc9_binary, is_training, initializer, 'binary_classification')

    def create_architecture(self, is_training):
        self.build_network(is_training)
        self.add_loss()
        layers_to_output = {}
        layers_to_output.update(self.losses)
        return layers_to_output

    def add_loss(self):
        with tf.variable_scope('LOSS') as scope:
            cls_score = self.predictions["cls_score"]
            label_H = self.gt_class_H
            label_HO = self.gt_class_HO
            label_binary = self.gt_binary_label
            H_mask = self.Mask_H
            HO_mask = self.Mask_HO
            HO_cross_entropy = tf.reduce_mean(
                tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_HO, logits=cls_score), HO_mask))
            self.losses['HO_cross_entropy'] = HO_cross_entropy
            loss = HO_cross_entropy

            if self.use_binary:
                cls_score_binary = self.predictions["cls_score_binary"]
                binary_cross_entropy = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=label_binary, logits=cls_score_binary))
                self.losses['binary_cross_entropy'] = binary_cross_entropy
                loss += (HO_cross_entropy + binary_cross_entropy)
            if self.use_Hsolo:
                cls_score_H = self.predictions["cls_score_H"]
                H_cross_entropy = tf.reduce_mean(
                    tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_H, logits=cls_score_H), H_mask))
                self.losses['H_cross_entropy'] = H_cross_entropy
                loss += 2 * H_cross_entropy

            self.losses['total_loss'] = loss
        return loss

    def train_step(self, sess, blobs, lr, train_op):
        # [N, 51+6]
        SGinput = blobs['SGinput'].reshape(-1, 3, 1, 17+2, 1)
        if self.num_SGnodes != 19:
            SGinput = SGinput[:, :, :, :self.num_SGnodes, :]
        feed_dict = {self.image: blobs['image'], self.head: blobs['head'], self.H_num: blobs['H_num'], self.lr: lr,
                     self.H_boxes: blobs['H_boxes'], self.O_boxes: blobs['O_boxes'], self.U_boxes: blobs['U_boxes'],
                     self.spatial: blobs['sp'], self.SGinput: SGinput,
                     self.skeboxes: blobs['skeboxes'], self.bodyparts: blobs['bodyparts'],
                     self.gt_class_HO: blobs['gt_class_HO'], self.gt_class_H: blobs['gt_class_H'],
                     self.gt_binary_label: blobs['gt_binary_label'],
                     self.Mask_H: blobs['Mask_H'], self.Mask_HO: blobs['Mask_HO']
                     }
        loss_cls_HO, loss, _ = sess.run([self.losses['HO_cross_entropy'],
                                         self.losses['total_loss'],
                                         train_op],
                                         feed_dict=feed_dict)
        return loss_cls_HO, loss

    def test_image_HO(self, sess, image, blobs):
        SGinput = blobs['SGinput'].reshape(-1, 3, 1, 17+2, 1)
        if self.num_SGnodes != 19:
            SGinput = SGinput[:, :, :, :self.num_SGnodes, :]
        feed_dict = {self.image: image, self.head: blobs['head'],
                     self.H_num: blobs['H_num'],
                     self.spatial: blobs['sp'],
                     self.skeboxes: blobs['skeboxes'], self.bodyparts: blobs['bodyparts'],
                     self.SGinput: SGinput,
                     self.H_boxes: blobs['H_boxes'], self.O_boxes: blobs['O_boxes'], self.U_boxes: blobs['U_boxes']
                     }
        cls_prob = sess.run([self.predictions["cls_prob"]], feed_dict=feed_dict)
        return cls_prob
