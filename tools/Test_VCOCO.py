# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen Gao, based on code from Zheqi he and Xinlei Chen
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import tensorflow as tf
import json
import argparse
import pickle
import os
from networks.SIGAN_VCOCO import SIGAN
from utils.vsrl_eval import VCOCOeval
from utils.config import cfg
from models.test_VCOCO import test_net
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser(description='Test an iCAN on HICO')
    parser.add_argument('--num_iteration', dest='iteration', help='Specify which weight to load', default=300000, type=int)
    parser.add_argument('--prior_flag', dest='prior_flag', help='whether use prior_flag or not', default=3, type=int)
    parser.add_argument('--model', dest='model', help='Select model', default='SIGAN_VCOCO', type=str)
    parser.add_argument('--object_thres', dest='object_thres', help='Object threshold', default=0.6, type=float)
    parser.add_argument('--human_thres', dest='human_thres', help='Human threshold', default=0.8, type=float)
    parser.add_argument('--skebox', dest='use_skebox', help='use_skebox or not', action='store_true', default=False)
    parser.add_argument('--bp', dest='use_bodypart', help='use_bodypart or not', action='store_true', default=False)
    parser.add_argument('--pm', dest='use_pm', help='use posemap or not', action='store_true', default=False)
    parser.add_argument('--u', dest='use_u', help='use union box or not', action='store_true', default=False)
    parser.add_argument('--sg', dest='use_sg', help='use posegraph or not', action='store_true', default=False)
    parser.add_argument('--sg_att', dest='use_sg_att', help='use Spatial posegraph or not', action='store_true', default=False)
    parser.add_argument('--ag', dest='use_ag', help='use Appearance posegraph or not', action='store_true', default=False)
    parser.add_argument('--ag_att', dest='use_ag_att', help='use posegraph or not', action='store_true', default=False)
    parser.add_argument('--bi', dest='use_binary', help='use binary or not', action='store_true', default=False)
    parser.add_argument('--tmp', dest='tmp', help='use tmp or not', action='store_true', default=False)
    parser.add_argument('--cuda', dest='cuda', help='cuda device id', default='0', type=str)
    parser.add_argument('--eval', dest='eval', help='use eval or not', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_VCOCO_with_pose.pkl', "rb"))
    prior_mask = pickle.load(open(cfg.DATA_DIR + '/' + 'prior_mask.pkl', "rb"))
    Action_dic = json.load(open(cfg.DATA_DIR + '/' + 'action_index.json'))
    Action_dic_inv = {y: x for x, y in Action_dic.iteritems()}

    vcocoeval = VCOCOeval(cfg.DATA_DIR + '/' + 'v-coco/data/vcoco/vcoco_test.json',
                          cfg.DATA_DIR + '/' + 'v-coco/data/instances_vcoco_all_2014.json',
                          cfg.DATA_DIR + '/' + 'v-coco/data/splits/vcoco_test.ids')
    weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    print('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(
        args.iteration) + ', path = ' + weight)

    output_file = cfg.ROOT_DIR + '/Results/' + str(args.iteration) + '_' + args.model + '.pkl'

    # init session
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    print('Use skebox: ' + str(args.use_skebox))
    print('Use bodypart: ' + str(args.use_bodypart))
    print('Use posemap: ' + str(args.use_pm))
    print('Use union box: ' + str(args.use_u))
    print('Use S graph: ' + str(args.use_sg))
    print('Use S graph att: ' + str(args.use_sg_att))
    print('Use A graph: ' + str(args.use_ag))
    print('Use A graph att: ' + str(args.use_ag_att))
    print('Use binary: ' + str(args.use_binary))
    net = SIGAN(use_skebox=args.use_skebox, use_bodypart=args.use_bodypart,
                use_pm=args.use_pm, use_u=args.use_u,
                use_sg=args.use_sg, use_sg_att=args.use_sg_att,
                use_ag=args.use_ag, use_ag_att=args.use_ag_att,
                use_binary=args.use_binary,
                is_training=False)
    net.create_architecture(False)
    saver = tf.train.Saver()
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')

    test_net(sess, net, Test_RCNN, prior_mask, Action_dic_inv, output_file,
             args.object_thres, args.human_thres,
             args.prior_flag, args.use_pm)
    sess.close()

    vcocoeval._do_eval(output_file, ovr_thresh=0.5)
