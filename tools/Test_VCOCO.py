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

from ult.vsrl_eval import VCOCOeval
from networks.SPGCN_VCOCO_v2 import SPGCN
from ult.config import cfg
from models.test_VCOCO import test_net
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser(description='Test an iCAN on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
                        help='Specify which weight to load',
                        default=300000, type=int)
    parser.add_argument('--prior_flag', dest='prior_flag',
                        help='whether use prior_flag or not',
                        default=3, type=int)
    parser.add_argument('--model', dest='model',
                        help='Select model',
                        default='iCAN_ResNet50_HICO', type=str)
    parser.add_argument('--object_thres', dest='object_thres',
                        help='Object threshold',
                        default=0.2, type=float)
    parser.add_argument('--human_thres', dest='human_thres',
                        help='Human threshold',
                        default=0.8, type=float)
    parser.add_argument('--version', dest='version',
                        help='latefusion type',
                        default='v1', type=str)
    parser.add_argument('--num_fc', dest='num_fc',
                        help='Number of FC2',
                        default=1024, type=int)
    parser.add_argument('--posemap', dest='posemap',
                        help='use posemap or not',
                        action='store_true', default=False)
    parser.add_argument('--semantic', dest='semantic',
                        help='use semantic or not',
                        action='store_true', default=False)
    parser.add_argument('--posegraph', dest='posegraph',
                        help='use posegraph or not',
                        action='store_true', default=False)
    parser.add_argument('--posegraph_att', dest='posegraph_att',
                        help='use posegraph or not',
                        action='store_true', default=False)
    parser.add_argument('--binary', dest='binary',
                        help='use binary or not',
                        action='store_true', default=False)
    parser.add_argument('--bi_posegraph', dest='bi_posegraph',
                        help='use bi_posegraph or not',
                        action='store_true', default=False)
    parser.add_argument('--tmp', dest='tmp',
                        help='use tmp or not',
                        action='store_true', default=False)
    parser.add_argument('--bodypoint', dest='bodypoint',
                        help='use bodypoint or not',
                        action='store_true', default=False)
    parser.add_argument('--bodypart', dest='bodypart',
                        help='use bodypart or not',
                        action='store_true', default=False)
    parser.add_argument('--latefusion', dest='latefusion',
                        help='use latefusion or not',
                        action='store_true', default=False)
    parser.add_argument('--eval', dest='eval',
                        help='use eval or not',
                        action='store_true', default=False)
    parser.add_argument('--H', dest='H',
                        help='use H or not',
                        action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

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

    posetype = 2 if args.posemap else 1
    if args.binary or args.version == 'v2':
        posetype = 2
    print('Posetype: ' + str(posetype))
    print('Bodypoint: ' + str(args.bodypoint))
    print('Bodypart: ' + str(args.bodypart))
    print('Posemap: ' + str(args.posemap))
    print('Posegraph: ' + str(args.posegraph))
    print('Posegraph_att: ' + str(args.posegraph_att))
    print('Bi-posegraph: ' + str(args.bi_posegraph))
    print('Binary: ' + str(args.binary))
    print('Semantic: ' + str(args.semantic))
    print('Latefusion: ' + str(args.latefusion))
    print('H: ' + str(args.H))
    print('Posenorm: ' + str(cfg.POSENORM))

    net = SPGCN(posetype=posetype, num_fc=args.num_fc,
                posemap=args.posemap, posegraph=args.posegraph, posegraph_att=args.posegraph_att,
                bodypoint=args.bodypoint, bodypart=args.bodypart,
                #H=args.H,
                binary=args.binary, bi_posegraph=args.bi_posegraph,
                semantic=args.semantic, is_training=False)
    net.create_architecture(False)

    saver = tf.train.Saver()
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')

    test_net(sess, net, Test_RCNN, prior_mask, Action_dic_inv, output_file, args.object_thres, args.human_thres,
             args.prior_flag, posetype)
    sess.close()

    vcocoeval._do_eval(output_file, ovr_thresh=0.5)
