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
import argparse
import pickle
import os

from networks.SPGCN_HICO import SPGCN
from ult.config import cfg
from ult.Generate_HICO_detection import Generate_HICO_detection
from models.test_HICO import test_net
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser(description='Test an iCAN on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
                        help='Specify which weight to load',
                        default=1800000, type=int)
    parser.add_argument('--model', dest='model',
                        help='Select model',
                        default='iCAN_ResNet50_HICO', type=str)
    parser.add_argument('--object_thres', dest='object_thres',
                        help='Object threshold',
                        default=0.3, type=float)
    parser.add_argument('--human_thres', dest='human_thres',
                        help='Human threshold',
                        default=0.8, type=float)
    parser.add_argument('--num_fc', dest='num_fc',
                        help='Number of FC2',
                        default=1024, type=int)
    parser.add_argument('--latefusion', dest='latefusion',
                        help='latefusion type',
                        default='iCAN', type=str)
    parser.add_argument('--posemap', dest='posemap',
                        help='use posemap or not',
                        action='store_true', default=False)
    parser.add_argument('--semantic', dest='semantic',
                        help='use semantic or not',
                        action='store_true', default=False)
    parser.add_argument('--posegraph', dest='posegraph',
                        help='use posegraph or not',
                        action='store_true', default=False)
    parser.add_argument('--binary', dest='binary',
                        help='use binary or not',
                        action='store_true', default=False)
    parser.add_argument('--tmp', dest='tmp',
                        help='use tmp or not',
                        action='store_true', default=False)
    parser.add_argument('--bodypart', dest='bodypart',
                        help='use bodypart or not',
                        action='store_true', default=False)
    parser.add_argument('--eval', dest='eval',
                        help='use eval or not',
                        action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if not args.tmp:
        Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose.pkl', "rb"))
    else:
        Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Val_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose.pkl', "rb"))
    weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    print('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(
        args.iteration) + ', path = ' + weight)

    output_file = cfg.ROOT_DIR + '/Results/' + str(args.iteration) + '_' + args.model + '.pkl'

    HICO_dir = cfg.ROOT_DIR + '/Results/HICO/' + str(args.iteration) + '_' + args.model + '/'

    # init session
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    posetype = 2 if args.posemap else 1
    if args.binary:
        posetype = 2
    print('Posetype: ' + str(posetype))
    print('Posemap: ' + str(args.posemap))
    print('Posegraph: ' + str(args.posegraph))
    print('Binary: ' + str(args.binary))
    print('Latefusion: ' + str(args.latefusion))
    print('Semantic: ' + str(args.semantic))

    if not args.eval:
        net = SPGCN(posetype=posetype, num_fc=args.num_fc,
                    posemap=args.posemap, posegraph=args.posegraph,
                    binary=args.binary,
                    semantic=args.semantic, is_training=False)
        net.create_architecture(False)

        saver = tf.train.Saver()
        saver.restore(sess, weight)

        print('Pre-trained weights loaded.')

        test_net(sess, net, Test_RCNN, output_file, args.object_thres, args.human_thres, posetype)
        sess.close()
    else:
        Generate_HICO_detection(output_file, HICO_dir)
        os.chdir(cfg.ROOT_DIR + '/HICO-DET_Benchmark/')
        os.system(
            "python Generate_HICO_detection_nis.py " + output_file + ' ' + cfg.ROOT_DIR + "/Results/HICO/" +
            str(args.iteration) + '_' + args.model + "_filter/ " + str(0.9) + " " + str(0.1))


# matlab -nodesktop -nosplash -r "Generate_detection ../Results/"
