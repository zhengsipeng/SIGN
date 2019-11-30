from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tools._init_paths
import os
import tensorflow as tf
import argparse
import pickle
from networks.Interact_GCN import SPGCN
from ult.config import cfg
from models.test_HICO_Interact import test_net
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser(description='Test an iCAN on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=1800000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='INTERACT', type=str)
    parser.add_argument('--object_thres', dest='object_thres',
            help='Object threshold',
            default=0.3, type=float)
    parser.add_argument('--human_thres', dest='human_thres',
            help='Human threshold',
            default=0.8, type=float)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose.pkl', "rb"))

    weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    print('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(
        args.iteration) + ', path = ' + weight)

    output_file = cfg.ROOT_DIR + '/Results/' + str(args.iteration) + '_' + args.model + '.pkl'
    HICO_dir = cfg.ROOT_DIR + '/Results/HICO/' + str(args.iteration) + '_' + args.model + '/'

    # init session
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    net = SPGCN(is_training=False, posetype=cfg.POSETYPE)
    net.create_architecture(False)

    saver = tf.train.Saver()
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')

    test_net(sess, net, Test_RCNN, output_file, args.object_thres, args.human_thres, cfg.POSETYPE, )
    sess.close()

    # Generate HICO binary detection results
