from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pickle
import argparse
import _init_paths
import numpy as np
from utils.config import cfg
from models.train_Solver_HICO import train_net
from networks.SIGAN_HICO import SIGAN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser(description='Train an iCAN on VCOCO')
    parser.add_argument('--num_iteration', dest='max_iters', help='Number of iterations to perform', default=300000, type=int)
    parser.add_argument('--model', dest='model', help='Select model', default='SIGCN_HICO', type=str)
    parser.add_argument('--Pos_augment', dest='Pos_augment',
                        help='Number of augmented detection for each one. (By jittering the object detections)',
                        default=30, type=int)
    parser.add_argument('--Neg_select', dest='Neg_select', help='Number of Negative example selected for each image',
                        default=60, type=int)
    parser.add_argument('--Restore_flag', dest='Restore_flag', help='How many ResNet blocks are there?', default=5, type=int)
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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.tmp:
        Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_with_pose_tmp.pkl', "rb"))
        Trainval_N = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_HICO_with_pose_tmp.pkl', "rb"))
    else:
        Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Part_GT_HICO_with_pose.pkl', "rb"))
        Trainval_N = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_HICO_with_pose.pkl', "rb"))

    np.random.seed(cfg.RNG_SEED)
    if cfg.TRAIN_MODULE_CONTINUE == 1:
        weight = cfg.ROOT_DIR + '/Weights/{:s}/HOI_iter_420000.ckpt'.format(args.model)  # from ckpt which you wish to continue
    else:
        weight = cfg.ROOT_DIR + '/Weights/res50_faster_rcnn_iter_1190000.ckpt'  # only restore ResNet weights

    # output directory where the models are saved
    output_dir = cfg.ROOT_DIR + '/Weights/' + args.model + '/'

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
                use_binary=args.use_binary)

    train_net(net, Trainval_GT, Trainval_N, output_dir, weight,
              args.Pos_augment, args.Neg_select, args.Restore_flag,
              max_iters=args.max_iters)
