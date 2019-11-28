from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import _init_paths
import numpy as np
import argparse
import pickle
from ult.config import cfg
from models.train_Solver_VCOCO import train_net
from networks.SPGCN_VCOCO_v2 import SPGCN
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser(description='Train an iCAN on VCOCO')
    parser.add_argument('--num_iteration', dest='max_iters',
                        help='Number of iterations to perform',
                        default=3000000, type=int)
    parser.add_argument('--model', dest='model',
                        help='Select model',
                        default='SPGCN_HICO', type=str)
    parser.add_argument('--Pos_augment', dest='Pos_augment',
                        help='Number of augmented detection for each one. (By jittering the object detections)',
                        default=15, type=int)
    parser.add_argument('--Neg_select', dest='Neg_select',
                        help='Number of Negative example selected for each image',
                        default=60, type=int)
    parser.add_argument('--Restore_flag', dest='Restore_flag',
                        help='How many ResNet blocks are there?',
                        default=5, type=int)
    parser.add_argument('--num_fc', dest='num_fc',
                        help='Number of FC2',
                        default=1024, type=int)
    parser.add_argument('--version', dest='version',
                        help='latefusion type',
                        default='v1', type=str)
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
    parser.add_argument('--bi_posegraph', dest='bi_posegraph',
                        help='use bi_posegraph or not',
                        action='store_true', default=False)
    parser.add_argument('--binary', dest='binary',
                        help='use binary or not',
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
    parser.add_argument('--H', dest='H',
                        help='use H or not',
                        action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    if not args.tmp:
        Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_VCOCO_with_pose.pkl', "rb"))
        Trainval_N = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_VCOCO_with_pose.pkl', "rb"))
    else:
        Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_VCOCO_with_pose_tmp.pkl', "rb"))
        Trainval_N = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_VCOCO_with_pose_tmp.pkl', "rb"))

    np.random.seed(cfg.RNG_SEED)
    weight = cfg.ROOT_DIR + '/Weights/res50_faster_rcnn_iter_1190000.ckpt'

    # output directory where the logs are saved
    tb_dir = cfg.ROOT_DIR + '/logs/' + args.model + '/'

    # output directory where the models are saved
    output_dir = cfg.ROOT_DIR + '/Weights/' + args.model + '/'

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
                bi_posegraph=args.bi_posegraph,
                H=args.H,
                binary=args.binary, bodypoint=args.bodypoint, bodypart=args.bodypart,
                semantic=args.semantic, latefusion=args.latefusion)

    train_net(net, Trainval_GT, Trainval_N, output_dir, tb_dir, args.Pos_augment,
              args.Neg_select, args.Restore_flag, posetype, weight, max_iters=args.max_iters)
