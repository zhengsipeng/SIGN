from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tools._init_paths
import numpy as np
import argparse
import pickle
from ult.config import cfg
from models.train_Solver_HICO_Interact import train_net
from networks.Interact_GCN import SPGCN
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # use GPU 0,1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser(description='Train Interactive network with SPGCN')
    parser.add_argument('--num_iteration', dest='max_iters',
                        help='Number of iterations to perform',
                        default=2000000, type=int)
    parser.add_argument('--model', dest='model',
                        help='Select model',
                        default='INTERACT', type=str)
    parser.add_argument('--Pos_augment', dest='Pos_augment',
                        help='Number of augmented detection for each one. (By jittering the object detections)',
                        default=15, type=int)
    parser.add_argument('--Neg_select', dest='Neg_select',
                        help='Number of Negative example selected for each image',
                        default=60, type=int)
    parser.add_argument('--Restore_flag', dest='Restore_flag',
                        help='How many ResNet blocks are there?',
                        default=5, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    Trainval_GT = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_with_pose.pkl', "rb"))
    Trainval_N = pickle.load(open(cfg.DATA_DIR + '/' + 'Trainval_Neg_HICO_with_pose.pkl', "rb"))

    np.random.seed(cfg.RNG_SEED)

    if cfg.TRAIN_MODULE_CONTINUE == 1:
        weight = cfg.ROOT_DIR + '/Weights/{:s}/HOI_iter_600000.ckpt'.format(args.model)  # from ckpt which you wish to continue
    else:
        weight = cfg.ROOT_DIR + '/Weights/res50_faster_rcnn_iter_1190000.ckpt'  # only restore ResNet weights

    # output directory where the logs are saved
    tb_dir = cfg.ROOT_DIR + '/logs/' + args.model + '/'

    # output directory where the models are saved
    output_dir = cfg.ROOT_DIR + '/Weights/' + args.model + '/'

    net = SPGCN(posetype=cfg.POSETYPE)
    train_net(net, Trainval_GT, Trainval_N, output_dir, tb_dir, args.Pos_augment,
              args.Neg_select,  args.Restore_flag, cfg.POSETYPE, weight, max_iters=args.max_iters)
