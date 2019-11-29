# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen Gao
# --------------------------------------------------------

"""
Generating training instance
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import random
from random import randint
import cv2
from .config import cfg


def bbox_trans(human_box_ori, object_box_ori, ratio, size=64):
    human_box = human_box_ori.copy()
    object_box = object_box_ori.copy()

    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                          max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1

    if height > width:
        ratio = 'height'
    else:
        ratio = 'width'

    # shift the top-left corner to (0,0)

    human_box[0] -= InteractionPattern[0]
    human_box[2] -= InteractionPattern[0]
    human_box[1] -= InteractionPattern[1]
    human_box[3] -= InteractionPattern[1]
    object_box[0] -= InteractionPattern[0]
    object_box[2] -= InteractionPattern[0]
    object_box[1] -= InteractionPattern[1]
    object_box[3] -= InteractionPattern[1]

    if ratio == 'height':  # height is larger than width

        human_box[0] = 0 + size * human_box[0] / height
        human_box[1] = 0 + size * human_box[1] / height
        human_box[2] = (size * width / height - 1) - size * (width - 1 - human_box[2]) / height
        human_box[3] = (size - 1) - size * (height - 1 - human_box[3]) / height

        object_box[0] = 0 + size * object_box[0] / height
        object_box[1] = 0 + size * object_box[1] / height
        object_box[2] = (size * width / height - 1) - size * (width - 1 - object_box[2]) / height
        object_box[3] = (size - 1) - size * (height - 1 - object_box[3]) / height

        # Need to shift horizontally  
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                              max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
        # assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[3] == 63) & (InteractionPattern[2] <= 63)
        if human_box[3] > object_box[3]:
            human_box[3] = size - 1
        else:
            object_box[3] = size - 1

        shift = size / 2 - (InteractionPattern[2] + 1) / 2
        human_box += [shift, 0, shift, 0]
        object_box += [shift, 0, shift, 0]

    else:  # width is larger than height

        human_box[0] = 0 + size * human_box[0] / width
        human_box[1] = 0 + size * human_box[1] / width
        human_box[2] = (size - 1) - size * (width - 1 - human_box[2]) / width
        human_box[3] = (size * height / width - 1) - size * (height - 1 - human_box[3]) / width

        object_box[0] = 0 + size * object_box[0] / width
        object_box[1] = 0 + size * object_box[1] / width
        object_box[2] = (size - 1) - size * (width - 1 - object_box[2]) / width
        object_box[3] = (size * height / width - 1) - size * (height - 1 - object_box[3]) / width

        # Need to shift vertically 
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                              max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

        # assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[2] == 63) & (InteractionPattern[3] <= 63)

        if human_box[2] > object_box[2]:
            human_box[2] = size - 1
        else:
            object_box[2] = size - 1

        shift = size / 2 - (InteractionPattern[3] + 1) / 2

        human_box = human_box + [0, shift, 0, shift]
        object_box = object_box + [0, shift, 0, shift]

    return np.round(human_box), np.round(object_box)


def Get_next_sp(human_box, object_box):
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                          max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    if height > width:
        H, O = bbox_trans(human_box, object_box, 'height')
    else:
        H, O = bbox_trans(human_box, object_box, 'width')

    Pattern = np.zeros((64, 64, 2))
    Pattern[int(H[1]):int(H[3]) + 1, int(H[0]):int(H[2]) + 1, 0] = 1
    Pattern[int(O[1]):int(O[3]) + 1, int(O[0]):int(O[2]) + 1, 1] = 1

    return Pattern


def Get_next_semantic(clsid, clsid2cls, objs_bert):
    clsname = clsid2cls[clsid]
    semantic = objs_bert[clsname]
    semantic = np.asarray(semantic).reshape(-1, 1024)
    return semantic


def Get_next_sp_with_pose(human_box, object_box, human_pose, objscore):
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                          max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    if height > width:
        H, O = bbox_trans(human_box, object_box, 'height')
    else:
        H, O = bbox_trans(human_box, object_box, 'width')
    Pattern = np.zeros((64, 64, 2))
    Pattern[int(H[1]):int(H[3]) + 1, int(H[0]):int(H[2]) + 1, 0] = 1
    Pattern[int(O[1]):int(O[3]) + 1, int(O[0]):int(O[2]) + 1, 1] = 1

    Pose_nodes = human_pose
    Pose_nodes += [object_box[0], object_box[1], float(objscore), object_box[2], object_box[3], float(objscore)]
    len_nodes = len(Pose_nodes)
    Pose_nodes = np.asarray(Pose_nodes).reshape(-1, len_nodes)
    return Pattern, Pose_nodes


def bb_IOU(boxA, boxB):
    ixmin = np.maximum(boxA[0], boxB[0])
    iymin = np.maximum(boxA[1], boxB[1])
    ixmax = np.minimum(boxA[2], boxB[2])
    iymax = np.minimum(boxA[3], boxB[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((boxB[2] - boxB[0] + 1.) * (boxB[3] - boxB[1] + 1.) +
           (boxA[2] - boxA[0] + 1.) *
           (boxA[3] - boxA[1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps


def Augmented_box(bbox, shape, image_id, augment=15):
    thres_ = 0.7

    box = np.array([0, bbox[0], bbox[1], bbox[2], bbox[3]]).reshape(1, 5)
    box = box.astype(np.float64)

    count = 0
    time_count = 0
    while count < augment:

        time_count += 1
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]

        height_cen = (bbox[3] + bbox[1]) / 2
        width_cen = (bbox[2] + bbox[0]) / 2

        ratio = 1 + randint(-10, 10) * 0.01

        height_shift = randint(-np.floor(height), np.floor(height)) * 0.1
        width_shift = randint(-np.floor(width), np.floor(width)) * 0.1

        H_0 = max(0, width_cen + width_shift - ratio * width / 2)
        H_2 = min(shape[1] - 1, width_cen + width_shift + ratio * width / 2)
        H_1 = max(0, height_cen + height_shift - ratio * height / 2)
        H_3 = min(shape[0] - 1, height_cen + height_shift + ratio * height / 2)

        if bb_IOU(bbox, np.array([H_0, H_1, H_2, H_3])) > thres_:
            box_ = np.array([0, H_0, H_1, H_2, H_3]).reshape(1, 5)
            box = np.concatenate((box, box_), axis=0)
            count += 1
        if time_count > 150:
            return box

    return box


def Generate_action(action_list):
    action_ = np.zeros(29)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1, 29)
    return action_


def Generate_action_HICO(action_list):
    action_ = np.zeros(600)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1, 600)
    return action_


# =============
# VCOCO
# =============
def Get_Next_Instance_VCOCO(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length, use_pm):
    GT = trainval_GT[iter % Data_length]
    image_id = GT[0][0]
    im_file = cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' \
              + (str(image_id)).zfill(12) + '.jpg'
    im = cv2.imread(im_file)
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    spatial, SGinput, Human_augmented, Object_augmented, Union_augmented, skeboxes, bodyparts,\
    action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label, pose_none_flag = \
        Augmented_VCOCO(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select, use_pm)

    blobs = dict()
    blobs['image'] = im_orig
    blobs['image_id'] = image_id
    blobs['H_num'] = len(action_H)
    blobs['H_boxes'] = Human_augmented
    blobs['O_boxes'] = Object_augmented
    blobs['U_boxes'] = Union_augmented
    blobs['SGinput'] = SGinput
    blobs['sp'] = spatial
    blobs['skeboxes'] = skeboxes
    blobs['bodyparts'] = bodyparts

    blobs['gt_class_sp'] = action_sp
    blobs['gt_class_HO'] = action_HO
    blobs['gt_class_H'] = action_H
    blobs['Mask_sp'] = mask_sp
    blobs['Mask_HO'] = mask_HO
    blobs['Mask_H'] = mask_H
    blobs['binary_label'] = binary_label

    blobs['pose_none_flag'] = pose_none_flag
    return blobs


def Augmented_VCOCO(GT, Trainval_Neg, shape, Pos_augment, Neg_select, use_pm):
    image_id = GT[0][0]
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    pose_none_flag = 1

    Human_augmented, Object_augmented, Union_augmented, action_HO, action_H = [], [], [], [], []
    SGinput = []
    ori_poses = []
    skeboxes = np.zeros([0, 17, 5])
    bodyparts = np.zeros([0, 6, 5])
    for i in range(GT_count):
        Human = GT[i][2]
        Object = GT[i][3]

        Human_augmented_temp = Augmented_box(Human, shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)
        length_min = min(len(Human_augmented_temp), len(Object_augmented_temp))
        Human_augmented_temp = Human_augmented_temp[:length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]

        Union_augmented_temp = np.zeros([0, 5])
        for j in range(length_min):
            Union_augmented_temp = np.concatenate([Union_augmented_temp, get_union(Human_augmented_temp[j], Object_augmented_temp[j])], axis=0)

        action_H__temp = Generate_action(GT[i][4])
        action_H_temp = action_H__temp
        for j in range(length_min - 1):
            action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

        action_HO__temp = Generate_action(GT[i][1])
        action_HO_temp = action_HO__temp
        for j in range(length_min - 1):
            action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

        Human_augmented.extend(Human_augmented_temp)
        Object_augmented.extend(Object_augmented_temp)
        Union_augmented.extend(Union_augmented_temp)
        action_HO.extend(action_HO_temp)
        action_H.extend(action_H_temp)

        sgraph_in, pose_none_flag = get_sgraph_in(GT[i][5], Human, pose_none_flag)  # list with length of 51
        for j in range(length_min):
            ori_poses.append(GT[i][5])
            Object_nodeA = [Object_augmented_temp[j][1], Object_augmented_temp[j][2],
                            1]  # left up corner of the object box
            Object_nodeB = [Object_augmented_temp[j][3], Object_augmented_temp[j][4],
                            1]  # right down corner of the object box
            norm_sgraph_in = sgraph_in_norm(sgraph_in + Object_nodeA + Object_nodeB, Human, cfg.POSENORM)
            SGinput.append(norm_sgraph_in)
            skeboxes = np.concatenate([skeboxes, generate_skebox(GT[i][5], Human, shape)], axis=0)  # [1, 17, 5]
            bodyparts = np.concatenate([bodyparts, generate_bodypart(GT[i][5], Human, shape)], axis=0)

    action_sp = np.array(action_HO).copy()
    num_pos = len(Human_augmented)

    if image_id in Trainval_Neg:
        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Neg_Pose_Nodes, pose_none_flag = get_sgraph_in(Neg[7], Neg[2], pose_none_flag)
                Human_augmented = np.concatenate(
                    (Human_augmented, np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1, 5)), axis=0)
                Object_augmented = np.concatenate(
                    (Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1, 5)), axis=0)
                Union_augmented = np.concatenate([Union_augmented, get_union(Neg[2], Neg[3])], axis=0)
                ori_poses.append(Neg[7])
                Object_nodeA_B = [Neg[3][0], Neg[3][1], Neg[6], Neg[3][2], Neg[3][3], Neg[6]]
                norm_sgraph_in = sgraph_in_norm(Neg_Pose_Nodes + Object_nodeA_B, Neg[2], cfg.POSENORM)
                SGinput.append(norm_sgraph_in)
                skeboxes = np.concatenate([skeboxes, generate_skebox(Neg[7], Neg[2], shape)], axis=0)
                bodyparts = np.concatenate([bodyparts, generate_bodypart(Neg[7], Neg[2], shape)], axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Neg_Pose_Nodes, pose_none_flag = get_sgraph_in(Neg[7], Neg[2], pose_none_flag)
                Human_augmented = np.concatenate(
                    (Human_augmented, np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1, 5)), axis=0)
                Object_augmented = np.concatenate(
                    (Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1, 5)), axis=0)
                Union_augmented = np.concatenate([Union_augmented, get_union(Neg[2], Neg[3])], axis=0)
                ori_poses.append(Neg[7])
                Object_nodeA_B = [Neg[3][0], Neg[3][1], Neg[6], Neg[3][2], Neg[3][3], Neg[6]]
                norm_sgraph_in = sgraph_in_norm(Neg_Pose_Nodes + Object_nodeA_B, Neg[2], cfg.POSENORM)
                SGinput.append(norm_sgraph_in)
                skeboxes = np.concatenate([skeboxes, generate_skebox(Neg[7], Neg[2], shape)], axis=0)
                bodyparts = np.concatenate([bodyparts, generate_bodypart(Neg[7], Neg[2], shape)], axis=0)
    num_pos_neg = len(Human_augmented)

    # spatial
    if use_pm:
        spatial = np.empty((0, 64, 64, 3), dtype=np.float32)
        for i in range(num_pos_neg):
            spatial_ = Get_next_sp_with_posemap(Human_augmented[i][1:], Object_augmented[i][1:],
                                                ori_poses[i]).reshape(1, 64, 64, 3)
            spatial = np.concatenate((spatial, spatial_), axis=0)
        spatial = spatial.reshape(num_pos_neg, 64, 64, 3)
    else:
        spatial = np.empty((0, 64, 64, 2), dtype=np.float32)
        for i in range(num_pos_neg):
            spatial_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
            spatial = np.concatenate((spatial, spatial_), axis=0)
        spatial = spatial.reshape(num_pos_neg, 64, 64, 2)

    mask_sp_ = np.asarray(
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]).reshape(1, 29)
    mask_HO_ = np.asarray(
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]).reshape(1, 29)
    mask_H_ = np.asarray(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(1, 29)
    mask_sp = mask_sp_
    mask_HO = mask_HO_
    mask_H = mask_H_

    for i in range(num_pos - 1):
        mask_HO = np.concatenate((mask_HO, mask_HO_), axis=0)
        mask_H = np.concatenate((mask_H, mask_H_), axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp = np.concatenate((mask_sp, mask_sp_), axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.zeros(29).reshape(1, 29)), axis=0)  # neg has all 0 in 29 label

    Human_augmented = np.array(Human_augmented, dtype='float32')
    Object_augmented = np.array(Object_augmented, dtype='float32')
    Union_augmented = np.asarray(Union_augmented, dtype='float32')
    action_HO = np.array(action_HO, dtype='int32')
    action_sp = np.array(action_sp, dtype='int32')
    action_H = np.array(action_H, dtype='int32')

    Human_augmented = Human_augmented.reshape(num_pos_neg, 5)
    Object_augmented = Object_augmented.reshape(num_pos_neg, 5)
    Union_augmented = Union_augmented.reshape(num_pos_neg, 5)

    action_sp = action_sp.reshape(num_pos_neg, 29)
    action_HO = action_HO.reshape(num_pos, 29)
    action_H = action_H.reshape(num_pos, 29)
    mask_sp = mask_sp.reshape(num_pos_neg, 29)
    mask_HO = mask_HO.reshape(num_pos, 29)
    mask_H = mask_H.reshape(num_pos, 29)
    SGinput = np.asarray(SGinput).reshape(num_pos_neg, 51 + 6)
    skeboxes = skeboxes.reshape([-1, 17, 5])
    bodyparts = bodyparts.reshape([-1, 6, 5])
    binary_label = np.zeros((num_pos_neg, 1), dtype='int32')
    for i in range(num_pos):
        binary_label[i] = 1  # pos is at 0
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i] = 0  # neg is at 1

    return spatial, SGinput, Human_augmented, Object_augmented, Union_augmented, skeboxes, bodyparts, \
           action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label, pose_none_flag


# =============
# HICO
# =============
def Get_Next_Instance_HO_Neg_HICO_v2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length,
                                  clsid2cls, objs_bert, posetype):
    GT = trainval_GT[iter % Data_length]
    image_id = GT[0][0]
    im_file = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(
        8) + '.jpg'
    im = cv2.imread(im_file)
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    blobs = dict()

    Pattern, Gnodes, semantic, Human_augmented, Object_augmented, action_HO, num_pos, \
    binary_label, partboxes, pose_none_flag = \
            Augmented_HO_Neg_HICO_v2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select,
                                     clsid2cls, objs_bert, posetype)

    blobs['image'] = im_orig
    blobs['image_id'] = image_id
    blobs['semantic'] = semantic
    blobs['H_boxes'] = Human_augmented
    blobs['O_boxes'] = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp'] = Pattern
    blobs['Gnodes'] = Gnodes
    blobs['H_num'] = num_pos #len(action_HO)
    blobs['binary_label'] = binary_label
    blobs['partboxes'] = partboxes
    blobs['pose_none_flag'] = pose_none_flag
    blobs['num_pose'] = num_pos
    return blobs


def Augmented_HO_Neg_HICO_v2(GT, Trainval_Neg, shape, Pos_augment, Neg_select, clsid2cls, objs_bert, posetype):
    pose_none_flag = 1
    GT_count = len(GT)  # num_of_interval_divide
    aug_all = int(Pos_augment / GT_count)
    image_id = GT[0][0]

    Human_augmented, Object_augmented, action_HO = [], [], []
    # Gnodes for keypoints and object points
    Gnodes = []
    objclsid = []
    origin_poses = []
    partboxes = np.zeros([0, 17, 5])
    for i in range(GT_count):
        Human = GT[i][2]
        Object = GT[i][3]

        Human_augmented_temp = Augmented_box(Human, shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)
        num_min = min(len(Human_augmented_temp), len(Object_augmented_temp))
        Human_augmented_temp = Human_augmented_temp[: num_min]
        Object_augmented_temp = Object_augmented_temp[:num_min]

        action_HO__temp = Generate_action_HICO(GT[i][1])
        action_HO_temp = action_HO__temp
        for j in range(num_min - 1):
            action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

        Human_augmented.extend(Human_augmented_temp)
        Object_augmented.extend(Object_augmented_temp)
        action_HO.extend(action_HO_temp)

        Pose_nodes, pose_none_flag = get_sgraph_in(GT[i][5], Human, pose_none_flag)  # list with length of 51
        for j in range(num_min):
            origin_poses.append(GT[i][5])
            objclsid.append(GT[i][-1])
            Object_nodeA = [Object_augmented_temp[j][1], Object_augmented_temp[j][2], 1]  # left up corner of the object box
            Object_nodeB = [Object_augmented_temp[j][3], Object_augmented_temp[j][4], 1]  # right down corner of the object box
            norm_nodes = sgraph_in_norm(Pose_nodes + Object_nodeA + Object_nodeB, Human, cfg.POSENORM)
            Gnodes_ = np.asarray(norm_nodes)
            Gnodes.append(Gnodes_)
            # generate 17 part boxes  [1, 17, 5]
            partboxes = np.concatenate([partboxes, generate_skebox(GT[i][5], Human)], axis=0)

    num_pos = len(Human_augmented)

    if image_id in Trainval_Neg:
        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Neg_Pose_Nodes, pose_none_flag = get_sgraph_in(Neg[7], Neg[2], pose_none_flag)
                Human_augmented = np.concatenate(
                    (Human_augmented, np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1, 5)), axis=0)
                Object_augmented = np.concatenate(
                    (Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1, 5)), axis=0)
                action_HO = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
                origin_poses.append(Neg[7])
                objclsid.append(Neg[5])
                Object_nodeA_B = [Neg[3][0], Neg[3][1], Neg[6], Neg[3][2], Neg[3][3], Neg[6]]
                norm_nodes = sgraph_in_norm(Neg_Pose_Nodes + Object_nodeA_B, Neg[2], cfg.POSENORM)
                Gnodes.append(norm_nodes)
                partboxes = np.concatenate([partboxes, generate_skebox(Neg[7], Neg[2])], axis=0)

        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Neg_Pose_Nodes, pose_none_flag = get_sgraph_in(Neg[7], Neg[2], pose_none_flag)
                Human_augmented = np.concatenate(
                    (Human_augmented, np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1, 5)), axis=0)
                Object_augmented = np.concatenate(
                    (Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1, 5)), axis=0)
                action_HO = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
                origin_poses.append(Neg[7])
                objclsid.append(Neg[5])
                Object_nodeA_B = [Neg[3][0], Neg[3][1], Neg[6], Neg[3][2], Neg[3][3], Neg[6]]
                norm_nodes = sgraph_in_norm(Neg_Pose_Nodes + Object_nodeA_B, Neg[2], cfg.POSENORM)
                Gnodes.append(norm_nodes)
                partboxes = np.concatenate([partboxes, generate_skebox(Neg[7], Neg[2])], axis=0)

    num_pos_neg = len(Human_augmented)

    # Semantic Representation
    semantic = []
    for i in objclsid:
        clsname = clsid2cls[i]
        semantic.append(objs_bert[clsname])
    semantic = np.asarray(semantic).reshape(-1, 1024)

    # Pattern
    if posetype == 1:
        Pattern = np.empty((0, 64, 64, 2), dtype=np.float32)
        for i in range(num_pos_neg):
            Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
            Pattern = np.concatenate((Pattern, Pattern_), axis=0)
        Pattern = Pattern.reshape(num_pos_neg, 64, 64, 2)
    elif posetype == 2:
        Pattern = np.empty((0, 64, 64, 3), dtype=np.float32)
        for i in range(num_pos_neg):
            Pattern_ = Get_next_sp_with_posemap(Human_augmented[i][1:], Object_augmented[i][1:],
                                                origin_poses[i]).reshape(1, 64, 64, 3)
            Pattern = np.concatenate((Pattern, Pattern_), axis=0)
        Pattern = Pattern.reshape(num_pos_neg, 64, 64, 3)
    else:
        raise NotImplementedError

    Human_augmented = np.asarray(Human_augmented).reshape(num_pos_neg, 5)
    Object_augmented = np.asarray(Object_augmented).reshape(num_pos_neg, 5)
    action_HO = np.asarray(action_HO).reshape(num_pos_neg, 600)
    Gnodes = np.asarray(Gnodes).reshape(num_pos_neg, 51 + 6)
    partboxes = partboxes.reshape([-1, 17, 5])

    no_inter_ids = no_inter_turn_zero(action_HO)
    binary_label = np.zeros((num_pos_neg, 2), dtype='int32')
    sparse_binary_label = np.zeros((num_pos_neg, 1), dtype='int32')
    for i in range(num_pos):
        if i not in no_inter_ids:
            binary_label[i][0] = 1
            sparse_binary_label[i] = 1
        else:
            binary_label[i][1] = 1
            sparse_binary_label[i] = 0
    for i in range(num_pos, num_pos_neg):
        binary_label[i][1] = 1  # neg is at 1
        sparse_binary_label[i] = 0
    pose_none_flag = 1
    return Pattern, Gnodes, semantic, Human_augmented, Object_augmented, action_HO, num_pos, \
           sparse_binary_label, partboxes, pose_none_flag


def get_sgraph_in(pose_nodes, Human, pose_none_flag):
    if pose_nodes is None:  # the human pose is None
        centric_x = (Human[0] + Human[2]) / 2.
        centric_y = (Human[1] + Human[3]) / 2.
        sgraph_in = []
        for i in range(17):
            sgraph_in += [centric_x, centric_y, 1]
        pose_none_flag = 0
    else:
        sgraph_in = pose_nodes

    return sgraph_in, pose_none_flag


def sgraph_in_norm(Pose_nodes, Human, pose_norm):
    if pose_norm == 1:
        center_x, center_y = Pose_nodes[0], Pose_nodes[1]
        for i in range(len(Pose_nodes)):
            if i % 3 == 0:
                Pose_nodes[i] = Pose_nodes[i] / float(center_x)
            elif i % 3 == 1:
                Pose_nodes[i] = Pose_nodes[i] / float(center_y)
    elif pose_norm == 2:  # human-centric and human box
        w = Human[2] - Human[0]
        h = Human[3] - Human[1]
        for i in range(len(Pose_nodes)):
            center_x, center_y = Pose_nodes[0], Pose_nodes[1]
            if i % 3 == 0:
                Pose_nodes[i] = (Pose_nodes[i] - center_x) / float(w)
            elif i % 3 == 1:
                Pose_nodes[i] = (Pose_nodes[i] - center_y) / float(h)
    elif pose_norm == 3:
        w = Human[2] - Human[0]
        h = Human[3] - Human[1]
        center_x, center_y = Pose_nodes[0], Pose_nodes[1]
        for i in range(len(Pose_nodes)):
            if i % 3 == 0:
                Pose_nodes[i] = (Pose_nodes[i] - center_x) / w
            elif i % 3 == 1:
                Pose_nodes[i] = (Pose_nodes[i] - center_y) / h
    else:
        raise NotImplementedError
    return Pose_nodes


def no_inter_turn_zero(action_HO):
    # all the 80 no-interaction HOI index in HICO dataset
    hoi_no_inter_all = [10, 24, 31, 46, 54, 65, 76, 86, 92, 96, 107, 111, 129, 146, 160, 170, 174, 186,
                        194, 198, 208, 214, 224, 232, 235, 239, 243, 247, 252, 257, 264, 273, 283, 290,
                        295, 305, 313, 325, 330, 336, 342, 348, 352, 356, 363, 368, 376, 383, 389, 393,
                        397, 407, 414, 418, 429, 434, 438, 445, 449, 453, 463, 474, 483, 488, 502, 506,
                        516, 528, 533, 538, 546, 550, 558, 562, 567, 576, 584, 588, 595, 600]
    hoi_no_inter_all = [i - 1 for i in hoi_no_inter_all]
    action_HO_80 = action_HO[:, hoi_no_inter_all]
    assert action_HO_80.shape[1] == 80

    tmp = action_HO_80.sum(axis=1).reshape(-1)
    assert tmp.shape[0] == action_HO.shape[0]

    no_inter_ids = list(np.where(tmp != 0)[0])  # the ids where containing hoi_no_inter_all class

    return no_inter_ids


def draw_relation(human_pattern, joints, size=64):
    joint_relation = [[1, 3], [2, 4], [0, 1], [0, 2], [0, 17], [5, 17], [6, 17], [5, 7], [6, 8], [7, 9], [8, 10],
                      [11, 17], [12, 17], [11, 13], [12, 14], [13, 15], [14, 16]]
    joint_relation2 = [[1, 3], [2, 4], [0, 1], [0, 2], [0, 17], [5, 17], [6, 17], [5, 7], [6, 8], [7, 9], [8, 10],
                      [11, 17], [12, 17], [11, 13], [12, 14], [13, 15], [14, 16], ]

    color = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    skeleton = np.zeros((size, size, 1), dtype="float32")

    for i in range(len(joint_relation)):
        cv2.line(skeleton, tuple(joints[joint_relation[i][0]]), tuple(joints[joint_relation[i][1]]), (color[i]))

    return skeleton


def get_skeleton(human_box, human_pose, human_pattern, num_joints = 17, size = 64):
    width = human_box[2] - human_box[0] + 1
    height = human_box[3] - human_box[1] + 1
    pattern_width = human_pattern[2] - human_pattern[0] + 1
    pattern_height = human_pattern[3] - human_pattern[1] + 1
    joints = np.zeros((num_joints + 1, 2), dtype='int32')

    for i in range(num_joints):
        joint_x, joint_y, joint_score = human_pose[3 * i: 3 * (i + 1)]
        x_ratio = (joint_x - human_box[0]) / float(width)
        y_ratio = (joint_y - human_box[1]) / float(height)
        joints[i][0] = min(size - 1, int(round(x_ratio * pattern_width + human_pattern[0])))
        joints[i][1] = min(size - 1, int(round(y_ratio * pattern_height + human_pattern[1])))
    joints[num_joints] = (joints[5] + joints[6]) / 2

    return draw_relation(human_pattern, joints)


def Get_next_sp_with_posemap(human_box, object_box, human_pose, num_joints=17):
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                          max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    if height > width:
        H, O = bbox_trans(human_box, object_box, 'height')
    else:
        H, O = bbox_trans(human_box, object_box, 'width')

    Pattern = np.zeros((64, 64, 2), dtype='float32')
    Pattern[int(H[1]):int(H[3]) + 1, int(H[0]):int(H[2]) + 1, 0] = 1
    Pattern[int(O[1]):int(O[3]) + 1, int(O[0]):int(O[2]) + 1, 1] = 1

    if human_pose != None and len(human_pose) == 51:
        skeleton = get_skeleton(human_box, human_pose, H, num_joints)
    else:
        skeleton = np.zeros((64, 64, 1), dtype='float32')
        skeleton[int(H[1]):int(H[3]) + 1, int(H[0]):int(H[2]) + 1, 0] = 0.05

    Pattern = np.concatenate((Pattern, skeleton), axis=2)

    return Pattern


def generate_skebox(pose, Human, shape):
    if pose is None:  # the human pose is None
        centric_x = (Human[0] + Human[2]) / 2.
        centric_y = (Human[1] + Human[3]) / 2.
        humanpose = []
        for i in range(17):
            humanpose += [centric_x, centric_y, 1]
    else:
        humanpose = pose

    boxes = np.zeros([0, 5])
    edgelen = (Human[2] - Human[0]) * 0.1
    halflen = edgelen / 2
    for i in range(0, len(humanpose), 3):
        x = humanpose[i]
        y = humanpose[i+1]
        box = np.asarray([0, max(x-halflen, 0), max(y-halflen, 0), min(x+halflen, shape[1]), min(y+halflen, shape[0])]).reshape([1, 5])
        boxes = np.concatenate([boxes, box], axis=0)
    boxes = boxes.reshape([1, 17, 5])
    return boxes


def generate_bodypart(pose, Human, shape):
    if pose is None:  # the human pose is None
        centric_x = (Human[0] + Human[2]) / 2.
        centric_y = (Human[1] + Human[3]) / 2.
        humanpose = []
        for i in range(17):
            humanpose += [centric_x, centric_y, 1]
    else:
        humanpose = pose
    # head
    hx1 = max(min([humanpose[0], humanpose[3], humanpose[6], humanpose[9], humanpose[12]])-10, 0)
    hy1 = max(min([humanpose[1], humanpose[4], humanpose[7], humanpose[10], humanpose[13]])-10, 0)
    hx2 = min(max([humanpose[0], humanpose[3], humanpose[6], humanpose[9], humanpose[12]])+10, shape[1])
    hy2 = min(max([humanpose[1], humanpose[4], humanpose[7], humanpose[10], humanpose[13]])+10, shape[0])
    head = np.asarray([0, hx1, hy1, hx2, hy2]).reshape(-1, 5)
    # body
    bx1 = max(min([humanpose[15], humanpose[18], humanpose[33], humanpose[36]])-10, 0)
    by1 = max(min([humanpose[16], humanpose[19], humanpose[34], humanpose[37]])-10, 0)
    bx2 = min(max([humanpose[15], humanpose[18], humanpose[33], humanpose[36]])+10, shape[1])
    by2 = min(max([humanpose[16], humanpose[19], humanpose[34], humanpose[37]])+10, shape[0])
    body = np.asarray([0, bx1, by1, bx2, by2]).reshape(-1, 5)
    # left hand
    lhx1 = max(min([humanpose[15], humanpose[21], humanpose[27]])-10, 0)
    lhy1 = max(min([humanpose[16], humanpose[22], humanpose[28]])-10, 0)
    lhx2 = min(max([humanpose[15], humanpose[21], humanpose[27]])+10, shape[1])
    lhy2 = min(max([humanpose[16], humanpose[22], humanpose[28]])+10, shape[0])
    lefthand = np.asarray([0, lhx1, lhy1, lhx2, lhy2]).reshape(-1, 5)
    # right hand
    rhx1 = max(min([humanpose[18], humanpose[24], humanpose[30]])-10, 0)
    rhy1 = max(min([humanpose[19], humanpose[25], humanpose[31]])-10, 0)
    rhx2 = min(max([humanpose[18], humanpose[24], humanpose[30]])+10, shape[1])
    rhy2 = min(max([humanpose[19], humanpose[25], humanpose[31]])+10, shape[0])
    righthand = np.asarray([0, rhx1, rhy1, rhx2, rhy2]).reshape(-1, 5)
    # left foot
    lfx1 = max(min([humanpose[33], humanpose[39], humanpose[45]])-10, 0)
    lfy1 = max(min([humanpose[34], humanpose[40], humanpose[46]])-10, 0)
    lfx2 = min(max([humanpose[33], humanpose[39], humanpose[45]])+10, shape[1])
    lfy2 = min(max([humanpose[34], humanpose[40], humanpose[46]])+10, shape[0])
    leftfoot = np.asarray([0, lfx1, lfy1, lfx2, lfy2]).reshape(-1, 5)
    # right foot
    rfx1 = max(min([humanpose[36], humanpose[42], humanpose[48]])-10, 0)
    rfy1 = max(min([humanpose[37], humanpose[43], humanpose[49]])-10, 0)
    rfx2 = min(max([humanpose[36], humanpose[42], humanpose[48]])+10, shape[1])
    rfy2 = min(max([humanpose[37], humanpose[43], humanpose[49]])+10, shape[0])
    rightfoot = np.asarray([0, rfx1, rfy1, rfx2, rfy2]).reshape(-1, 5)

    boxes = np.concatenate([head, body, lefthand, righthand, leftfoot, rightfoot], axis=0).reshape([1, 6, 5])
    return boxes


def get_union(boxA, boxB):
    if boxA.shape[0] == 5:
        x1 = min(boxA[1], boxB[1])
        y1 = min(boxA[2], boxB[2])
        x2 = max(boxA[3], boxB[3])
        y2 = max(boxA[4], boxB[4])
    elif boxA.shape[0] == 4:
        x1 = min(boxA[0], boxB[0])
        y1 = min(boxA[1], boxB[1])
        x2 = max(boxA[2], boxB[2])
        y2 = max(boxA[3], boxB[3])
    else:
        raise NotImplementedError
    return np.asarray([0, x1, y1, x2, y2]).reshape([-1, 5])
