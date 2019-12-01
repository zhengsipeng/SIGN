import random
from config import cfg
from datatools import *


# =============
# VCOCO
# =============
def Get_Next_Instance_VCOCO(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):
    GT = trainval_GT[iter % Data_length]
    image_id = GT[0][0]
    im_file = cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im = cv2.imread(im_file)
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    sp_masks, SGinput, Human_augmented, Object_augmented, Union_augmented, skeboxes, bodyparts,\
    action_HO, action_H, mask_HO, mask_H, gt_binary_label, pose_none_flag = \
        Augmented_VCOCO(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)

    blobs = dict()
    blobs['image'] = im_orig
    blobs['image_id'] = image_id
    blobs['H_num'] = len(action_H)
    blobs['H_boxes'] = Human_augmented
    blobs['O_boxes'] = Object_augmented
    blobs['U_boxes'] = Union_augmented
    blobs['SGinput'] = SGinput
    blobs['sp'] = sp_masks
    blobs['skeboxes'] = skeboxes
    blobs['bodyparts'] = bodyparts
    blobs['gt_class_HO'] = action_HO
    blobs['gt_class_H'] = action_H
    blobs['Mask_HO'] = mask_HO
    blobs['Mask_H'] = mask_H
    blobs['gt_binary_label'] = gt_binary_label
    blobs['pose_none_flag'] = pose_none_flag
    return blobs


def Augmented_VCOCO(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
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
            Object_nodeA = [Object_augmented_temp[j][1], Object_augmented_temp[j][2], 1]  # left up corner
            Object_nodeB = [Object_augmented_temp[j][3], Object_augmented_temp[j][4], 1]  # right down corner
            norm_sgraph_in = sgraph_in_norm(sgraph_in + Object_nodeA + Object_nodeB, Human, cfg.POSENORM)
            SGinput.append(norm_sgraph_in)
            skeboxes = np.concatenate([skeboxes, generate_skebox(GT[i][5], Human, shape)], axis=0)  # [1, 17, 5]
            bodyparts = np.concatenate([bodyparts, generate_bodypart(GT[i][5], Human, shape)], axis=0)

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
    sp_masks = np.empty((0, 64, 64, 3), dtype=np.float32)
    for i in range(num_pos_neg):
        sp_masks_ = Get_next_sp_with_posemap(Human_augmented[i][1:], Object_augmented[i][1:],
                                             ori_poses[i]).reshape(1, 64, 64, 3)
        sp_masks = np.concatenate((sp_masks, sp_masks_), axis=0)
    sp_masks = sp_masks.reshape(num_pos_neg, 64, 64, 3)

    mask_HO_ = np.asarray(
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]).reshape(1, 29)
    mask_H_ = np.asarray(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(1, 29)
    mask_HO_ = mask_H_  # whether need all 29 categories

    mask_H = mask_H_
    mask_HO = mask_HO_

    for i in range(num_pos - 1):
        mask_H = np.concatenate((mask_H, mask_H_), axis=0)
    for i in range(num_pos_neg - 1):
        mask_HO = np.concatenate((mask_HO, mask_HO_), axis=0)
    for i in range(num_pos_neg - num_pos):
        action_HO = np.concatenate((action_HO, np.zeros(29).reshape(1, 29)), axis=0)  # neg has all 0 in 29 label

    Human_augmented = np.array(Human_augmented, dtype='float32')
    Object_augmented = np.array(Object_augmented, dtype='float32')
    Union_augmented = np.asarray(Union_augmented, dtype='float32')
    action_HO = np.array(action_HO, dtype='int32')
    action_H = np.array(action_H, dtype='int32')

    Human_augmented = Human_augmented.reshape(num_pos_neg, 5)
    Object_augmented = Object_augmented.reshape(num_pos_neg, 5)
    Union_augmented = Union_augmented.reshape(num_pos_neg, 5)
    action_HO = action_HO.reshape(num_pos_neg, 29)
    action_H = action_H.reshape(num_pos, 29)

    mask_HO = mask_HO.reshape(num_pos_neg, 29)
    mask_H = mask_H.reshape(num_pos, 29)

    SGinput = np.asarray(SGinput).reshape(num_pos_neg, 51 + 6)
    skeboxes = skeboxes.reshape([-1, 17, 5])
    bodyparts = bodyparts.reshape([-1, 6, 5])
    binary_label = np.zeros((num_pos_neg, 1), dtype='int32')
    for i in range(num_pos):
        binary_label[i] = 1  # pos is at 0
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i] = 0  # neg is at 1

    return sp_masks, SGinput, Human_augmented, Object_augmented, Union_augmented, skeboxes, bodyparts, \
           action_HO, action_H, mask_HO, mask_H, binary_label, pose_none_flag


# =============
# HICO
# =============
def Get_Next_Instance_HICO(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length, clsid2cls, objs_bert):
    GT = trainval_GT[iter % Data_length]
    image_id = GT[0][0]
    im_file = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + \
              (str(image_id)).zfill(8) + '.jpg'
    im = cv2.imread(im_file)
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig = im_orig.reshape(1, im_shape[0], im_shape[1], 3)  # 1, H, W, 3

    blobs = dict()
    sp_masks, SGinput, semantic, Human_augmented, Object_augmented, skeboxes, bodyparts, \
    action_HO, num_pos, gt_binary_label, pose_none_flag = \
            Augmented_HICO(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select, clsid2cls, objs_bert)

    blobs['image'] = im_orig
    blobs['image_id'] = image_id
    blobs['semantic'] = semantic
    blobs['H_boxes'] = Human_augmented
    blobs['O_boxes'] = Object_augmented
    blobs['sp'] = sp_masks
    blobs['skeboxes'] = skeboxes
    blobs['Gnodes'] = SGinput
    blobs['H_num'] = num_pos
    blobs['gt_class_HO'] = action_HO
    blobs['gt_binary_label'] = gt_binary_label
    blobs['pose_none_flag'] = pose_none_flag
    return blobs


def Augmented_HICO(GT, Trainval_Neg, shape, Pos_augment, Neg_select, clsid2cls, objs_bert):
    pose_none_flag = 1
    GT_count = len(GT)  # num_of_interval_divide
    aug_all = int(Pos_augment / GT_count)
    image_id = GT[0][0]

    Human_augmented, Object_augmented, Union_augmented, action_HO = [], [], [], []
    SGinput = []
    origin_poses = []
    skeboxes = np.zeros([0, 17, 5])
    bodyparts = np.zeros([0, 6, 5])
    objclsid = []
    for i in range(GT_count):
        Human = GT[i][2]
        Object = GT[i][3]

        Human_augmented_temp = Augmented_box(Human, shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)
        length_min = min(len(Human_augmented_temp), len(Object_augmented_temp))
        Human_augmented_temp = Human_augmented_temp[: length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]

        Union_augmented_temp = np.zeros([0, 5])
        for j in range(length_min):
            Union_augmented_temp = np.concatenate(
                [Union_augmented_temp, get_union(Human_augmented_temp[j], Object_augmented_temp[j])], axis=0)

        action_HO__temp = Generate_action_HICO(GT[i][1])
        action_HO_temp = action_HO__temp
        for j in range(length_min - 1):
            action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

        Human_augmented.extend(Human_augmented_temp)
        Object_augmented.extend(Object_augmented_temp)
        Union_augmented.extend(Union_augmented_temp)
        action_HO.extend(action_HO_temp)

        sgraph_in, pose_none_flag = get_sgraph_in(GT[i][5], Human, pose_none_flag)  # list with length of 51
        for j in range(length_min):
            origin_poses.append(GT[i][5])
            objclsid.append(GT[i][-1])
            Object_nodeA = [Object_augmented_temp[j][1], Object_augmented_temp[j][2], 1]  # left up corner
            Object_nodeB = [Object_augmented_temp[j][3], Object_augmented_temp[j][4], 1]  # right down corner
            norm_sgraph_in = sgraph_in_norm(sgraph_in + Object_nodeA + Object_nodeB, Human, cfg.POSENORM)
            SGinput.append(norm_sgraph_in)
            skeboxes = np.concatenate([skeboxes, generate_skebox(GT[i][5], Human, shape)], axis=0)
            bodyparts = np.concatenate([bodyparts, generate_bodypart(GT[i][5], Human, shape)], axis=0)

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
                action_HO = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
                origin_poses.append(Neg[7])
                objclsid.append(Neg[5])

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
                action_HO = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
                origin_poses.append(Neg[7])
                objclsid.append(Neg[5])

                Object_nodeA_B = [Neg[3][0], Neg[3][1], Neg[6], Neg[3][2], Neg[3][3], Neg[6]]
                norm_sgraph_in = sgraph_in_norm(Neg_Pose_Nodes + Object_nodeA_B, Neg[2], cfg.POSENORM)
                SGinput.append(norm_sgraph_in)
                skeboxes = np.concatenate([skeboxes, generate_skebox(Neg[7], Neg[2], shape)], axis=0)
                bodyparts = np.concatenate([bodyparts, generate_bodypart(Neg[7], Neg[2], shape)], axis=0)

    num_pos_neg = len(Human_augmented)

    # Semantic Representation
    semantic = []
    for i in objclsid:
        clsname = clsid2cls[i]
        semantic.append(objs_bert[clsname])
    semantic = np.asarray(semantic).reshape(-1, 1024)

    # spatial
    sp_masks = np.empty((0, 64, 64, 3), dtype=np.float32)
    for i in range(num_pos_neg):
        sp_masks_ = Get_next_sp_with_posemap(Human_augmented[i][1:], Object_augmented[i][1:],
                                            origin_poses[i]).reshape(1, 64, 64, 3)
        sp_masks = np.concatenate((sp_masks, sp_masks_), axis=0)
    sp_masks = sp_masks.reshape(num_pos_neg, 64, 64, 3)

    Human_augmented = np.asarray(Human_augmented).reshape(num_pos_neg, 5)
    Object_augmented = np.asarray(Object_augmented).reshape(num_pos_neg, 5)
    action_HO = np.asarray(action_HO).reshape(num_pos_neg, 600)
    SGinput = np.asarray(SGinput).reshape(num_pos_neg, 51 + 6)
    skeboxes = skeboxes.reshape([-1, 17, 5])
    bodyparts = bodyparts.reshape([-1, 6, 5])

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
    return sp_masks, SGinput, semantic, Human_augmented, Object_augmented, skeboxes, bodyparts, \
           action_HO, num_pos, sparse_binary_label, pose_none_flag
