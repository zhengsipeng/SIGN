import random
from config import cfg
from datatools import *


# =============
# VCOCO
# =============
def Get_Next_Instance_VCOCO(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length, use_pm):
    GT = trainval_GT[iter % Data_length]
    image_id = GT[0][0]
    im_file = cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
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
    #action_HO = action_HO.reshape(num_pos, 29)
    action_HO = action_sp
    action_H = action_H.reshape(num_pos, 29)
    mask_sp = mask_sp.reshape(num_pos_neg, 29)
    #mask_HO = mask_HO.reshape(num_pos, 29)
    mask_HO = mask_sp
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