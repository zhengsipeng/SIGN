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
import time
import cv2
from random import randint


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


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


def sgraph_in_norm(Pose_nodes, Human, Object, pose_norm):
    num_nodes = len(Pose_nodes) / 3
    # Pose_nodes = list
    obj_rela_loc = []
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
        wo = Object[2] - Object[0]
        ho = Object[3] - Object[1]
        for i in range(len(Pose_nodes)):
            if i % 3 == 0:
                obj_rela_loc.append((Pose_nodes[i] - Object[0]) / float(wo))
            if i % 3 == 1:
                obj_rela_loc.append((Pose_nodes[i] - Object[1]) / float(ho))

        w = Human[2] - Human[0]
        h = Human[3] - Human[1]
        for i in range(len(Pose_nodes)):
            if i % 3 == 0:
                Pose_nodes[i] = (Pose_nodes[i] - Human[0]) / float(w)
            elif i % 3 == 1:
                Pose_nodes[i] = (Pose_nodes[i] - Human[1]) / float(h)
    else:
        raise NotImplementedError

    if len(obj_rela_loc) > 0:
        obj_rela_loc = np.asarray(obj_rela_loc).reshape(1, -1, 2)
        Pose_nodes= np.asarray(Pose_nodes).reshape(1, -1, 3)
        Pose_nodes = np.concatenate([Pose_nodes, obj_rela_loc], axis=2)
        Pose_nodes = Pose_nodes.reshape(-1).tolist()
        assert len(Pose_nodes) == num_nodes * 5

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


def Get_next_semantic(clsid, clsid2cls, objs_bert):
    clsname = clsid2cls[clsid]
    semantic = objs_bert[clsname]
    semantic = np.asarray(semantic).reshape(-1, 1024)
    return semantic


def Get_next_sp_coords(human_box, object_box):
    sx1, sy1, sx2, sy2 = human_box.astype(np.float32)
    ox1, oy1, ox2, oy2 = object_box.astype(np.float32)
    sw, sh, ow, oh = sx2-sx1, sy2-sy1, ox2-ox1, oy2-oy1
    xy = np.array([(sx1-ox1)/ow, (sy1-oy1)/oh, (ox1-sx1)/sw, (oy1-sy1)/sh])
    wh = np.log(np.array([sw/ow, sh/oh, ow/sw, oh/sh]))
    return np.hstack((xy, wh)).reshape(-1, 8)


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

    if human_pose is not None and len(human_pose) == 51:
        skeleton = get_skeleton(human_box, human_pose, H, num_joints)
    else:
        skeleton = np.zeros((64, 64, 1), dtype='float32')
        skeleton[int(H[1]):int(H[3]) + 1, int(H[0]):int(H[2]) + 1, 0] = 0.05

    Pattern = np.concatenate((Pattern, skeleton), axis=2)
    return Pattern




