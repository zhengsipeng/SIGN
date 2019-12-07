from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.config import cfg
from utils.datatools import Get_next_sp_with_posemap
from utils.datatools import Get_next_semantic
from utils.datatools import get_sgraph_in, sgraph_in_norm
from utils.datatools import get_union, generate_skebox, generate_bodypart, Timer
import cv2
import pickle
import numpy as np
import glob
import copy
human_num_thres = 4
object_num_thres = 4

classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
           'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
           'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
           'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
           'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
           'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
           'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cellphone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
           'teddy bear', 'hair drier', 'toothbrush']
clsid2cls = dict()
for k in range(len(classes)):
    clsid2cls[k+1] = classes[k]

all_remaining = set(
    [20, 25, 54, 60, 66, 71, 74, 94, 154, 155, 184, 200, 229, 235, 242, 249, 273, 280, 289, 292, 315, 323, 328, 376,
     400, 421, 432, 436, 461, 551, 554, 578, 613, 626, 639, 641, 642, 704, 705, 768, 773, 776, 796, 809, 827, 845, 850,
     855, 862, 886, 901, 947, 957, 963, 965, 1003, 1011, 1014, 1028, 1042, 1044, 1057, 1090, 1092, 1097, 1099, 1119,
     1171, 1180, 1231, 1241, 1250, 1346, 1359, 1360, 1391, 1420, 1450, 1467, 1495, 1498, 1545, 1560, 1603, 1605, 1624,
     1644, 1659, 1673, 1674, 1677, 1709, 1756, 1808, 1845, 1847, 1849, 1859, 1872, 1881, 1907, 1910, 1912, 1914, 1953,
     1968, 1979, 2039, 2069, 2106, 2108, 2116, 2126, 2142, 2145, 2146, 2154, 2175, 2184, 2218, 2232, 2269, 2306, 2308,
     2316, 2323, 2329, 2390, 2397, 2406, 2425, 2463, 2475, 2483, 2494, 2520, 2576, 2582, 2591, 2615, 2624, 2642, 2646,
     2677, 2703, 2707, 2712, 2717, 2763, 2780, 2781, 2818, 2830, 2833, 2850, 2864, 2873, 2913, 2961, 2983, 3021, 3040,
     3042, 3049, 3057, 3066, 3082, 3083, 3111, 3112, 3122, 3157, 3200, 3204, 3229, 3293, 3309, 3328, 3341, 3373, 3393,
     3423, 3439, 3449, 3471, 3516, 3525, 3537, 3555, 3616, 3636, 3653, 3668, 3681, 3709, 3718, 3719, 3733, 3737, 3744,
     3756, 3762, 3772, 3780, 3784, 3816, 3817, 3824, 3855, 3865, 3885, 3891, 3910, 3916, 3918, 3919, 3933, 3949, 3980,
     4009, 4049, 4066, 4089, 4112, 4143, 4154, 4200, 4222, 4243, 4254, 4257, 4259, 4266, 4269, 4273, 4308, 4315, 4320,
     4331, 4343, 4352, 4356, 4369, 4384, 4399, 4411, 4424, 4428, 4445, 4447, 4466, 4477, 4482, 4492, 4529, 4534, 4550,
     4566, 4596, 4605, 4606, 4620, 4648, 4710, 4718, 4734, 4771, 4773, 4774, 4801, 4807, 4811, 4842, 4845, 4849, 4874,
     4886, 4887, 4907, 4926, 4932, 4948, 4960, 4969, 5000, 5039, 5042, 5105, 5113, 5159, 5161, 5174, 5183, 5197, 5214,
     5215, 5216, 5221, 5264, 5273, 5292, 5293, 5353, 5438, 5447, 5452, 5465, 5468, 5492, 5498, 5520, 5543, 5551, 5575,
     5581, 5605, 5617, 5623, 5671, 5728, 5759, 5766, 5777, 5799, 5840, 5853, 5875, 5883, 5886, 5898, 5919, 5922, 5941,
     5948, 5960, 5962, 5964, 6034, 6041, 6058, 6080, 6103, 6117, 6134, 6137, 6138, 6163, 6196, 6206, 6210, 6223, 6228,
     6232, 6247, 6272, 6273, 6281, 6376, 6409, 6430, 6438, 6473, 6496, 6595, 6608, 6635, 6678, 6687, 6692, 6695, 6704,
     6712, 6724, 6757, 6796, 6799, 6815, 6851, 6903, 6908, 6914, 6948, 6957, 7065, 7071, 7073, 7089, 7099, 7102, 7114,
     7147, 7169, 7185, 7219, 7226, 7232, 7271, 7285, 7315, 7323, 7341, 7378, 7420, 7433, 7437, 7467, 7489, 7501, 7513,
     7514, 7523, 7534, 7572, 7580, 7614, 7619, 7625, 7658, 7667, 7706, 7719, 7727, 7752, 7813, 7826, 7829, 7868, 7872,
     7887, 7897, 7902, 7911, 7936, 7942, 7945, 8032, 8034, 8042, 8044, 8092, 8101, 8156, 8167, 8175, 8176, 8205, 8234,
     8237, 8244, 8301, 8316, 8326, 8350, 8362, 8385, 8441, 8463, 8479, 8534, 8565, 8610, 8623, 8651, 8671, 8678, 8689,
     8707, 8735, 8761, 8763, 8770, 8779, 8800, 8822, 8835, 8923, 8942, 8962, 8970, 8984, 9010, 9037, 9041, 9122, 9136,
     9140, 9147, 9164, 9165, 9166, 9170, 9173, 9174, 9175, 9185, 9186, 9200, 9210, 9211, 9217, 9218, 9246, 9248, 9249,
     9250, 9254, 9307, 9332, 9337, 9348, 9364, 9371, 9376, 9379, 9389, 9404, 9405, 9408, 9415, 9416, 9417, 9418, 9419,
     9421, 9424, 9433, 9434, 9493, 9501, 9505, 9519, 9520, 9521, 9522, 9526, 9529, 9531, 9637, 9654, 9655, 9664, 9686,
     9688, 9701, 9706, 9709, 9712, 9716, 9717, 9718, 9731, 9746, 9747, 9748, 9753, 9765])


def get_blob(image_id):
    im_file = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/HICO_test2015_' + (str(image_id)).zfill(
        8) + '.jpg'
    im = cv2.imread(im_file)
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    return im_orig, im_shape


def im_detect(sess, net, image_id, Test_RCNN, object_thres, human_thres, detection):
    This_image = []
    if int(image_id) in all_remaining:
        return 0

    im_orig, im_shape = get_blob(image_id)
    objs_bert = pickle.load(open(cfg.DATA_DIR + '/' + 'objs_bert1024.pkl', 'rb'))
    blobs = dict()
    blobs['H_num'] = 1
    blobs['head'] = np.load('Temp/test/'+str(image_id)+'.npy')

    Hnum = 0
    H_boxes, O_boxes, U_boxes = np.zeros([0, 5]), np.zeros([0, 5]), np.zeros([0, 5])
    spatial = np.zeros([0, 64, 64, 3])
    SGinput = np.zeros([0, 19*3 + 19*2])
    skeboxes = np.zeros([0, 17, 5])
    bodyparts = np.zeros([0, 6, 5])
    semantic = np.zeros([0, 768])
    obj_class, hscore, oscore = [], [], []

    for Human_out in Test_RCNN[image_id]:
        if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'):  # This is a valid human
            H_box = np.array([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]).reshape(1, 5)
            for Object in Test_RCNN[image_id]:
                if (np.max(Object[5] > object_thres)) and not (np.all(Object[2] == Human_out[2])):  # valid object
                    Hnum += 1
                    O_box = np.array([0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]]).reshape(1, 5)
                    U_box = get_union(H_box[0], O_box[0])
                    posenodes, pose_none_flag = get_sgraph_in(copy.deepcopy(Human_out[6]), Human_out[2], 1)
                    sgraph_in = posenodes + [Object[2][0], Object[2][1], Object[5], Object[2][2], Object[2][3], Object[5]]
                    norm_gnode = sgraph_in_norm(sgraph_in, Human_out[2], cfg.POSENORM).reshape([1, 51 + 6])
                    spatial_ = Get_next_sp_with_posemap(Human_out[2], Object[2], copy.deepcopy(Human_out[6]), num_joints=17)
                    objname = clsid2cls[Object[4]]
                    semantic_ = np.asarray(objs_bert[objname]).reshape(-1, 768)

                    SGinput = np.concatenate([SGinput, norm_gnode], axis=0)
                    H_boxes = np.concatenate([H_boxes, H_box], axis=0)
                    O_boxes = np.concatenate([O_boxes, O_box], axis=0)
                    U_boxes = np.concatenate([U_boxes, U_box], axis=0)
                    spatial = np.concatenate([spatial, spatial_.reshape(1, 64, 64, 3)], axis=0)
                    semantic = np.concatenate([semantic, semantic_], axis=0)
                    skeboxes = np.concatenate([skeboxes, generate_skebox(copy.deepcopy(Human_out[6]),
                                                                         Human_out[2], im_shape)], axis=0)
                    bodyparts = np.concatenate([bodyparts, generate_bodypart(copy.deepcopy(Human_out[6]),
                                                                             Human_out[2], im_shape)], axis=0)
                    obj_class.append(Object[4])
                    hscore.append(Human_out[5])
                    oscore.append(Object[5])

    if Hnum == 0:
        print('Dealing with zero-sample test Image' + str(image_id))
        list_human_included = []
        list_object_included = []
        Human_out_list = []
        Object_list = []

        test_pair_all = Test_RCNN[image_id]  # bounding box num
        length = len(test_pair_all)

        while (len(list_human_included) < human_num_thres) or (len(list_object_included) < object_num_thres):
            h_max = [-1, -1.]
            o_max = [-1, -1.]
            flag_continue_searching = 0
            for i in range(length):
                if test_pair_all[i][1] == 'Human':  # search the top 4 human
                    if (np.max(test_pair_all[i][5]) > h_max[1]) and not (i in list_human_included) \
                            and len(list_human_included) < human_num_thres:
                        h_max = [i, np.max(test_pair_all[i][5])]
                        flag_continue_searching = 1
                else:  # search the top 4 object
                    if np.max(test_pair_all[i][5]) > o_max[1] and not (i in list_object_included) \
                            and len(list_object_included) < object_num_thres:
                        o_max = [i, np.max(test_pair_all[i][5])]
                        flag_continue_searching = 1
            if flag_continue_searching == 0:
                break

            list_human_included.append(h_max[0])
            list_object_included.append(o_max[0])

            Human_out_list.append(test_pair_all[h_max[0]])
            Object_list.append(test_pair_all[o_max[0]])

        for Human_out in Human_out_list:
            for Object in Object_list:
                Hnum += 1
                H_box = np.array([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]).reshape(1, 5)
                O_box = np.array([0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]]).reshape(1, 5)
                U_box = get_union(H_box[0], O_box[0])
                posenodes, pose_none_flag = get_sgraph_in(copy.deepcopy(Human_out[6]), Human_out[2], 1)
                sgraph_in = posenodes + [Object[2][0], Object[2][1], Object[5], Object[2][2], Object[2][3], Object[5]]
                norm_gnode = np.asarray(sgraph_in_norm(sgraph_in, Human_out[2], cfg.POSENORM)).reshape([1, 51 + 6])
                spatial_ = Get_next_sp_with_posemap(Human_out[2], Object[2], copy.deepcopy(Human_out[6]), num_joints=17)
                objname = clsid2cls[Object[4]]
                semantic_ = np.asarray(objs_bert[objname]).reshape(-1, 768)

                SGinput = np.concatenate([SGinput, norm_gnode], axis=0)
                H_boxes = np.concatenate([H_boxes, H_box], axis=0)
                O_boxes = np.concatenate([O_boxes, O_box], axis=0)
                U_boxes = np.concatenate([U_boxes, U_box], axis=0)
                spatial = np.concatenate([spatial, spatial_.reshape(1, 64, 64, 3)], axis=0)
                semantic = np.concatenate([semantic, semantic_], axis=0)
                skeboxes = np.concatenate([skeboxes, generate_skebox(copy.deepcopy(Human_out[6]),
                                                                     Human_out[2], im_shape)], axis=0)
                bodyparts = np.concatenate([bodyparts, generate_bodypart(copy.deepcopy(Human_out[6]),
                                                                         Human_out[2], im_shape)], axis=0)
                obj_class.append(Object[4])
                hscore.append(Human_out[5])
                oscore.append(Object[5])

    H_boxes = H_boxes.reshape([-1, 5])
    O_boxes = O_boxes.reshape([-1, 5])
    U_boxes = U_boxes.reshape([-1, 5])
    spatial = spatial.reshape([-1, 64, 64, 3])
    SGinput = SGinput.reshape([-1, (17+2)*5])
    skeboxes = skeboxes.reshape([-1, 17, 5])
    bodyparts = bodyparts.reshape(-1, 6, 5)
    semantic = semantic.reshape([-1, 768])
    if Hnum != 0:
        divide = int(Hnum / 300) + 1
        for i in range(divide):
            start = i * 300
            end = min((i + 1) * 300, Hnum)
            num = end - start
            if num == 0:
                continue
            blobs = {'H_num': num,
                     'H_boxes': H_boxes[start: end], 'O_boxes': O_boxes[start: end], 'U_boxes': U_boxes[start: end],
                     'skeboxes': skeboxes[start: end], 'bodyparts': bodyparts[start: end],
                     'sp': spatial[start: end], 'semantic': semantic[start: end], 'SGinput': SGinput[start: end],
                     'head': np.load('Temp/test/' + str(image_id) + '.npy')}
            prediction_HO = net.test_image_HO(sess, im_orig, blobs)
            for j in range(num):
                # Human box, Object box, Object class, prediction score, Human scoe, Object score
                temp = [blobs['H_boxes'][j, 1:], blobs['O_boxes'][j, 1:], obj_class[i * 300 + j],
                        prediction_HO[0][j], hscore[i * 300 + j], oscore[i * 300 + j]]
                This_image.append(temp)
    detection[image_id] = This_image
    return len(This_image)


def test_net(sess, net, Test_RCNN, output_dir, object_thres, human_thres):
    np.random.seed(cfg.RNG_SEED)
    detection = {}
    count = 0
    _t = {'im_detect': Timer(), 'misc': Timer()}
    total_num = 0
    imagekeys = Test_RCNN.keys()
    for line in glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/*.jpg'):
        _t['im_detect'].tic()
        image_id = int(line[-9: -4])
        if image_id not in imagekeys:
            continue
        pair_num = im_detect(sess, net, image_id, Test_RCNN, object_thres, human_thres, detection)
        total_num += pair_num
        print('im_detect: {:d}/{:d} pair_num {:d}/total_num {:d}, {:.3f}s'.format(
            count + 1, 9658, pair_num, total_num, _t['im_detect'].average_time))
        count += 1
    pickle.dump(detection, open(output_dir, 'wb'))
