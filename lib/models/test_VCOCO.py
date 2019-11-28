from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.config import cfg
from ult.timer import Timer
from ult.ult import Get_next_sp_with_pose
from ult.ult import Get_next_sp_with_posemap, gnodes_norm
from ult.ult import get_pose_nodes, get_union, generate_pointbox, generate_partbox
from ult.apply_prior import apply_prior
import copy
import os
import cv2
import pickle
import numpy as np


def get_blob(image_id):
    im_file = cfg.DATA_DIR + '/' + 'v-coco/coco/images/val2014/COCO_val2014_' + (str(image_id)).zfill(12) + '.jpg'
    im = cv2.imread(im_file)
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    return im_orig, im_shape


def im_detect(sess, net, image_id, Test_RCNN, prior_mask, Action_dic_inv, object_thres, human_thres, prior_flag,
              detection, posetype):
    im_orig, im_shape = get_blob(image_id)

    blobs = {}

    for Human_out in Test_RCNN[image_id]:
        if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'):  # This is a valid human

            # Predict actrion using human appearance only
            H_box = np.array(
                [0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]).reshape(1, 5)
            #if os.path.exists('Temp/vcoco/test/' + str(image_id) + '.npy'):
            #    continue
            #else:
            #    print(image_id)
            #prediction_H = net.test_image_HO(sess, im_orig, blobs)
            #prediction_H = [prediction_H]
            # save image information

            # all object bboxes in image
            Object_bbox = []
            # classes for objects
            Object_class = []
            # object detection score
            Object_det = []
            # prediction_HO
            HO_Score = []

            dic = dict()
            dic['image_id'] = image_id
            dic['person_box'] = Human_out[2]
            dic['H_det'] = np.max(Human_out[5])

            #dic['H_Score'] = prediction_H

            if posetype == 1:
                c = 2
            else:
                c = 3
            H_num = 0
            H_boxes, O_boxes, U_boxes = np.zeros([0, 5]), np.zeros([0, 5]), np.zeros([0, 5])
            Gnodes = np.zeros([0, 51+6])
            Patterns = np.zeros([0, 64, 64, c])
            pointboxes = np.zeros([0, 17, 5])
            partboxes = np.zeros([0, 6, 5])
            for Object in Test_RCNN[image_id]:
                if (np.max(Object[5]) > object_thres) and not (np.all(Object[2] == Human_out[2])):  # This is a valid object

                    H_num += 1
                    O_box = np.array([0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]]).reshape(1, 5)
                    U_box = get_union(H_box[0], O_box[0])
                    human_pose = copy.deepcopy(Human_out[6])
                    Pose_nodes, pose_none_flag = get_pose_nodes(human_pose, Human_out[2], 1, im_shape)
                    Pattern, gnode = Get_next_sp_with_pose(Human_out[2], Object[2], Pose_nodes, im_shape, Object[5])
                    norm_gnode = gnodes_norm(gnode[0], Human_out[2], cfg.POSENORM).reshape([1, 51+6])
                    if posetype == 2:
                        Pattern = Get_next_sp_with_posemap(Human_out[2], Object[2],
                                                           human_pose, num_joints=17)
                    Pattern = Pattern.reshape(1, 64, 64, -1)
                    Gnodes = np.concatenate([Gnodes, norm_gnode])
                    H_boxes = np.concatenate([H_boxes, H_box], axis=0)
                    O_boxes = np.concatenate([O_boxes, O_box], axis=0)
                    U_boxes = np.concatenate([U_boxes, U_box], axis=0)
                    Patterns = np.concatenate([Patterns, Pattern], axis=0)
                    pointboxes = np.concatenate([pointboxes, generate_pointbox(
                        copy.deepcopy(Human_out[6]), Human_out[2], im_shape)], axis=0)
                    partboxes = np.concatenate([partboxes, generate_partbox(
                        copy.deepcopy(Human_out[6]), Human_out[2], im_shape)], axis=0)

                    Object_bbox.append(Object[2])
                    Object_class.append(Object[4])
                    Object_det.append(np.max(Object[5]))

            if len(Object_bbox) == 0:
                continue

            # all object bboxes in image
            dic['object_box'] = Object_bbox

            # classes for objects
            dic['object_class'] = Object_class

            # object detection score
            dic['O_det'] = Object_det

            blobs['H_num'] = H_num
            blobs['H_boxes'] = H_boxes
            blobs['O_boxes'] = O_boxes
            blobs['U_boxes'] = U_boxes
            blobs['Gnodes'] = Gnodes
            blobs['sp'] = Patterns
            blobs['pointboxes'] = pointboxes
            blobs['partboxes'] = partboxes
            #prediction_HO, prediction_binary = net.test_image_HO(sess, im_orig, blobs)
            if H_num > 0:
                blobs['head'] = np.load('Temp/vcoco/test/' + str(image_id) + '.npy')
                prediction_HO = net.test_image_HO(sess, im_orig, blobs)
                # prediction_HO
                dic['HO_Score'] = prediction_HO[0]
                #if not os.path.exists('Temp/vcoco/test/' + str(image_id) + '.npy'):
                #    print(image_id, 'saved')
                #    np.save('Temp/vcoco/test/' + str(image_id) + '.npy', head)
            # remain iCAN format
            #print(prediction_HO[0].shape)
            #print(prediction_HO.shape)
            
            # Predict actrion using human and object appearance

            Score_obj = np.empty((0, 4 + 29), dtype=np.float32)
            onum = 0
            for Object in Test_RCNN[image_id]:
                if (np.max(Object[5]) > object_thres) and not (np.all(Object[2] == Human_out[2])):
                    #print(i, prediction_HO[0].shape)
                    i_prediction_HO = [prediction_HO[0][onum].reshape(-1, 29)]
                    if prior_flag == 1:
                        i_prediction_HO = apply_prior(Object, i_prediction_HO)
                    if prior_flag == 2:
                        i_prediction_HO = i_prediction_HO * prior_mask[:, Object[4]].reshape(1, 29)
                    if prior_flag == 3:
                        i_prediction_HO = apply_prior(Object, i_prediction_HO)
                        i_prediction_HO = i_prediction_HO * prior_mask[:, Object[4]].reshape(1, 29)

                    This_Score_obj = np.concatenate((Object[2].reshape(1, 4), i_prediction_HO[0] * np.max(Object[5])),
                                                    axis=1)
                    Score_obj = np.concatenate((Score_obj, This_Score_obj), axis=0)
                    onum += 1


            # There is only a single human detected in this image. I just ignore it.
            # Might be better to add Nan as object box.
            if Score_obj.shape[0] == 0:
                continue

            # Find out the object box associated with highest action score
            max_idx = np.argmax(Score_obj, 0)[4:]

            # agent mAP
            for i in range(29):
                # '''
                # walk, smile, run, stand
                if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                    agent_name = Action_dic_inv[i] + '_agent'
                    dic[agent_name] = np.max(Human_out[5]) * 0 #prediction_H[0][0][i]
                    continue

                # cut
                if i == 2:
                    agent_name = 'cut_agent'
                    dic[agent_name] = np.max(Human_out[5]) * max(Score_obj[max_idx[2]][4 + 2],
                                                                 Score_obj[max_idx[4]][4 + 4])
                    continue
                if i == 4:
                    continue

                    # eat
                if i == 9:
                    agent_name = 'eat_agent'
                    dic[agent_name] = np.max(Human_out[5]) * max(Score_obj[max_idx[9]][4 + 9],
                                                                 Score_obj[max_idx[16]][4 + 16])
                    continue
                if i == 16:
                    continue

                # hit
                if i == 19:
                    agent_name = 'hit_agent'
                    dic[agent_name] = np.max(Human_out[5]) * max(Score_obj[max_idx[19]][4 + 19],
                                                                 Score_obj[max_idx[20]][4 + 20])
                    continue
                if i == 20:
                    continue

                # These 2 classes need to save manually because there is '_' in action name
                if i == 6:
                    agent_name = 'talk_on_phone_agent'
                    dic[agent_name] = np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i]
                    continue

                if i == 8:
                    agent_name = 'work_on_computer_agent'
                    dic[agent_name] = np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i]
                    continue

                # all the rest
                agent_name = Action_dic_inv[i].split("_")[0] + '_agent'
                dic[agent_name] = np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i]
                # '''

                '''
                if i == 6:
                    agent_name = 'talk_on_phone_agent'  
                    dic[agent_name] = np.max(Human_out[5]) * prediction_H[0][0][i]
                    continue

                if i == 8:
                    agent_name = 'work_on_computer_agent'  
                    dic[agent_name] = np.max(Human_out[5]) * prediction_H[0][0][i]
                    continue 

                agent_name =  Action_dic_inv[i].split("_")[0] + '_agent'  
                dic[agent_name] = np.max(Human_out[5]) * prediction_H[0][0][i]
                '''

            # role mAP
            for i in range(29):
                # walk, smile, run, stand. Won't contribute to role mAP
                if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                    dic[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1, 4),
                                                       np.max(Human_out[5]) * 0) #prediction_H[0][0][i])
                    continue

                # Impossible to perform this action
                if np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i] == 0:
                    dic[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1, 4),
                                                       np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i])

                # Action with >0 score
                else:
                    dic[Action_dic_inv[i]] = np.append(Score_obj[max_idx[i]][:4],
                                                       np.max(Human_out[5]) * Score_obj[max_idx[i]][4 + i])

            detection.append(dic)


def test_net(sess, net, Test_RCNN, prior_mask, Action_dic_inv, output_dir,
             object_thres, human_thres, prior_flag, posetype):
    np.random.seed(cfg.RNG_SEED)
    detection = []
    count = 0

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    for line in open(cfg.DATA_DIR + '/' + '/v-coco/data/splits/vcoco_test.ids', 'r'):
        _t['im_detect'].tic()

        image_id = int(line.rstrip())

        im_detect(sess, net, image_id, Test_RCNN, prior_mask, Action_dic_inv, object_thres, human_thres, prior_flag,
                  detection, posetype)

        _t['im_detect'].toc()

        print('im_detect: {:d}/{:d} {:.3f}s'.format(count + 1, 4946, _t['im_detect'].average_time))
        count += 1

    pickle.dump(detection, open(output_dir, "wb"))