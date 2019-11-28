"""
Change the HICO-DET detection results to the right format.
"""

import pickle
import numpy as np
import scipy.io as sio
import os


# all image index in test set without any pair
all_remaining = set([20, 25, 54, 60, 66, 71, 74, 94, 154, 155, 184, 200, 229, 235, 242, 249, 273, 280, 289, 292, 315, 323, 328, 376, 400, 421, 432, 436, 461, 551, 554, 578, 613, 626, 639, 641, 642, 704, 705, 768, 773, 776, 796, 809, 827, 845, 850, 855, 862, 886, 901, 947, 957, 963, 965, 1003, 1011, 1014, 1028, 1042, 1044, 1057, 1090, 1092, 1097, 1099, 1119, 1171, 1180, 1231, 1241, 1250, 1346, 1359, 1360, 1391, 1420, 1450, 1467, 1495, 1498, 1545, 1560, 1603, 1605, 1624, 1644, 1659, 1673, 1674, 1677, 1709, 1756, 1808, 1845, 1847, 1849, 1859, 1872, 1881, 1907, 1910, 1912, 1914, 1953, 1968, 1979, 2039, 2069, 2106, 2108, 2116, 2126, 2142, 2145, 2146, 2154, 2175, 2184, 2218, 2232, 2269, 2306, 2308, 2316, 2323, 2329, 2390, 2397, 2406, 2425, 2463, 2475, 2483, 2494, 2520, 2576, 2582, 2591, 2615, 2624, 2642, 2646, 2677, 2703, 2707, 2712, 2717, 2763, 2780, 2781, 2818, 2830, 2833, 2850, 2864, 2873, 2913, 2961, 2983, 3021, 3040, 3042, 3049, 3057, 3066, 3082, 3083, 3111, 3112, 3122, 3157, 3200, 3204, 3229, 3293, 3309, 3328, 3341, 3373, 3393, 3423, 3439, 3449, 3471, 3516, 3525, 3537, 3555, 3616, 3636, 3653, 3668, 3681, 3709, 3718, 3719, 3733, 3737, 3744, 3756, 3762, 3772, 3780, 3784, 3816, 3817, 3824, 3855, 3865, 3885, 3891, 3910, 3916, 3918, 3919, 3933, 3949, 3980, 4009, 4049, 4066, 4089, 4112, 4143, 4154, 4200, 4222, 4243, 4254, 4257, 4259, 4266, 4269, 4273, 4308, 4315, 4320, 4331, 4343, 4352, 4356, 4369, 4384, 4399, 4411, 4424, 4428, 4445, 4447, 4466, 4477, 4482, 4492, 4529, 4534, 4550, 4566, 4596, 4605, 4606, 4620, 4648, 4710, 4718, 4734, 4771, 4773, 4774, 4801, 4807, 4811, 4842, 4845, 4849, 4874, 4886, 4887, 4907, 4926, 4932, 4948, 4960, 4969, 5000, 5039, 5042, 5105, 5113, 5159, 5161, 5174, 5183, 5197, 5214, 5215, 5216, 5221, 5264, 5273, 5292, 5293, 5353, 5438, 5447, 5452, 5465, 5468, 5492, 5498, 5520, 5543, 5551, 5575, 5581, 5605, 5617, 5623, 5671, 5728, 5759, 5766, 5777, 5799, 5840, 5853, 5875, 5883, 5886, 5898, 5919, 5922, 5941, 5948, 5960, 5962, 5964, 6034, 6041, 6058, 6080, 6103, 6117, 6134, 6137, 6138, 6163, 6196, 6206, 6210, 6223, 6228, 6232, 6247, 6272, 6273, 6281, 6376, 6409, 6430, 6438, 6473, 6496, 6595, 6608, 6635, 6678, 6687, 6692, 6695, 6704, 6712, 6724, 6757, 6796, 6799, 6815, 6851, 6903, 6908, 6914, 6948, 6957, 7065, 7071, 7073, 7089, 7099, 7102, 7114, 7147, 7169, 7185, 7219, 7226, 7232, 7271, 7285, 7315, 7323, 7341, 7378, 7420, 7433, 7437, 7467, 7489, 7501, 7513, 7514, 7523, 7534, 7572, 7580, 7614, 7619, 7625, 7658, 7667, 7706, 7719, 7727, 7752, 7813, 7826, 7829, 7868, 7872, 7887, 7897, 7902, 7911, 7936, 7942, 7945, 8032, 8034, 8042, 8044, 8092, 8101, 8156, 8167, 8175, 8176, 8205, 8234, 8237, 8244, 8301, 8316, 8326, 8350, 8362, 8385, 8441, 8463, 8479, 8534, 8565, 8610, 8623, 8651, 8671, 8678, 8689, 8707, 8735, 8761, 8763, 8770, 8779, 8800, 8822, 8835, 8923, 8942, 8962, 8970, 8984, 9010, 9037, 9041, 9122, 9136, 9140, 9147, 9164, 9165, 9166, 9170, 9173, 9174, 9175, 9185, 9186, 9200, 9210, 9211, 9217, 9218, 9246, 9248, 9249, 9250, 9254, 9307, 9332, 9337, 9348, 9364, 9371, 9376, 9379, 9389, 9404, 9405, 9408, 9415, 9416, 9417, 9418, 9419, 9421, 9424, 9433, 9434, 9493, 9501, 9505, 9519, 9520, 9521, 9522, 9526, 9529, 9531, 9637, 9654, 9655, 9664, 9686, 9688, 9701, 9706, 9709, 9712, 9716, 9717, 9718, 9731, 9746, 9747, 9748, 9753, 9765])

def save_HICO(HICO, HICO_dir, classid, begin, finish):
    all_boxes = []
    for i in range(finish - begin + 1):
        total = []
        score = []
        for key, value in HICO.iteritems():
            if int(key) in all_remaining:
                continue
            for element in value:
                if element[2] == classid:
                    temp = []
                    temp.append(element[0].tolist())  # Human box
                    temp.append(element[1].tolist())  # Object box
                    temp.append(int(key))  # image id
                    temp.append(int(i))  # action id (0-599)
                    temp.append(element[3][begin - 1 + i] * element[4] * element[5])
                    total.append(temp)
                    score.append(element[3][begin - 1 + i] * element[4] * element[5])

        idx = np.argsort(score, axis=0)[::-1]
        for i_idx in range(min(len(idx), 19999)):
            all_boxes.append(total[idx[i_idx]])
    savefile = HICO_dir + 'detections_' + str(classid).zfill(2) + '.mat'
    sio.savemat(savefile, {'all_boxes': all_boxes})


def Generate_HICO_detection(output_file, HICO_dir):
    if not os.path.exists(HICO_dir):
        os.makedirs(HICO_dir)

    # Remove previous results
    filelist = [f for f in os.listdir(HICO_dir)]
    for f in filelist:
        os.remove(os.path.join(HICO_dir, f))

    HICO = pickle.load(open(output_file, "rb"))

    save_HICO(HICO, HICO_dir, 1, 161, 170)  # 1 person
    save_HICO(HICO, HICO_dir, 2, 11, 24)  # 2 bicycle
    save_HICO(HICO, HICO_dir, 3, 66, 76)  # 3 car
    save_HICO(HICO, HICO_dir, 4, 147, 160)  # 4 motorcycle
    save_HICO(HICO, HICO_dir, 5, 1, 10)  # 5 airplane
    save_HICO(HICO, HICO_dir, 6, 55, 65)  # 6 bus
    save_HICO(HICO, HICO_dir, 7, 187, 194)  # 7 train
    save_HICO(HICO, HICO_dir, 8, 568, 576)  # 8 truck
    save_HICO(HICO, HICO_dir, 9, 32, 46)  # 9 boat
    save_HICO(HICO, HICO_dir, 10, 563, 567)  # 10 traffic light
    save_HICO(HICO, HICO_dir, 11, 326, 330)  # 11 fire_hydrant
    save_HICO(HICO, HICO_dir, 12, 503, 506)  # 12 stop_sign
    save_HICO(HICO, HICO_dir, 13, 415, 418)  # 13 parking_meter
    save_HICO(HICO, HICO_dir, 14, 244, 247)  # 14 bench
    save_HICO(HICO, HICO_dir, 15, 25, 31)  # 15 bird
    save_HICO(HICO, HICO_dir, 16, 77, 86)  # 16 cat
    save_HICO(HICO, HICO_dir, 17, 112, 129)  # 17 dog
    save_HICO(HICO, HICO_dir, 18, 130, 146)  # 18 horse
    save_HICO(HICO, HICO_dir, 19, 175, 186)  # 19 sheep
    save_HICO(HICO, HICO_dir, 20, 97, 107)  # 20 cow
    save_HICO(HICO, HICO_dir, 21, 314, 325)  # 21 elephant
    save_HICO(HICO, HICO_dir, 22, 236, 239)  # 22 bear
    save_HICO(HICO, HICO_dir, 23, 596, 600)  # 23 zebra
    save_HICO(HICO, HICO_dir, 24, 343, 348)  # 24 giraffe
    save_HICO(HICO, HICO_dir, 25, 209, 214)  # 25 backpack
    save_HICO(HICO, HICO_dir, 26, 577, 584)  # 26 umbrella
    save_HICO(HICO, HICO_dir, 27, 353, 356)  # 27 handbag
    save_HICO(HICO, HICO_dir, 28, 539, 546)  # 28 tie
    save_HICO(HICO, HICO_dir, 29, 507, 516)  # 29 suitcase
    save_HICO(HICO, HICO_dir, 30, 337, 342)  # 30 Frisbee
    save_HICO(HICO, HICO_dir, 31, 464, 474)  # 31 skis
    save_HICO(HICO, HICO_dir, 32, 475, 483)  # 32 snowboard
    save_HICO(HICO, HICO_dir, 33, 489, 502)  # 33 sports_ball
    save_HICO(HICO, HICO_dir, 34, 369, 376)  # 34 kite
    save_HICO(HICO, HICO_dir, 35, 225, 232)  # 35 baseball_bat
    save_HICO(HICO, HICO_dir, 36, 233, 235)  # 36 baseball_glove
    save_HICO(HICO, HICO_dir, 37, 454, 463)  # 37 skateboard
    save_HICO(HICO, HICO_dir, 38, 517, 528)  # 38 surfboard
    save_HICO(HICO, HICO_dir, 39, 534, 538)  # 39 tennis_racket
    save_HICO(HICO, HICO_dir, 40, 47, 54)  # 40 bottle
    save_HICO(HICO, HICO_dir, 41, 589, 595)  # 41 wine_glass
    save_HICO(HICO, HICO_dir, 42, 296, 305)  # 42 cup
    save_HICO(HICO, HICO_dir, 43, 331, 336)  # 43 fork
    save_HICO(HICO, HICO_dir, 44, 377, 383)  # 44 knife
    save_HICO(HICO, HICO_dir, 45, 484, 488)  # 45 spoon
    save_HICO(HICO, HICO_dir, 46, 253, 257)  # 46 bowl
    save_HICO(HICO, HICO_dir, 47, 215, 224)  # 47 banana
    save_HICO(HICO, HICO_dir, 48, 199, 208)  # 48 apple
    save_HICO(HICO, HICO_dir, 49, 439, 445)  # 49 sandwich
    save_HICO(HICO, HICO_dir, 50, 398, 407)  # 50 orange
    save_HICO(HICO, HICO_dir, 51, 258, 264)  # 51 broccoli
    save_HICO(HICO, HICO_dir, 52, 274, 283)  # 52 carrot
    save_HICO(HICO, HICO_dir, 53, 357, 363)  # 53 hot_dog
    save_HICO(HICO, HICO_dir, 54, 419, 429)  # 54 pizza
    save_HICO(HICO, HICO_dir, 55, 306, 313)  # 55 donut
    save_HICO(HICO, HICO_dir, 56, 265, 273)  # 56 cake
    save_HICO(HICO, HICO_dir, 57, 87, 92)  # 57 chair
    save_HICO(HICO, HICO_dir, 58, 93, 96)  # 58 couch
    save_HICO(HICO, HICO_dir, 59, 171, 174)  # 59 potted_plant
    save_HICO(HICO, HICO_dir, 60, 240, 243)  # 60 bed
    save_HICO(HICO, HICO_dir, 61, 108, 111)  # 61 dining_table
    save_HICO(HICO, HICO_dir, 62, 551, 558)  # 62 toilet
    save_HICO(HICO, HICO_dir, 63, 195, 198)  # 63 TV
    save_HICO(HICO, HICO_dir, 64, 384, 389)  # 64 laptop
    save_HICO(HICO, HICO_dir, 65, 394, 397)  # 65 mouse
    save_HICO(HICO, HICO_dir, 66, 435, 438)  # 66 remote
    save_HICO(HICO, HICO_dir, 67, 364, 368)  # 67 keyboard
    save_HICO(HICO, HICO_dir, 68, 284, 290)  # 68 cell_phone
    save_HICO(HICO, HICO_dir, 69, 390, 393)  # 69 microwave
    save_HICO(HICO, HICO_dir, 70, 408, 414)  # 70 oven
    save_HICO(HICO, HICO_dir, 71, 547, 550)  # 71 toaster
    save_HICO(HICO, HICO_dir, 72, 450, 453)  # 72 sink
    save_HICO(HICO, HICO_dir, 73, 430, 434)  # 73 refrigerator
    save_HICO(HICO, HICO_dir, 74, 248, 252)  # 74 book
    save_HICO(HICO, HICO_dir, 75, 291, 295)  # 75 clock
    save_HICO(HICO, HICO_dir, 76, 585, 588)  # 76 vase
    save_HICO(HICO, HICO_dir, 77, 446, 449)  # 77 scissors
    save_HICO(HICO, HICO_dir, 78, 529, 533)  # 78 teddy_bear
    save_HICO(HICO, HICO_dir, 79, 349, 352)  # 79 hair_drier
    save_HICO(HICO, HICO_dir, 80, 559, 562)  # 80 toothbrush
