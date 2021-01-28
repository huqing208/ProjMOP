import numpy as np
import cv2
import random
import math
import time
import Koutu
import Julei

"""
约定旋转最小单位
平移最小单位
缩小最小单位
放大最小单位
即每次变换都由若干个最小单位组成
"""
ANGLE_UNIT = 0.5
TRANSLATION_UNIT = 1
SCALE_SMALL_UNIT = 0.99
SCALE_BIG_UNIT = 1.01
"""
约定干扰数目占总数的最少最多数目
是否显示中间过程生产的图片（是否写磁盘）
"""
DISTURBESUM_MIN = 1 / 8
DISTURBESUM_MAX = 1 / 5
SHOW_TEMP_IMG = True
"""
针对旋转变化特供，考虑到模拟抖动的话应该要加上旋转中心抖动
是否开旋转中心抖动
水平方向抖动的距离（占图片宽度的百分比）
垂直方向抖动的距离（占图片高度的百分比）
"""
ROTATE_CENTER_DISTURBE = True
ROTATE_CENTER_DISTURBE_X = 1 / 200
ROTATE_CENTER_DISTURBE_Y = 1 / 200

"""
图像灯光预处理的阈值是最亮值的占比
增强（减弱）的因子
"""
THRESHOLD_RATE = 0.5
STRENENGTHEN = 0.2


# 平移
def get_translation(step_x, step_y):
    """
    :param step_x: 水平移动距离
    :param step_y: 垂直移动距离
    :return: 平移变换矩阵
    """
    t = [[1, 0, step_x], [0, 1, step_y]]
    t = np.array(t, dtype=np.float32)
    return t


# 初始化Trans类型列表，可以用对应的下标选取变换类型
def init_trans(original_img):
    """
    :param original_img: 图片，为了使用图片的尺寸参数
    :return: 列表，包含9个基础变换，0~1为逆、顺时针旋转，2~5为上、下、左、右平移，6~7为缩小、放大,8停顿（空操作）
    """
    t = []
    # 旋转因子
    # 逆时针
    rotateFactor1 = cv2.getRotationMatrix2D((original_img.shape[1] / 2, original_img.shape[0] / 2), ANGLE_UNIT, 1)
    t.append(rotateFactor1)
    # 顺时针
    rotateFactor2 = cv2.getRotationMatrix2D((original_img.shape[1] / 2, original_img.shape[0] / 2), -ANGLE_UNIT, 1)
    t.append(rotateFactor2)
    # 平移因子
    # 上移
    translationFactor1 = get_translation(0, TRANSLATION_UNIT)
    t.append(translationFactor1)
    # 下移
    translationFactor2 = get_translation(0, -TRANSLATION_UNIT)
    t.append(translationFactor2)
    # 左移
    translationFactor3 = get_translation(-TRANSLATION_UNIT, 0)
    t.append(translationFactor3)
    # 右移
    translationFactor4 = get_translation(TRANSLATION_UNIT, 0)
    t.append(translationFactor4)
    # 缩小因子
    scaleSmallFactor = cv2.getRotationMatrix2D((original_img.shape[1] / 2, original_img.shape[0] / 2), 0, SCALE_SMALL_UNIT)
    t.append(scaleSmallFactor)
    # 放大因子
    scaleBigFactor = cv2.getRotationMatrix2D((original_img.shape[1] / 2, original_img.shape[0] / 2), 0, SCALE_BIG_UNIT)
    t.append(scaleBigFactor)
    # 空操作模拟停顿
    t.append(None)
    return t


# 获得扰动变换矩阵
def get_disturbe_arr(trans_num, disturbe_num, disturbe_type=None):
    """
    :param trans_num: 变换次数
    :param disturbe_num: 扰动变换次数
    :param disturbe_type: 扰动类型列表
    :return: 扰动变换矩阵
    """
    # 抽取扰动次数个随机变换，扰动类型在列表中，初始值设为-1，并转成列矩阵
    disturbe_arr = -np.ones(trans_num, dtype=int, order='C').reshape(-1, 1)
    random_index = np.random.choice(trans_num, disturbe_num, replace='False')  # 随机抽取扰动下标，无重复
    for index in random_index:
        disturbe = random.randint(0, 7)
        # 扰动，保证非旋转变换
        if 0:
            while disturbe == rotate_trans_type:
                disturbe = random.randint(0, 7)
        else:
            if disturbe_type == None:
                # 直接类似摇骰子吧，0-11给平移，12-15给停顿，16-18给旋转，19给缩放，控制权重
                weight = random.randint(0, 19)
                if weight in [0, 11]:
                    disturbe = random.randint(2, 5)
                elif weight in [12, 15]:
                    disturbe = 8
                elif weight in [16, 18]:
                    disturbe = random.randint(0, 1)
                elif weight == None:
                    disturbe = random.randint(6, 7)
                else:
                    disturbe = -1
            else:
                disturbe_index = random.randint(0, len(disturbe_type) - 1)
                disturbe = disturbe_type[disturbe_index]

        disturbe_arr[index] = disturbe
    return disturbe_arr


def get_rotate_type_nums(angle):
    """
    :param angle: 旋转角度
    :return: 旋转类型，变换操作次数
    """
    # 判断是顺时针还是逆时针
    rotate_trans_type = 0
    if angle > 0:
        rotate_trans_type = 0
    elif angle < 0:
        rotate_trans_type = 1
    trans_num = int(angle / ANGLE_UNIT)
    return rotate_trans_type, trans_num


def get_trans_array_withDisturbe_rotate(angle):
    """
    :param angle: 旋转角度
    :return: 旋转基础变换列表, 扰动变换列表, 变换操作的次数
    """
    if angle == 0:
        return None, None, 0

    # 变换操作的次数，拆成若干次最小单位变换,并加上扰动次数
    rotate_trans_type, trans_num = get_rotate_type_nums(angle)
    disturbe_num = random.randint(int(trans_num * DISTURBESUM_MIN), int(trans_num * DISTURBESUM_MAX))

    disturbe_arr = get_disturbe_arr(trans_num, disturbe_num)
    # 生成旋转+扰动变换
    trans = rotate_trans_type * np.ones(trans_num, dtype=int, order='C').reshape(-1, 1)  # 第一列保证原始变换
    return trans, disturbe_arr, trans_num


def get_translation_type_nums(dx, dy):
    """
    :param dx: 水平位移
    :param dy: 垂直位移
    :return: （水平平移类型、次数）,（垂直平移类型、次数）
    """
    # 计算水平平移类型、次数和垂直平移类型、次数
    translation_trans_type_x = 0
    trans_num_x = int(abs(dx) / TRANSLATION_UNIT)
    translation_trans_type_y = 0
    trans_num_y = int(abs(dy) / TRANSLATION_UNIT)
    if dx > 0:
        translation_trans_type_x = 5
    elif dx < 0:
        translation_trans_type_x = 4
    else:
        translation_trans_type_x = None
    if dy > 0:
        translation_trans_type_y = 2
    elif dy < 0:
        translation_trans_type_y = 3
    else:
        translation_trans_type_y = None
    return (translation_trans_type_x, trans_num_x), (translation_trans_type_y, trans_num_y)


def get_trans_array_withDisturbe_translation(dx, dy):
    """
    :param dx: 水平位移
    :param dy: 垂直位移
    :return: 平移基础变换列表, 扰动变换列表, 变换操作的次数
    """
    # 获得平移类型、次数
    (translation_trans_type_x, trans_num_x), (translation_trans_type_y, trans_num_y) = get_translation_type_nums(dx, dy)
    if translation_trans_type_x is None and translation_trans_type_y is None:
        return None, None, 0
    elif translation_trans_type_x is not None and translation_trans_type_y is not None:
        # 双向平移
        trans = get_mixed_trans_array(2, translation_trans_type_x, translation_trans_type_y, trans_num_x, trans_num_y)
        trans_num = trans_num_x + trans_num_y
        disturbe_num = random.randint(int(trans_num * DISTURBESUM_MIN), int(trans_num * DISTURBESUM_MAX))

        disturbe_arr = get_disturbe_arr(trans_num, disturbe_num)
        return trans, disturbe_arr, trans_num
    else:
        # 单向平移
        if translation_trans_type_x is None:
            translationDrection = translation_trans_type_y
            trans_num = trans_num_y
        else:
            translationDrection = translation_trans_type_x
            trans_num = trans_num_x
        # 变换操作的次数，拆成若干次最小单位变换,并加上扰动次数
        disturbe_num = random.randint(int(trans_num * DISTURBESUM_MIN), int(trans_num * DISTURBESUM_MAX))

        disturbe_arr = get_disturbe_arr(trans_num, disturbe_num)

        # 生成平移变换
        trans = translationDrection * np.ones(trans_num, dtype=int, order='C').reshape(-1, 1)  # 第一列保证原始变换
        return trans, disturbe_arr, trans_num


def get_scale_type_nums(scale_ratio):
    """
    :param scale_ratio: 缩放系数
    :return: 缩放类型，变换操作次数
    """
    # 判断放大缩小
    scale_trans_type = 0
    trans_sum = 0  # 变换操作的次数，拆成若干次最小单位变换
    # 判断是顺时针还是逆时针
    if scale_ratio < 1:
        scale_trans_type = 6
        trans_num = int(math.log(scale_ratio) / math.log(SCALE_SMALL_UNIT))
    elif scale_ratio > 1:
        scale_trans_type = 7
        trans_num = int(math.log(scale_ratio) / math.log(SCALE_BIG_UNIT))
    return scale_trans_type, trans_num


def get_trans_array_withDisturbe_scale(scale_ratio):
    """
    :param scale_ratio: 缩放系数
    :return: 缩放基础变换列表, 扰动变换列表, 变换操作的次数
    """
    if scale_ratio == 1:
        return None, None, 0
    scale_trans_type, trans_num = get_scale_type_nums(scale_ratio)
    disturbe_num = random.randint(int(trans_num * DISTURBESUM_MIN), int(trans_num * DISTURBESUM_MAX))
    disturbe_arr = get_disturbe_arr(trans_num, disturbe_num)

    # 生成缩小或放大变换
    trans = scale_trans_type * np.ones(trans_num, dtype=int, order='C').reshape(-1, 1)  # 第一列保证原始变换
    return trans, disturbe_arr, trans_num


def get_mixed_trans_array(trans_type_num, *trans_type_option_num):
    """
    :param trans_type_num: 总共有几种变化要混合，应当输入int
    暂定应该就四种吧，水平平移，垂直平移，旋转，缩放
    :param trans_type_option_num: 应当输入trans_type_num个trans_type和对应trans_type的trans_option_num
    即前半部分是类型，后半部分是类型对应的数目，都为int型
    :return: 基础变换列表, 变换操作的次数
    """
    # 判断混合的操作数是否符合要求
    if trans_type_num < 2:
        return None
    if not (len(trans_type_option_num) == 2 * trans_type_num):
        return None
    # 放大系数，用来之后的映射
    amplification_factor = np.zeros(trans_type_num - 1, dtype=float)
    # 操作数从大到小排列
    big2small_trans_option_num = np.zeros(trans_type_num, dtype=int)
    # 操作数从大到小排列对应的种类
    big2small_trans_type = np.zeros(trans_type_num, dtype=int)
    # 提取数据
    for i in range(0, trans_type_num):
        big2small_trans_type[i] = trans_type_option_num[i]
        big2small_trans_option_num[i] = trans_type_option_num[i + trans_type_num]
    # 比较然后将操作数从大到小排列，因为数据量较少就直接冒泡算了
    for i in range(0, trans_type_num):
        for j in range(i, trans_type_num):
            if big2small_trans_option_num[i] < big2small_trans_option_num[j]:
                # 将操作数从大到小排列
                tmp = big2small_trans_option_num[i]
                big2small_trans_option_num[i] = big2small_trans_option_num[j]
                big2small_trans_option_num[j] = tmp
                # 得到操作数从大到小排列对应的种类
                tmp = big2small_trans_type[i]
                big2small_trans_type[i] = big2small_trans_type[j]
                big2small_trans_type[j] = tmp
    # 得到放大系数
    for i in range(0, trans_type_num - 1):
        amplification_factor[i] = big2small_trans_option_num[0] / big2small_trans_option_num[i + 1]
    # 生成trans_type_num个数组，当作各变化的下标，下标从1开始
    trans_option_index = np.zeros([trans_type_num, big2small_trans_option_num[0]], dtype=float)
    for i in range(0, trans_type_num):
        trans_option_index[i] = np.arange(1, big2small_trans_option_num[0] + 1, 1)
        trans_option_index[i][trans_option_index[i] > big2small_trans_option_num[i]] = 0
    # 映射到最大的区间，为了后续排序
    for i in range(1, trans_type_num):
        trans_option_index[i] = trans_option_index[i] * amplification_factor[i - 1]
    # 此时各个trans_option_index都是有序排列的
    # 若将各个trans_option_index混合，同时按照混合的下标同时混合多种操作，即可得到均匀分布的混合操作
    trans_option_index_i = np.zeros(trans_type_num, dtype=int)  # 用来访问各个trans_option_index，即第二维
    trans_option_num = 0  # 总的操作次数
    for i in range(0, trans_type_num):
        trans_option_num += big2small_trans_option_num[i]
    trans = np.ones((trans_option_num, 1), dtype=int)  # 用来返回最终结果
    trans_i = 0  # 用来访问trans
    trans_i_limit = trans_option_num - trans_type_num  # 当最后的时候，下标相同的时候访问会越界
    while trans_i < trans_i_limit:
        # 争取两个下标相同（一个映射后）的时候先放多的操作，那么能尽量均匀
        # 挑出下标最小的
        tmp_min_type = 0  # 暂时记录下标最小的一类，即trans_option_index的第一维度
        for i in range(0, trans_type_num):
            if (trans_option_index[tmp_min_type][trans_option_index_i[tmp_min_type]] >
                    trans_option_index[i][trans_option_index_i[i]]):
                tmp_min_type = i
        trans_option_index_i[tmp_min_type] += 1
        trans[trans_i] = big2small_trans_type[tmp_min_type]
        trans_i += 1
    for i in range(0, trans_type_num):
        trans[trans_i_limit - i + trans_type_num - 1] = big2small_trans_type[trans_type_num - i - 1]  # 因为跳过了最后几次所以要补上
    return trans


def get_trans_array_withDisturbe_rotate_scale(angle, scale_ratio):
    """
    :param angle: 旋转角度
    :param scale_ratio: 缩放系数
    :return: 变换矩阵， 扰动矩阵，变换次数
    """
    if angle == 0 or scale_ratio == 1:
        return None, None, 0
    # 旋转类型、次数
    # 缩放类型、次数
    rotate_trans_type, rotate_trans_num = get_rotate_type_nums(angle)
    scale_trans_type, scale_trans_num = get_scale_type_nums(scale_ratio)

    # 总的变换次数
    trans_num = rotate_trans_num + scale_trans_num

    # 扰动次数、扰动变换矩阵
    disturbe_num = random.randint(int(trans_num * DISTURBESUM_MIN), int(trans_num * DISTURBESUM_MAX))
    disturbe_arr = get_disturbe_arr(trans_num, disturbe_num)

    # 均匀融合变换
    trans = get_mixed_trans_array(2, rotate_trans_type, scale_trans_type, rotate_trans_num, scale_trans_num)
    trans = trans[:disturbe_arr.shape[0]]
    trans = trans.reshape(-1, 1)  # 第一列保证原始变换
    return trans, disturbe_arr, trans_num


def get_trans_array_withDisturbe_rotate_translation(angle, dx, dy):
    """
    :param angle: 旋转角度
    :param dx: 水平位移
    :param dy: 垂直位移
    :return: 变换矩阵， 扰动矩阵，变换次数
    """
    # 获得平移类型、次数
    (translation_trans_type_x, trans_num_x), (translation_trans_type_y, trans_num_y) = get_translation_type_nums(dx, dy)
    if angle == 0 or (translation_trans_type_x is None and translation_trans_type_y is None):
        return None, None, 0

    # 旋转类型、次数
    rotate_trans_type, rotate_trans_num = get_rotate_type_nums(angle)

    if translation_trans_type_y is not None and translation_trans_type_x is not None:
        # 总的变换次数
        trans_num = rotate_trans_num + trans_num_x + trans_num_y
        # 混合变换
        trans = get_mixed_trans_array(3, rotate_trans_type, translation_trans_type_x, translation_trans_type_y,
                                      rotate_trans_num, trans_num_x, trans_num_y)
    elif translation_trans_type_x is not None:
        # 总的变换次数
        trans_num = rotate_trans_num + trans_num_x
        # 混合变换
        trans = get_mixed_trans_array(2, rotate_trans_type, translation_trans_type_x, rotate_trans_num, trans_num_x)
    else:
        # 总的变换次数
        trans_num = rotate_trans_num + trans_num_y
        # 混合变换
        trans = get_mixed_trans_array(2, rotate_trans_type, translation_trans_type_y, rotate_trans_num, trans_num_y)
    # 扰动次数、扰动变换矩阵
    disturbe_num = random.randint(int(trans_num * DISTURBESUM_MIN), int(trans_num * DISTURBESUM_MAX))
    disturbe_arr = get_disturbe_arr(trans_num, disturbe_num)

    trans = trans[:disturbe_arr.shape[0]]
    trans = trans.reshape(-1, 1)  # 第一列保证原始变换

    return trans, disturbe_arr, trans_num


def get_trans_array_withDisturbe_translation_scale(dx, dy, scale_ratio):
    """
    :param dx: 水平平移
    :param dy: 垂直平移
    :param scale_ratio: 缩放系数
    :return: 变换矩阵， 扰动矩阵，变换次数
    """
    # 获得平移类型、次数
    (translation_trans_type_x, trans_num_x), (translation_trans_type_y, trans_num_y) = get_translation_type_nums(dx, dy)
    if scale_ratio == 1 or (translation_trans_type_x is None and translation_trans_type_y is None):
        return None, None, 0
    # 缩放类型、次数
    scale_trans_type, scale_trans_num = get_scale_type_nums(scale_ratio)

    if translation_trans_type_y is not None and translation_trans_type_x is not None:
        # 总的变换次数
        trans_num = scale_trans_num + trans_num_x + trans_num_y
        # 混合变换
        trans = get_mixed_trans_array(3, scale_trans_type, translation_trans_type_x, translation_trans_type_y,
                                      scale_trans_num, trans_num_x, trans_num_y)
    elif translation_trans_type_x is not None:
        # 总的变换次数
        trans_num = scale_trans_num + trans_num_x
        # 混合变换
        trans = get_mixed_trans_array(2, scale_trans_type, translation_trans_type_x, scale_trans_num, trans_num_x)
    else:
        # 总的变换次数
        trans_num = scale_trans_num + trans_num_y
        # 混合变换
        trans = get_mixed_trans_array(2, scale_trans_type, translation_trans_type_y, scale_trans_num, trans_num_y)
    # 扰动次数、扰动变换矩阵
    disturbe_num = random.randint(int(trans_num * DISTURBESUM_MIN), int(trans_num * DISTURBESUM_MAX))
    disturbe_arr = get_disturbe_arr(trans_num, disturbe_num)

    trans = trans[:disturbe_arr.shape[0]]
    trans = trans.reshape(-1, 1)  # 第一列保证原始变换

    return trans, disturbe_arr, trans_num


def get_trans_array_withDisturbe_rotate_translation_scale(angle, dx, dy, scale_ratio):
    """
    :param angle: 旋转角度
    :param dx: 水平平移
    :param dy: 垂直平移
    :param scale_ratio: 缩放系数
    :return: 变换矩阵， 扰动矩阵，变换次数
    """
    # 获得平移类型、次数
    (translation_trans_type_x, trans_num_x), (translation_trans_type_y, trans_num_y) = get_translation_type_nums(dx, dy)
    if scale_ratio == 1 or (translation_trans_type_x is None and translation_trans_type_y is None):
        return None, None, 0
    # 旋转类型、次数
    rotate_trans_type, rotate_trans_num = get_rotate_type_nums(angle)
    # 缩放类型、次数
    scale_trans_type, scale_trans_num = get_scale_type_nums(scale_ratio)

    if translation_trans_type_y is not None and translation_trans_type_x is not None:
        # 总的变换次数
        trans_num = scale_trans_num + trans_num_x + trans_num_y + rotate_trans_num
        # 混合变换
        trans = get_mixed_trans_array(4, scale_trans_type, translation_trans_type_x, translation_trans_type_y,
                                      rotate_trans_type, scale_trans_num, trans_num_x, trans_num_y, rotate_trans_num)
    elif translation_trans_type_x is not None:
        # 总的变换次数
        trans_num = scale_trans_num + trans_num_x + rotate_trans_num
        # 混合变换
        trans = get_mixed_trans_array(3, scale_trans_type, translation_trans_type_x, rotate_trans_type,
                                      scale_trans_num, trans_num_x, rotate_trans_num)
    else:
        # 总的变换次数
        trans_num = scale_trans_num + trans_num_y + rotate_trans_num
        # 混合变换
        trans = get_mixed_trans_array(3, scale_trans_type, translation_trans_type_y, rotate_trans_type,
                                      scale_trans_num, trans_num_y, rotate_trans_num)
    # 扰动次数、扰动变换矩阵
    disturbe_num = random.randint(int(trans_num * DISTURBESUM_MIN), int(trans_num * DISTURBESUM_MAX))
    disturbe_arr = get_disturbe_arr(trans_num, disturbe_num)

    trans = trans[:disturbe_arr.shape[0]]
    trans = trans.reshape(-1, 1)  # 第一列保证原始变换

    return trans, disturbe_arr, trans_num


def get_trans_array_withDisturbe(trans_type='', angle=0, dx=0, dy=0, scale_ratio=0):
    """
    :param trans_type: 操作名字，Rotate旋转，Translation平移，Scale缩放，Rotate_Scale旋转+缩放，
    Rotate_Translation旋转+平移，Translation_Scale平移+缩放，Rotate_Translation_Scale旋转+平移+缩放
    :param angle: 旋转角度
    :param dx: 水平平移距离
    :param dy: 垂直平移距离
    :param scale_ratio: 缩放比例
    :param ...更多变换有待开发
    :return: 一个包含操作序列的list
    """
    # 操作序列的数组，操作次数，扰动序列
    global trans, trans_num, disturbe_arr
    # 判断是哪个变换
    # 旋转
    if trans_type == 'Rotate':
        trans, disturbe_arr, trans_num = get_trans_array_withDisturbe_rotate(angle)
    # 平移
    elif trans_type == 'Translation':
        trans, disturbe_arr, trans_num = get_trans_array_withDisturbe_translation(dx, dy)
    # 缩放
    elif trans_type == 'Scale':
        trans, disturbe_arr, trans_num = get_trans_array_withDisturbe_scale(scale_ratio)
    # 旋转加缩放
    elif trans_type == 'Rotate_Scale':
        trans, disturbe_arr, trans_num = get_trans_array_withDisturbe_rotate_scale(angle, scale_ratio)
    # 旋转加平移
    elif trans_type == 'Rotate_Translation':
        trans, disturbe_arr, trans_num = get_trans_array_withDisturbe_rotate_translation(angle, dx, dy)
    # 平移加缩放
    elif trans_type == 'Translation_Scale':
        trans, disturbe_arr, trans_num = get_trans_array_withDisturbe_translation_scale(dx, dy, scale_ratio)
    # 旋转平移加缩放
    elif trans_type == 'Rotate_Translation_Scale':
        trans, disturbe_arr, trans_num = get_trans_array_withDisturbe_rotate_translation_scale(angle, dx, dy, scale_ratio)
    else:
        return None, 0
    # 生成对应的操作+扰动变换
    trans = np.concatenate((trans, disturbe_arr), axis=1)  # 将扰动加到第二列
    # 返回
    return trans.tolist(), trans_num


def trans_merge(t1, t2):
    """
    为了加速运算
    :param t1: 变换1矩阵
    :param t2: 变换2矩阵
    :return: 融合变换矩阵
    """
    # opencv使用齐次坐标变换，保证变换矩阵能混合，除第一个矩阵，其余矩阵加一行[0, 0, 1]
    temp = np.concatenate((t2, np.array([[0, 0, 1]])), axis=0)
    return np.dot(t1, temp)


def add_trans_center_disturbe(trans_array, height, width):
    """
    :param trans_array: 变换操作的数组
    :param height: 图像高度
    :param width: 图像宽度
    :return: 列表，包含8个基础变换，0~1为逆、顺时针旋转，2~5为上、下、左、右平移，6~7为缩小、放大
    """
    center_distube_x = random.randint(-int(width * ROTATE_CENTER_DISTURBE_X), int(width * ROTATE_CENTER_DISTURBE_X))
    center_distube_y = random.randint(-int(height * ROTATE_CENTER_DISTURBE_Y), int(height * ROTATE_CENTER_DISTURBE_Y))
    trans_array[0] = cv2.getRotationMatrix2D((height / 2 + center_distube_y, width / 2 + center_distube_x), ANGLE_UNIT, 1)
    trans_array[1] = cv2.getRotationMatrix2D((height / 2 + center_distube_y, width / 2 + center_distube_x), -ANGLE_UNIT, 1)
    return trans_array


def get_trans_img_withDisturbe(original_img, trans_disturbe, blur_type='', kneral_size=(5, 5), light_change_open=False):
    """
    :param original_img: 原始图像
    :param trans_disturbe: 变换下标列表，每个值对应8个基础变换
    :param blur_type: 滤波类型
    :param kneral_size: 滤波尺寸
    :param light_change_open: 是否开启亮度变化
    :return:
    """
    T = init_trans(original_img)  # 初始化8个基础变换
    rows, cols, channels = original_img.shape
    res = [original_img]
    pre_img = original_img
    # 遍历列表，每次变换
    for trans in trans_disturbe:
        flag = True    # 用来测试哪种方案好
        # 是否进行中心扰动
        if trans[0] in [0, 1] or trans[1] in [0, 1]:
            if ROTATE_CENTER_DISTURBE:
                T = add_trans_center_disturbe(T, rows, cols)
        t = T[trans[0]]
        # 判断是否有扰动，判断是不是停顿，停顿的话应该连压多张图片进结果，准备帧间滤波
        if trans[1] != -1 and trans[1] != 8:
            # 有扰动，进行变换矩阵融合
            t = trans_merge(t, T[trans[1]])
        # 进行图像变换
        cur_img = cv2.warpAffine(pre_img, t, (cols, rows))
        # 加入平滑（模糊）
        if blur_type == 'Gauss':
            cur_img = cv2.GaussianBlur(cur_img, kneral_size, 0)
        if light_change_open:
            cur_img = light_change_with_time(cur_img)
        pre_img = cur_img
        # 如果是停顿的话再压两张进结果
        if trans[1] == 8:
            res.append(cur_img)
            res.append(cur_img)
        res.append(cur_img)
    return res


def img_merge(list_img, merge_method=0, type='Mean'):
    """

    :param list_img: 图像列表
    :param merge_method: 图像融合方法，0-默认亮度优先，1-加权平均
    :param type:加权类型，Mean为相同占比，Gauss为高斯加权
    :return: 融合后的图像
    """
    if merge_method == 0:
        res = list_img[0]
        for n in range(1, len(list_img) - 1):
            if SHOW_TEMP_IMG:
                cv2.imwrite('./temp/move' + str(n) + '.jpg', list_img[n])
            res = np.maximum(res, list_img[n])
    elif merge_method == 1:
        if type == 'Mean':
            beta = 1 / len(list_img)
            res = list_img[0] * beta
            for n in range(1, len(list_img) - 1):
                if SHOW_TEMP_IMG:
                    cv2.imwrite('./temp/move' + str(n) + '.jpg', list_img[n])
                res = cv2.addWeighted(res, 1.0, np.array(list_img[n], dtype=np.float), beta, 0)

            # res = np.array(list_img[0], dtype=np.int)
            # for n in range(1, len(list_img) - 1):
            #     if SHOW_TEMP_IMG:
            #         cv2.imwrite('./temp/move' + str(n) + '.jpg', list_img[n])
            #     res += list_img[n]
            # res = res / len(list_img)
            # res = np.array(res, dtype=np.uint8)
        elif type == 'Gauss':
            gauss_beta = cv2.getGaussianKernel(len(list_img), 0)
            res = list_img[0] * gauss_beta[0]
            for n in range(1, len(list_img) - 1):
                if SHOW_TEMP_IMG:
                    cv2.imwrite('./temp/move' + str(n) + '.jpg', list_img[n])
                res = cv2.addWeighted(res, 1.0, np.array(list_img[n], dtype=np.float), float(gauss_beta[n]), 0)

    return np.array(res, dtype=np.uint8)


light_change_times = 0.0


def light_change_with_time(original_img):
    """
        :param original_img: 原始图像
        :return: 随时间变换亮度的图像
        """
    # 生成一个数用来进行亮度变化
    global light_change
    if 0:
        light_change = time.time()
    else:
        global light_change_times
        light_change = light_change_times
        light_change_times += math.pi/10
    # 求三角函数
    light_change = math.sin(light_change)
    # 约束到0~2
    light_change += 1
    # 约束到
    # 如果采用系统时间的话超过1基本就不变了
    light_change = light_change * 0.1 + 0.9
    # 进行亮度变换
    # img = original_img.flatten().reshape(original_img.shape)
    # 不知道为什么图片像素突破255直接置为1，只能这么写了
    tmp_img = np.minimum(255, light_change * original_img)
    return tmp_img
    # # 生成一个对角阵
    # light_change = np.eye(original_img.shape[1], dtype=float)
    # # 对对角线上元素赋值，当前系统时间
    # row, col = np.diag_indices_from(light_change)
    # if 1:
    #     light_change[row, col] = time.time()
    # else:
    #     global light_change_times
    #     light_change[row, col] = light_change_times
    #     light_change_times += math.pi/10
    # # 对矩阵求三角函数
    # light_change = np.sin(light_change)
    # # 约束到0~2
    # light_change[row, col] += 1
    # # 约束到
    # # 如果采用系统时间的话超过1基本就不变了
    # light_change[row, col] = light_change[row, col] * 0.1 + 0.85
    # # print('light_change: %s' % (light_change))
    # # 进行亮度变换
    # img = original_img.flatten().reshape(original_img.shape)
    # # 不知道为什么图片像素突破255直接置为1，只能这么写了
    # tmp_img = np.zeros([original_img.shape[0], original_img.shape[1]], dtype=float)
    # tmp_img = np.dot(original_img[:, :, 0], light_change)
    # # 进行亮度约束
    # tmp_img[tmp_img > 255] = 255
    # img[:, :, 0] = tmp_img
    # tmp_img = np.dot(original_img[:, :, 1], light_change)
    # # 进行亮度约束
    # tmp_img[tmp_img > 255] = 255
    # img[:, :, 1] = tmp_img
    # tmp_img = np.dot(original_img[:, :, 2], light_change)
    # # 进行亮度约束
    # tmp_img[tmp_img > 255] = 255
    # img[:, :, 2] = tmp_img
    # return img


color_change_times = 0.0


def color_change_with_time(original_img):
    """
        :param original_img: 原始图像
        :return: 随时间变换亮度的图像
        """
    # 生成一个数用来进行亮度变化
    global color_change
    if 0:
        color_change = time.time()
    else:
        global color_change_times
        color_change = color_change_times
        color_change_times += math.pi/10
    img = original_img.flatten().reshape(original_img.shape)
    phase = [math.pi/6, math.pi/4, math.pi/2]
    for i in range(3):
        # 求三角函数
        color_tri_change = math.sin(color_change + phase[i])
        # 约束到0~2
        color_tri_change += 1
        # 约束到
        # 如果采用系统时间的话超过1基本就不变了
        color_tri_change = color_tri_change * 0.1 + 0.9
        # 进行亮度变换
        img[:, :, i] = np.minimum(255, color_tri_change * img[:, :, i])
    return img
    # # 生成一个对角阵
    # color_change = np.eye(original_img.shape[1], dtype=np.float)
    # # 对对角线上元素赋值，当前系统时间
    # row, col = np.diag_indices_from(color_change)
    # if 0:
    #     color_change[row, col] = time.time()
    # else:
    #     global color_change_times
    #     color_change[row, col] = color_change_times
    #     color_change_times += 0.1
    # img = original_img.flatten().reshape(original_img.shape)
    # phase = [math.pi/6, math.pi/4, math.pi/2]
    # for i in range(3):
    #     color_tri_change = color_change.flatten().reshape(color_change.shape)
    #     # 添加相位
    #     color_tri_change[row, col] += phase[i]
    #     # 对矩阵求三角函数
    #     color_tri_change = np.sin(color_tri_change)
    #     # 约束到0~2
    #     color_tri_change[row, col] += 1
    #     # 如果采用系统时间的话超过1基本就不变了
    #     color_tri_change[row, col] = color_tri_change[row, col] * 0.1 + 0.9
    #     # print('light_change: %s' % (light_change))
    #     # 进行亮度变换
    #     img[:, :, i] = np.minimum(255, np.dot(img[:, :, i], color_tri_change))
    # return img


def img_pretreatment(original_img, method=0):
    """
    :param original_img: 原始图像
    :param method: 预处理阈值化方法，0默认亮度最大值*THRESHOLD_RATE作为阈值，全局阈值化，1 Otsu's二值化
    :return: 分离后的灯光图像，背景图像, 原始占空比， 分离阈值
    """
    # 遍历图像，进行亮度增强
    # img = original_img.flatten(order='A')
    # strlength = np.where(img >= threshold)
    # weaken = np.where(img < threshold)
    # img[strlength] = np.minimum(255, (1 + STRENENGTHEN) * img[strlength])
    # img[weaken] = (1 - STRENENGTHEN) * img[weaken]

    # 图像转成灰度图像
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    if method == 0:
        # 方法一
        # 阈值设为最大值*THRESHOLD_RATE
        light_max = np.max(original_img)
        threshold = light_max * THRESHOLD_RATE

        # # 方法二
        # dis_maxmin = np.ptp(original_img)
        # threshold = dis_maxmin * THRESHOLD_RATE

        # 全局阈值二值化
        ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        # cv2.imwrite('binary.jpg', binary)
    elif method == 1:
        # Otsu's二值化
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        return None, None
    background = np.where(binary != 255)  # 认为非灯光的像素下标
    light = np.where(binary == 255)  # 认为灯光的像素下标

    # 分别得到灯光图像和剩余的背景图像
    light_img = np.zeros(original_img.shape, dtype=np.uint8)
    light_img[light[0], light[1], :] = original_img[light[0], light[1], :]
    background_img = np.zeros(original_img.shape, dtype=np.uint8)
    background_img[background[0], background[1], :] = original_img[background[0], background[1], :]
    cv2.imwrite("binary.jpg", binary)
    cv2.imwrite("light.jpg", light_img)
    cv2.imwrite("backgroud.jpg", background_img)

    original_DutyCycle = 3 * len(light[0]) / original_img.size
    print("原始占空比：" + str(original_DutyCycle))

    return light_img, background_img, original_DutyCycle, ret


def split_light_background_trans(light_img, background_img):
    """
    :param light_img: 灯光图像
    :param background_img: 背景图像
    :return:
    """
    # 灯光图像进行变换
    light_trans, light_trans_sum = get_trans_array_withDisturbe(trans_type='Rotate', angle=180
                                                                )
    light_result = get_trans_img_withDisturbe(light_img, light_trans, light_change_open=False)
    # 图像叠加-灯光选择亮度优先
    light_res = img_merge(light_result, 0)
    cv2.imwrite("ouput/light_res.jpg", light_res)

    # 背景图像进行变换
    background_trans, background_trans_sum = get_trans_array_withDisturbe(trans_type='Rotate', angle=180)
    background_result = get_trans_img_withDisturbe(background_img, background_trans)
    # 图像叠加-背景选择加权平均
    background_res = img_merge(background_result, 1, type='Gauss')
    cv2.imwrite("output/backgroud_res.jpg", background_res)

    merge_img = cv2.addWeighted(light_res, 1.0, background_res, 1.0, 0)
    merge_img[merge_img > 255] = 255
    cv2.imwrite('output/merge.jpg', merge_img)

    return merge_img


def getDutyCycle(inputImg, threshold):
    """
    :param inputImg: 输入图像
    :param threshold: 分离阈值
    :return: 图像光源占空比
    """
    # 转灰度图
    img = cv2.cvtColor(inputImg, cv2.COLOR_BGR2GRAY)
    # 返回占空比
    return 3 * len(img[img > threshold]) / inputImg.size


if __name__ == "__main__":
    img_path = 'input/0.jpg'
    # 是否进行人像抠图
    KOUTU = False
    global img
    global person, mask
    if KOUTU:
        mask = Koutu.getKoutuMask(img_path)
        person, img = Koutu.getPersonAndBackground(img_path)
        # cv2.imwrite('mask.jpg', mask)
        # cv2.imwrite('person.jpg', person)
        # cv2.imwrite('img.jpg', img)
    else:
        img = cv2.imread(img_path)
    mycluster = Julei.myCluster(img, 200, 4)
    cluster_num, each_cluster_num = mycluster.get_cluster_num()
    centers = mycluster.get_cluster_center()
    fake_light_img = mycluster.get_fake_light_point(img, 1000, 5)
    cv2.imwrite('output/fake_light1000.jpg', fake_light_img)

    # 阈值化得到灯光、剩余背景
    light_img, background_img, original_DutyCycle, threshold = img_pretreatment(img, method=1)
    merge_img = split_light_background_trans(light_img, background_img)
    new_DutyCycle = getDutyCycle(merge_img, threshold)
    print("融合图像占空比：" + str(new_DutyCycle))
    # # 光线变换测试
    # for i in range(1, 100):
    #     light_img = light_change_with_time(light_img)
    #     cv2.imwrite("./output1/img" + str(i) + ".jpg", light_img)
    # rows, cols, channels = img.shape
    # # 几何变换
    # # trans, trans_sum = get_trans_array_withDisturbe(trans_type='Rotate', angle=30)
    # # trans, trans_sum = get_trans_array_withDisturbe(trans_type='Translation', dx=50, dy=0)
    # # trans, trans_sum = get_trans_array_withDisturbe(trans_type='Scale', scale_ratio=0.5)
    # # trans, trans_sum = get_trans_array_withDisturbe(trans_type='Rotate_Scale', scale_ratio=0.5, angle=30)
    # # trans, trans_sum = get_trans_array_withDisturbe(trans_type='Rotate_Translation', angle=30, dx=10, dy=10)
    # # result = get_trans_img_withDisturbe(img, trans, 'Gauss', (11, 11))
    # result = get_trans_img_withDisturbe(img, trans)
    #
    # # 图像叠加-默认亮度优先，可选加权平均
    # res = img_merge(result)
    #
    # if KOUTU:
    #     res = Koutu.fixPersonAndBackground(person, res, mask)
    #     # cv2.imwrite('res1.jpg', res)
    # cv2.imwrite('result.jpg', res)
    # cv2.waitKey()
