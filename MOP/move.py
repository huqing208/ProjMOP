import numpy as np
import cv2
import random
import math

def get_rotate_scale(center, step_angle, step_scale=1):
    """
    :param center: 旋转中心
    :param step_angle: 旋转角度
    :param step_scale: 缩放因子
    :return: 旋转+缩放变换矩阵
    """
    r = cv2.getRotationMatrix2D(center, step_angle, step_scale)
    return r


def get_translation(step_x, step_y):
    """
    :param step_x: 水平移动距离
    :param step_y: 垂直移动距离
    :return: 平移变换矩阵
    """
    t = [[1, 0, step_x], [0, 1, step_y]]
    t = np.array(t, dtype=np.float32)
    return t


def get_trans_img(frames, original_img, t, blur_type='', kneral_size=(5, 5)):
    """

    :param frames: 结果图像帧数
    :param original_img: 原始图像
    :param t: 几何变换
    :param blur_type: 滤波类型
    :param kneral_size: 滤波尺寸
    :return:
    """
    rows, cols, channels = original_img.shape
    res = [original_img]
    pre_img = original_img
    for i in range(frames):
        cur_img = cv2.warpAffine(pre_img, t, (cols, rows))
        if blur_type == 'Gauss':
            cur_img = cv2.GaussianBlur(cur_img, kneral_size, 0)
        pre_img = cur_img
        res.append(cur_img)
    return res

def trans_merge(t1, t2):
    """
    为了加速运算
    :param t1: 变换1矩阵
    :param t2: 变换2矩阵
    :return: 融合变换矩阵
    """
    transform = []
    for i in range(len(t1)):
        # opencv使用齐次坐标变换，保证变换矩阵能混合，除第一个矩阵，其余矩阵加一行[0, 0, 1]
        temp = np.concatenate((t2[i], np.array([[0, 0, 1]])), axis=0)
        transform.append(np.dot(t1[i], temp))
    return transform


def img_merge(list_img):
    """

    :param list_img: 图像列表
    :return: 融合后的图像-亮度优先
    """
    res = list_img[0]
    for n in range(1, len(list_img) - 1):
        res = np.maximum(res, list_img[n])

    return res


if __name__ == "__main__":
    img = cv2.imread('2.jpg')
    rows, cols, channels = img.shape
    # 几何变换
    rotate = get_rotate_scale((cols / 2, rows / 2), -3, 0.9)  # 旋转+缩小
    # translation = get_translation(10, 0)  # 平移
    # scale = get_rotate_scale(10, (cols/2, rows/2), 0, 0.9)  # 使用旋转的缩放因子，但旋转角度为0
    #trans = trans_merge(scale, translation)  # 变换融合
    result = get_trans_img(16, img, rotate, 'Gauss', (21, 21))  # 变换后的帧图像

    # 图像叠加-亮度优先
    res = img_merge(result)

    cv2.imwrite('result.jpg', res)
    cv2.waitKey()
