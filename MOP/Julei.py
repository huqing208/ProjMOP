#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import cv2

"""
层次聚类
"""


class myCluster:

    # 欧氏距离,d(x,y)=math.sqrt(sum((x-y)*(x-y)))
    def __distance(self, x1, x2):
        """
        :param x1: 输入样本x1
        :param x2: 输入样本x2
        :return: 返回两个样本之间的欧氏距离
        """
        # 用来存放中间结果，最后开方
        sumSquares = 0
        # self.cols就是特征维数，比如二维坐标就是2
        for k in range(0, self.__cols):
            sumSquares += (x1[k] - x2[k]) ** 2
        return math.sqrt(sumSquares)

    def __get_light_num(self, inputImg, threshold):
        """
        :param inputImg: 输入图像
        :param threshold: 分离阈值
        :return: 图像亮点个数
        """
        # 转灰度图
        img = cv2.cvtColor(inputImg, cv2.COLOR_BGR2GRAY)
        # 返回图像亮点个数
        return len(img[img > threshold])

    def __init__(self, original_img, threshold, combine_distance):
        """
        :param original_img: 输入应当是原始图像
        :param original_img: 原始图像分割阈值，筛选光源
        :param combine_distance: 聚类阈值距离
        """
        # 图像缩放
        self.__scale_ratio = 1
        # 如果图像中亮点太多的话就缩放
        # 获取图像中亮点个数
        light_num = self.__get_light_num(original_img, threshold)
        if light_num > 1000:
            # 获取缩放倍数，如，长宽缩小一半，亮点数目缩小到1/4
            self.__scale_ratio = math.sqrt(1000 / light_num)
            # 缩放
            img = cv2.resize(original_img, None, fx=self.__scale_ratio, fy=self.__scale_ratio,
                             interpolation=cv2.INTER_LINEAR)
            # 原图像转成灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            # 原图像转成灰度图
            gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

        # 获取行列坐标
        RowAndCol = np.argwhere(gray > threshold)
        # 存储数据,里面是一大堆列表，每个列表是一个集合
        self.__data = []
        # 数据的数目，在这里就是行数了
        self.__rows = RowAndCol.shape[0]
        # 每个数据的特征数，在这里就是列数了
        self.__cols = RowAndCol.shape[1]
        # 两两聚类之间的距离，初始就是n*n矩阵
        # 为了之后计算方便，这里定义当两个点在同一类中，两者距离为1,000,000
        self.__neighbor_distance = np.full((self.__rows, self.__rows), 1000000, dtype=float)
        # 循环读入
        for i in range(0, self.__rows):
            self.__data.append([RowAndCol[i][0], RowAndCol[i][1]])
        # 计算距离矩阵
        for i in range(0, self.__rows):
            for j in range(0, self.__rows):
                # 如果是自己的话那就不计算了
                if i == j:
                    continue
                self.__neighbor_distance[i][j] = self.__distance(self.__data[i], self.__data[j])
        # 聚类
        self.__cluster(combine_distance)
        # 计算聚类数目以及聚类中心每个聚类的样本数目，此时应该是缩小后的样本数目
        self.__cal_cluster_num()
        # 计算聚类中心
        self.__cal_cluster_center()

    # 聚类
    def __cluster(self, combine_distance):
        """
        :param combine_distance: 给定的合并距离
        :return: 返回聚类结果,是[x1,y1,x2,y2,x3,y3,...]这种形式
        """
        # 记录要被删除的类（被合并的类）
        delete_cluster = []
        # 聚类
        # 每次取最小的两类之间的距离进行判断，如果该距离小于给定的合并距离，那么合并
        # 直到最小值都大于给定的合并距离，那么结束
        while True:
            # 先求得最小值所在的位置
            # 第mindistance_r行，第mindistance_c列，也就是mindistance_r类和mindistance_c之间的距离
            mindistance_r, mindistance_c = self.__find_min_idx()
            # 得到最小值
            mindistance = self.__neighbor_distance[mindistance_r][mindistance_c]
            # 判断是否要结束
            if mindistance > combine_distance:
                break
            # 如果不结束那么就是要聚类这最近的两类
            # 默认是靠后的类合并到靠前的类，看输入的时候的顺序
            if mindistance_r < mindistance_c:
                # self.data[mindistance_r] = (self.data[mindistance_r], self.data[mindistance_c])
                self.__data[mindistance_r].extend(self.__data[mindistance_c])
                # 后一类被记录，等会统一删除，如果现在删除了会导致下标发生变化影响之后的聚类
                delete_cluster.append(mindistance_c)
            else:
                # self.data[mindistance_c] = (self.data[mindistance_r], self.data[mindistance_c])
                self.__data[mindistance_c].extend(self.__data[mindistance_r])
                # 后一类被记录，等会统一删除，如果现在删除了会导致下标发生变化影响之后的聚类
                delete_cluster.append(mindistance_r)
            # 先保存一下，等会1000000就没了，要恢复
            row1 = np.where([self.__neighbor_distance[mindistance_r] == 1000000], 1000000, -1)
            row2 = np.where([self.__neighbor_distance[mindistance_c] == 1000000], 1000000, -1)
            # 其实row1==col1，row2==col2，所以注释掉了
            # col1 = np.where([self.neighbor_distance[:, mindistance_r] == 1000000], 1000000, -1)
            # col2 = np.where([self.neighbor_distance[:, mindistance_c] == 1000000], 1000000, -1)
            # 更新距离，用类间最短距离法
            self.__neighbor_distance[mindistance_r] = self.__neighbor_distance[mindistance_c] = \
                np.minimum(self.__neighbor_distance[mindistance_r], self.__neighbor_distance[mindistance_c])
            self.__neighbor_distance[:, mindistance_r] = self.__neighbor_distance[:, mindistance_c] = \
                np.minimum(self.__neighbor_distance[:, mindistance_r], self.__neighbor_distance[:, mindistance_c])
            # 因为更新距离用了minimum，所以1,000,000也被更新了
            # 因为两个聚类合并了，其实1000000也要更新，因为本来相隔两地，现在团聚了，团聚是1000000
            update = np.maximum(row1, row2)
            # 恢复1000000
            self.__neighbor_distance[mindistance_r] = self.__neighbor_distance[mindistance_c] = \
                np.maximum(self.__neighbor_distance[mindistance_r], update)
            self.__neighbor_distance[:, mindistance_r] = self.__neighbor_distance[:, mindistance_c] = \
                np.maximum(self.__neighbor_distance[:, mindistance_r], update)
        # 删掉多余的聚类
        # 里面有重复的先要处理一下，顺便排序
        delete_cluster = list(set(delete_cluster))
        # 要重后往前删，每删一次下标会重置一次
        delete_cluster.reverse()
        for item in delete_cluster:
            del self.__data[item]
        # 返回聚类结果
        return self.__data

    def __find_min_idx(self):
        # 找到最小值所在的位置，但是是一个数
        idx = self.__neighbor_distance.argmin()
        # 计算距离矩阵中的位置（二维，self.rows*self.rows的大小）
        return int(idx / self.__rows), int(idx % self.__rows)

    def get_cluster_num(self):
        """
        :return: 返回聚类总数和每个聚类的数目，前者是一个int，后者是一个np行向量
        由于之前图像缩放，所以每个聚类的样本数目会不准确
        """
        # 返回
        return self.__cluster_num, (self.__each_cluster_num / (self.__scale_ratio * self.__scale_ratio)).astype(
            np.int16)

    def __cal_cluster_num(self):
        # 开一个大小为聚类数目的数组存放每个聚类中样本的数目
        self.__each_cluster_num = np.zeros([len(self.__data)], dtype=int)
        # 统计数量
        for i in range(0, len(self.__data)):
            self.__each_cluster_num[i] = len(self.__data[i]) / 2
            # 聚类数量
            self.__cluster_num = len(self.__data)

    def get_cluster_center(self):
        """
        :return: 返回聚类中心坐标
        """
        return (self.__centers / self.__scale_ratio).astype(np.int16)

    def __cal_cluster_center(self):
        # 开一个大小为聚类数目的数组存放中心坐标
        self.__centers = np.zeros([self.__cluster_num, 2], dtype=int)
        # 每个聚类的中心
        # 先全部加起来，然后求平均
        for i in range(0, self.__cluster_num):
            for j in range(0, self.__each_cluster_num[i]):
                self.__centers[i][0] += self.__data[i][2 * j]
                self.__centers[i][1] += self.__data[i][2 * j + 1]
            self.__centers[i] = self.__centers[i] / self.__each_cluster_num[i]

    def __cal_points_in_cluster_max_distance(self, index):
        """
        :param index: 计算第index个聚类中点之间的最大距离
        :return: 返回最大距离
        """
        cluster_points_num = int(len(self.__data[index]) / 2)
        # 开一个二维距离数组用来存放两两点之间的距离，同一个点的距离为-1
        distance = np.full([cluster_points_num, cluster_points_num], -1.0, dtype=float)
        # 计算距离，因为距离矩阵对称性，所以计算一半即可
        # 提取行列坐标
        rows = self.__data[index][::2]
        cols = self.__data[index][1::2]
        for i in range(0, cluster_points_num):
            for j in range(i, cluster_points_num):
                distance[i][j] = self.__distance([rows[i], cols[i]], [rows[j], cols[j]])
        # 得到最大距离
        max_distance = np.max(distance)
        # # 因为之前缩放了，所以最大距离也要恢复一下
        # return max_distance / self.__scale_ratio
        return max_distance

    def __cal_points_in_list_min_distance(self, *samples_list):
        """
        :param samples_list: 应当是一组列表的列表，里面的第i列表是每个样本的第i特征
        暂时只支持两个特征的，懒得扩展了
        :return: 返回最小距离
        """
        list_samples_num = len(samples_list[0][0])
        # 开一个二维距离数组用来存放两两点之间的距离，同一个点的距离为1000000
        distance = np.full((list_samples_num, list_samples_num), 1000000, dtype=float)
        # 提取列表，转为每一个样本一个列表
        samples = [[] for i in range(0, list_samples_num)]
        for j in range(0, list_samples_num):
            for i in range(0, len(samples_list[0])):
                samples[j].append(samples_list[0][i][j])
        # 计算距离，因为距离矩阵对称性，所以计算一半即可
        for i in range(0, list_samples_num):
            for j in range(i + 1, list_samples_num):
                distance[i][j] = self.__distance(samples[i], samples[j])
        # 得到最小距离
        min_distance = np.min(distance)
        return min_distance

    def get_fake_light_point(self, original_img, split_limit, fake_light_size):
        """
        :param: original_img: 原始图像
        :param: split_limit: 一个聚类是split_limit的多少倍，就划分成多少个虚拟亮点，向上取整
        :param: fake_light_size: 生成光源的大小，int型，是正方形，只能是奇数
        :return: 返回虚拟亮点的图
        """
        # 对split_limit进行缩放处理
        split_limit = split_limit * self.__scale_ratio * self.__scale_ratio
        # 生成矩阵用来存放虚拟亮点图
        fake_light_img = original_img.flatten().flatten().reshape(original_img.shape)
        # 获取每个聚类的虚拟亮点个数
        each_cluster_fake_light_num = (self.__each_cluster_num / split_limit).astype(np.int16) + 1
        # 生成虚拟亮点掩膜用来提取虚拟亮点，虚拟亮点白，背景黑，等会和fake_light_img取两者最小即可
        fake_light_mask = np.zeros([fake_light_img.shape[0], fake_light_img.shape[1]], dtype=int)
        # 生成虚拟亮点
        for i in range(0, self.__cluster_num):
            # 如果只要一个虚拟亮点，那么直接给聚类中心
            if each_cluster_fake_light_num[i] == 1:
                fake_light_mask[int(self.__centers[i][0] / self.__scale_ratio)][
                    int(self.__centers[i][1] / self.__scale_ratio)] = 255
            else:
                # 计算该聚类中点的最大距离，用来限制虚拟亮点不要太近
                max_distance_in_cluster = self.__cal_points_in_cluster_max_distance(i)
                distance_limit = max_distance_in_cluster / (each_cluster_fake_light_num[i] + 1)
                # 从列表中随机取几个点，满足距离限制即可
                while True:
                    # 随机抽取下标，无重复
                    index = np.random.choice(int(len(self.__data[i]) / 2), each_cluster_fake_light_num[i],
                                             replace='False')
                    # 取出这些点
                    light_points = [[] for j in range(2)]
                    for index_i in index:
                        light_points[0].append(self.__data[i][2 * index_i])
                        light_points[1].append(self.__data[i][2 * index_i + 1])
                    # 判断这些点是否满足条件
                    min_distance_in_list = self.__cal_points_in_list_min_distance(light_points)
                    if min_distance_in_list < distance_limit:
                        for j in range(0, len(light_points)):
                            fake_light_mask[int(light_points[0][j] / self.__scale_ratio)][
                                int(light_points[1][j] / self.__scale_ratio)] = 255
                        break
        # 扩大亮点
        down_limit = -int(fake_light_size / 2)
        up_limit = int(fake_light_size / 2) + 1
        fake_light_mask_index = np.argwhere(fake_light_mask == 255)
        for index in fake_light_mask_index:
            for i in range(down_limit, up_limit):
                for j in range(down_limit, up_limit):
                    fake_light_mask[index[0] + i][index[1] + j] = 255
        # 获得光点图
        fake_light_img[:, :, 0] = np.minimum(fake_light_img[:, :, 0], fake_light_mask)
        fake_light_img[:, :, 1] = np.minimum(fake_light_img[:, :, 1], fake_light_mask)
        fake_light_img[:, :, 2] = np.minimum(fake_light_img[:, :, 2], fake_light_mask)
        # 选中的亮点不能是0，不然接下来不能变化
        tmp = np.ones(fake_light_img.shape)
        tmp[:, :, 0] = np.minimum(tmp[:, :, 0], fake_light_mask)
        tmp[:, :, 1] = np.minimum(tmp[:, :, 1], fake_light_mask)
        tmp[:, :, 2] = np.minimum(tmp[:, :, 2], fake_light_mask)
        fake_light_img = np.maximum(fake_light_img, tmp)
        # 修改亮度
        for index in fake_light_mask_index:
            lighter_ratio = min(255 / fake_light_img[index[0]][index[1]][0], 255 / fake_light_img[index[0]][index[1]][1]
                                , 255 / fake_light_img[index[0]][index[1]][2])
            if lighter_ratio == 1:
                continue
            for i in range(down_limit, up_limit):
                for j in range(down_limit, up_limit):
                    fake_light_img[index[0] + i][index[1] + j][0] = \
                        min(255, int(lighter_ratio * fake_light_img[index[0] + i][index[1] + j][0]))
                    fake_light_img[index[0] + i][index[1] + j][1] = \
                        min(255, int(lighter_ratio * fake_light_img[index[0] + i][index[1] + j][1]))
                    fake_light_img[index[0] + i][index[1] + j][2] = \
                        min(255, int(lighter_ratio * fake_light_img[index[0] + i][index[1] + j][2]))
        # 返回结果
        return fake_light_img
