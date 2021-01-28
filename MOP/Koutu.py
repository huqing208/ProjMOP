# 用来实现抠出人物和背景，以及将抠出的人物贴回背景
import paddlehub as hub
import numpy as np
import cv2

def getKoutuMask(srcImgAddr):
    """
    :param srcImgAddr: 图片路径，飞桨给的函数不知道怎么传数组进去，不能直接传图片
    :return: 掩模图（黑白图）,是单通道的
    """
    # 加载模型
    module = hub.Module(name="deeplabv3p_xception65_humanseg")
    # 进行预测
    results = module.segmentation(data={"image": [srcImgAddr]})
    # 掩膜不是纯粹的二值化图像，影响抠图
    results[0].get('data')[results[0].get('data') > 127] = 255
    results[0].get('data')[results[0].get('data') < 127] = 0
    # 返回结果，results是一个列表，里面的每个元素都是字典类型
    return results[0].get('data')


def getPersonAndBackground(srcImgAddr):
    """
    :param srcImgAddr: 图片路径，飞桨给的函数不知道怎么传数组进去，不能直接传图片
    :return: 返回抠出的人物以及没有人物的单纯背景
    """
    # 先获得抠图的掩膜（人物目标部分为白）
    mask = getKoutuMask(srcImgAddr)
    # 掩模图做反相（人物目标部分为黑）
    tmp = np.full(mask.shape, 255.0)
    mask_opp = tmp - mask
    # cv2.imwrite('mask_opp.jpg', mask_opp)
    # 取出人物，将原图和掩膜进行取最小值即可
    person = cv2.imread(srcImgAddr)
    # 注意是引用还是备份，注释的不对
    # background = person
    background = person.copy()
    person[:, :, 0] = np.minimum(person[:, :, 0], mask)
    person[:, :, 1] = np.minimum(person[:, :, 1], mask)
    person[:, :, 2] = np.minimum(person[:, :, 2], mask)
    # 取出背景，将原图和掩膜的反相进行取最小值即可，为什么不用掩膜取最大值，考虑到人物扣去应该变成全黑，防止太亮影响光影
    background[:, :, 0] = np.minimum(background[:, :, 0], mask_opp)
    background[:, :, 1] = np.minimum(background[:, :, 1], mask_opp)
    background[:, :, 2] = np.minimum(background[:, :, 2], mask_opp)
    # 返回结果
    return person, background


def fixPersonAndBackground(personImg, backgroundImg, mask):
    """
    :param personImg: 人物图片，为numpy的矩阵，三通道，除了人物部分应该全是黑的
    :param backgroundImg: 背景图片，为numpy的矩阵，三通道
    :param mask: 掩膜图片，为numpy的矩阵，单通道
    :return: 返回人物和背景的合成
    """
    # 先将背景图片（此刻应该进行了光影变换）和mask的反相取最小值，剔出人物位置
    # cv2.imwrite('backgroundImg.jpg', backgroundImg)
    tmp = np.full(mask.shape, 255.0)
    mask_opp = tmp - mask
    backgroundImg[:, :, 0] = np.minimum(backgroundImg[:, :, 0], mask_opp)
    backgroundImg[:, :, 1] = np.minimum(backgroundImg[:, :, 1], mask_opp)
    backgroundImg[:, :, 2] = np.minimum(backgroundImg[:, :, 2], mask_opp)
    # cv2.imwrite('backgroundImg.jpg', backgroundImg)
    # 将人物贴回去，即将人物和上述处理的图片进行取最大值
    backgroundImg = np.maximum(backgroundImg, personImg)
    # 返回结果
    return backgroundImg