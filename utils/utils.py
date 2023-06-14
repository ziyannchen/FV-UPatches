import numpy as np
import os
import cv2
import yaml
import os
from natsort import os_sorted
import math
import numpy as np
import shutil


def check_name(img_name):
    return img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))


def getConfig(yamlPath):
    with open(yamlPath, 'r', encoding='utf-8') as f:
        cfg = f.read()
    d = yaml.full_load(cfg)
    return d


def getAllDirs(path):
    '''
    给出文件夹名，列出该文件夹下所有的文件夹名称，返回名称列表
    '''
    return list(filter(lambda f: str(f).find('.') == -1, os.listdir(path)))


def getFiles(path, postfix, sort='nat'):
    '''
    # path: 文件所在文件夹
    # postfix: 所需文件后缀
    # sort: 文件排序方式 nat为按照文件名中包含的自然数排序
    '''
    all_files = os.listdir(path)
    # all_files = [os.path.join(path, i) for i in all_files]
    res = list(filter(lambda f: str(f).endswith(postfix), all_files))

    if sort == 'nat':
        return os_sorted(res)
    return res


def rmfiles(path):
    '''
    给出文件夹名称，删除这个文件夹内所有的文件
    '''
    files = os.listdir(path)
    for i in files:
        p = os.path.join(path, i)
        try:
            os.remove(p)
        except:
            shutil.rmtree(p)
    return 1

def judge_prefix(query, refer, prefix_num, split='_'):
    '''
    构造匹配对时用文件名前缀判断是否属于同一类
    '''
    query = query.split(split)[:prefix_num]
    refer = refer.split(split)[:prefix_num]
    return query == refer


def np2kp(kp, size=5):
    '''
    根据细化图得到关键点，骨架位置为关键点位置
    骨架图中nb邻域内点的个数+一个常数为关键点的大小
    角度为offline计算的方向（~骨架走向的垂直方向）
    返回人为构造的cv2 keypoint类
    '''
    kpp = []
    angle = -1
    for i in kp:
        x = i[1]; y = i[0]
        kpp.append(cv2.KeyPoint(float(x), float(y), size, angle))
        
    return kpp



def my_loss(pt1, pt2):
    """
    pt2为右边的点集，不会出现k为无穷的情况
    """
    cot = []
    bad = []
    good = []
    tmp = {}
    for step, p1 in enumerate(pt1):
        p2 = pt2[step]
        p1_x = p1[0]
        p1_y = p1[1]
        p2_x = p2[0]
        p2_y = p2[1]

        k = (p2_y - p1_y) / (p2_x - p1_x)

        dis_s = math.sqrt((p2_y - p1_y) ** 2 + (p2_x - p1_x) ** 2)
        cot.append((k, dis_s))
    res = 0
    #     print(cot)
    for s1, i in enumerate(cot):
        
        num = 0
        for s2, j in enumerate(cot):
            if s1 == s2:
                continue
                
            k_d = abs(math.degrees(math.atan(i[0])) - math.degrees(math.atan(j[0])))
 
            #k_d = abs(i[0]-j[0])
            dis_d = abs(i[1] - j[1])
            if abs(k_d) < 6 and abs(dis_d) < 30:
                num += 1

        if num >= 0.5 * (len(cot)-1):
            res += 1
            good.append(s1)
        else:
            bad.append(s1)

    return res, good, bad

def judge_class_1st(session, name, prefix_num):
    if 'dual' in session:
        return name.split('_')[prefix_num] == '1' and name.split('_')[prefix_num+1] == 's1'
    return name.split('_')[prefix_num] == '1'

def judge_not_FVC(session, prefix_num, name1, name2, step=None):
    if not judge_class_1st(session, name1, prefix_num) or not judge_class_1st(session, name2, prefix_num):
        return True
#     if 'dual' in session:
#         judge1 = name1.split('_')[prefix_num+1]
#         judge2 = name2.split('_')[prefix_num+1]
#         if session == 'dual' and (judge1 != 's1' or judge2 != 's1'):
#             return True
#         if session == 'semi-dual' and (step <= 105 and (judge1 != 's1' or judge2 != 's2')):
#             return True
    return False
