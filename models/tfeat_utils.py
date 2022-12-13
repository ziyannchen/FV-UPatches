import cv2
import torch
import math
import numpy as np
import matplotlib.pyplot as plt

def describe_opencv(model, img, kpts, N, mag_factor, use_gpu=True):
    """
    Rectifies patches around openCV keypoints, and returns patches tensor
    """
    m, n = img.shape
    patches = []

    for kp in kpts:
        x, y = kp.pt
        s = kp.size
        a = kp.angle

        s = mag_factor * s / N
        cos = math.cos(a * math.pi / 180.0)
        sin = math.sin(a * math.pi / 180.0)

        M = np.matrix([
            [+s * cos, -s * sin, (-s * cos + s * sin) * N / 2.0 + x],
            [+s * sin, +s * cos, (-s * sin - s * cos) * N / 2.0 + y]])

        patch = cv2.warpAffine(img, M, (N, N),
                             flags=cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS)

        patches.append(patch)

    if len(patches) == 0:
        print('No patch to rectify!')
        return None

    patches = torch.from_numpy(np.asarray(patches)).float()
    patches = torch.unsqueeze(patches, 1)

    if use_gpu:
        patches = patches.cuda()
        res = model(patches).detach().cpu().numpy()
    else:
        res = model(patches).detach().numpy()

    return res#, patch

    # 如果GPU不足，使用下方代码，控制每次训练不大于300个patch
#         batch = 200
#         s = 0
#         e = batch
#         while s < len(patches):
#             if e > len(patches):
#                 d = model(patches[s:])
#             else:
#                 d = model(patches[s:e])
#             if s == 0:
#                 res = d.detach().cpu().numpy()
#             else:
#                 res = np.concatenate((res, d.detach().cpu().numpy()))
#             s += batch
#             e += batch

#         return res#, test
    
    

