import numpy as np
import imutils
import os

from utils.utils import np2kp, getConfig


class RidgeDetector:
    def __init__(self, c=4, ks=11):
        self.c = c
        self.ks = ks
        
        self.atype = 'RidgeDet'
        
    def kp_from_thi_im(self, thi_image):
        h, w = thi_image.shape
        t = np.argwhere(thi_image!=0)
        kps = self.clean(t, self.c, h, w)
        return kps
    
    def kp_from_skeleton(self, im, thresh=20):
        txt = imutils.skeletonize(im, size=(3, 3))
        kp = np.argwhere(txt>thresh)[::4]
        return im, txt, kp    
    
    def detect(self, thi_image):
        kps = self.kp_from_thi_im(thi_image)
        return kps
        
    def clean(self, kpp, c, h, w):
        '''
        simplify the keypoint set by window size c
        input: 
        kpp: keypoint set (numpy or list of cv2.Keypoints)
        '''
        def clean_window(mask, y, x):
#             mask = mask_tmp.copy()
            x = int(x)
            y = int(y)
            cur = mask[y][x]
            if cur == 1:
                wl = x-c if x>c else 0
                wr = x+c if x+c<w else w
                hu = y-c if y>c else 0
                hd = y+c if y+c<h else h
                mask[hu:hd, wl:wr] = 0
            return cur
        
        mask = np.ones([h, w])
        kpn = []
        
        if isinstance(kpp, np.ndarray):
            for (y, x) in kpp:
                x = int(x)
                y = int(y)
                if mask[y][x]:
                    wl = x-c if x>c else 0
                    wr = x+c if x+c<w else w
                    hu = y-c if y>c else 0
                    hd = y+c if y+c<h else h
                    mask[hu:hd, wl:wr] = 0
                    kpn.append([y, x])
#                 if clean_window(mask, y, x):
#                     kpn.append([y, x])
            kpn = np2kp(kpn, self.ks)
        else:
            for k in kpp:
                x, y = k.pt
                if clean_window(mask, y, x):
                    kpn.append(k)
#         print(len(kpn))
        return kpn