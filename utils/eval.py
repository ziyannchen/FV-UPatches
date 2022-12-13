from tqdm import tqdm
import time
import os
import cv2
import numpy as np
from utils.class_related import results_save
from utils.class_related import homographyMatch
# from utils.utils import *
import warnings

warnings.filterwarnings('ignore')
        

class Evaluator:
    def __init__(self, descriptor, matcher, data_root, device, detector=None, **kargs):
        self.data_root = data_root
        self.thi_files = os.path.join(self.data_root, kargs['thi_file'])
        self.im_files = os.path.join(self.data_root, kargs['im_file'])
        self.pair_file = kargs['pair_file']
        self.device = device
        self.descriptor = descriptor
        self.matcher = matcher
        self.prefix = kargs['prefix']
        self.detector = detector
        self.target_size = (225, 90) # A uniform
        self.load_pair()
        
    def read_im(self, name):
        im = cv2.imread(os.path.join(self.im_files, name+'.bmp'), 0)
        im = cv2.resize(im, self.target_size)
        return im
    
    def read_thi(self, name):
        trim = self.data_config['trim']
        txt = cv2.imread(os.path.join(self.thi_files, name+'.bmp'), -1)
        txt[:trim[0], :] = 0; txt[-1-trim[2]:, :] = 0
        
        return txt

    def local_descriptor(self, im1, im2, kp1, kp2):
        start1 = time.time()
        kp1, des1 = self.descriptor.compute(im1, kp1)
        kp2, des2 = self.descriptor.compute(im2, kp2)
        end1 = time.time()
        
        start2 = time.time()
        goodMatch, p1, p2 = self.matcher.BFmatch(des1, des2)
        match_num, goodMatch = homographyMatch(p1, p2, kp1, kp2, goodMatch)
        end2 = time.time()
        
        return match_num, goodMatch, end1-start1, end2-start2
        
    def __call__(self, save_path, log_file, verbose=True):
        os.makedirs(save_path, exist_ok=True)

        match_nums = [] # list of descriptor matching nums in each image pair 
        labels = [] # label of each image matching pair: 0 denotes imposter and 1 denotes genuine
        extract_time = []
        match_time = []

        for index, i in enumerate(tqdm(self.lss)):
            i = i.replace('\n', '')
            arr = i.split('_flag_')
            judge = int(arr[2])
            name1 = arr[0]; name2 = arr[1]
            im1 = self.read_im(name1)
            im2 = self.read_im(name2)
            
            if self.detector is None:
                kp1 = self.descriptor.detect(im1)
                kp2 = self.descriptor.detect(im2)
            else:
                kp1 = self.detector.detect(self.read_thi(name1))
                kp2 = self.detector.detect(self.read_thi(name2))
            
            match_num, goodMatch, time1, time2 = self.local_descriptor(im1, im2, kp1, kp2)
            
            extract_time.append(time1)
            match_time.append(time2)
            labels.append(judge)
            match_nums.append(match_num)

            if len(match_nums) % 200 == 1 or len(match_nums) == len(self.lss):
                fpr, tpr, eer, threshhold, roc_auc = results_save(labels, match_nums)
                res = f'Iter({index}/{len(self.lss)}):' + 'eer: ' + str(eer) + '; thresh: ' + str(threshhold) + '\n'
                self.logger(log_file, res)
                if verbose:
                    print(res)

        total_time = sum(extract_time)+sum(match_time)
        extract_time = sum(extract_time)/len(self.lss)
        match_time = sum(match_time)/len(self.lss)
        fps = 1/np.mean(extract_time+match_time)
        np.save(os.path.join(save_path, 'fpr-'+self.prefix+'.npy'), fpr)
        np.save(os.path.join(save_path, 'tpr-'+self.prefix+'.npy'), tpr)
        log_info = f'prefix: {self.prefix}\nextract_time: {extract_time}\nmatch_time: {match_time}\ntotal_time: {total_time/len(self.lss)}\nFPS: {fps}\n'
        self.logger(log_file, log_info)

        if verbose:
            print(f'All results are saved to {save_path}')
        return extract_time, match_time, fps, eer, int(threshhold)

    def logger(self, log_file, log_info):
        with open(log_file, 'w') as log_f:
            log_f.write(log_info)

    
    def load_pair(self):
        # to load .txt file which consists of lines of image matching pair file names 
        # format of each line: name1 name2 label (split with a specified separator)
        with open(self.pair_file) as fs:
            lss = fs.readlines()
        self.lss = np.array(lss)
        
        np.random.shuffle(self.lss)
