# -*- coding: utf-8 -*-
import sys
sys.path.append("./")
from PIL import Image

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

class Graspability(object):
    def __init__(self, finger_h, finger_w, open_w, rotation_step, depth_step, down_depth):
        """
        h_param: hand parameters
            finger_h [px]: finger height
            finger_w [px]: finger width
            open_w [px]: finger open width
        g_params: graspability parameters
            rotation-stepi [deg]: iteration step for rotation [0,45,90,...]
            depth_stepi [px]: iteration step for depth [0,30,60,..]
            down_depthi [px]: distance of fingertip moving below grasp point before closing
        """
        self.finger_h = finger_h
        self.finger_w = finger_w
        self.open_w = open_w
        self.rotation_step = rotation_step
        self.depth_step = depth_step
        self.down_depth = down_depth  # 50

        self.tplt_size = 500

        self.kernel_size = 75
        self.sigma = 25

    def gen_hand_model(self):
        """
        Open/closing models of gripper
        Args: 
            * when w,h,open_w are None, then use default values 
            * when x,y,theta(degree) all equals None, then use x=0,y=0,theta=0 
        Returns: 
            hand model image
        """

        c = int(self.tplt_size/2)
        how = int(self.open_w/2)
        hfh = int(self.finger_h/2)
        fw = int(self.finger_w)

        ho = np.zeros((self.tplt_size, self.tplt_size), dtype = "uint8") # open
        hc = np.zeros((self.tplt_size, self.tplt_size), dtype = "uint8") # close

        ho[(c-hfh):(c+hfh), (c-how-fw):(c-how)]=255
        ho[(c-hfh):(c+hfh), (c+how):(c+how+fw)]=255
        hc[(c-hfh):(c+hfh), (c-how):(c+how)]=255

        return ho, hc
    
    def rotate_img(self, img, angle, center=None, scale=1.0):
        (h,w) = img.shape[:2]

        if center is None:
            center=(w/2, h/2)

        M = cv2.getRotationMatrix2D(center, angle,scale)
        rotated = cv2.warpAffine(img, M, (w,h))
        return rotated
    
    def takefirst(self,elem):
        return elem[0]

    def normalize_depth(self, img, max_depth=255):
        return np.array(img * (max_depth/np.max(img)), dtype=np.uint8)

    def graspability_map(self, img, hand_open_mask, hand_close_mask):
        """Generate graspability map

        Args:
            img (array): W x H x 3
            hand_open_mask (array): 500 x 500 x 1
            hand_close_mask (arrau): 500 x 500 x1

        Returns:
            candidtates (array): N * [g_score, x, y, theta(degree)]
        """

        candidates = []
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # prepare rotated hand model
        ht_rot, hc_rot = [], []

        # hand_open_mask = Image.fromarray(np.uint8(hand_open_mask))
        # hand_close_mask = Image.fromarray(np.uint8(hand_close_mask))

        for r in np.arange(0, 180, self.rotation_step):
            # ht_= hand_close_mask.rotate(r)
            # hc_ = hand_open_mask.rotate(r)
            # ht_rot.append(np.array(ht_.convert('L')))
            # hc_rot.append(np.array(hc_.convert('L')))
            ht = self.rotate_img(hand_close_mask, r)
            hc = self.rotate_img(hand_open_mask, r)
            ht_rot.append(ht)
            hc_rot.append(hc)


        for d_idx, d in enumerate(np.arange(0, 201, self.depth_step)):

            _, Wc = cv2.threshold(gray, d, 255, cv2.THRESH_BINARY)
            _, Wt = cv2.threshold(gray, d + self.down_depth, 255, cv2.THRESH_BINARY)

            for r_idx, r in enumerate(np.arange(0, 180, self.rotation_step)):
                Hc = hc_rot[r_idx]
                Ht = ht_rot[r_idx]
                print("! Hc size: ", Hc.shape)
                print("! Wc size: ", Wc.shape)
                C = cv2.filter2D(Wc, -1, Hc) #Hc
                T = cv2.filter2D(Wt, -1, Ht) #Ht
                
                C_ = 255-C
                comb = T & C_
                G = cv2.GaussianBlur(comb,(self.kernel_size, self.kernel_size), self.sigma, self.sigma)

                _, thresh = cv2.threshold(comb, 122,255, cv2.THRESH_BINARY)
                ccwt = cv2.connectedComponentsWithStats(thresh)
                res = ccwt[3]

                for i in range(res[:,0].shape[0]):
                    y = int(res[:,1][i])
                    x = int(res[:,0][i])
                    score = G[y][x]
                    if score > 0: 
                        candidates.append([score,x,y,r])
    
        candidates.sort(key=self.takefirst, reverse=True)
        return np.asarray(candidates)


    def grasp_ranking(self, candidates, w, h, n, _dismiss=50, _distance=50):
        """Detect grasp with graspability

        Arguments:
            candidates {list} -- a list of grasps candidates
            w {int} -- image width
            h {int} -- image height
            n {int} -- number of expected grasps

        Keyword Arguments:
            _dismiss {int} -- distance that will not collide with the box (image boundaries) (default: {25})
            _distance {int} -- distance between multiple grasps

        Returns:
            [array] -- an array of sorted grasps [[x,y,r (degree)], ...]
        """
        i = 0
        k = 0
        grasps = []
        if len(candidates) < n:
            n = len(candidates)

        while (k < n and i < len(candidates)):
            x = candidates[i][1]
            y = candidates[i][2]
            ## consider dismiss/distance to rank grasp candidates
            if (_dismiss < x) and (x < w-_dismiss) and (_dismiss < y) and (y < h-_dismiss):
                if grasps == []:
                    grasps.append(candidates[i])
                    k += 1
                else:
                    # check the distance of this candidate and all others
                    g_array = np.array(grasps)
                    x_array = (np.ones(len(grasps)))*x
                    y_array = (np.ones(len(grasps)))*y
                    _d_array = (np.ones(len(grasps)))*_distance
                    if ((x_array - g_array[:,1])**2+(y_array - g_array[:,2])**2 > _d_array**2).all():
                        grasps.append(candidates[i])
                        k += 1
            i += 1
        if grasps == []:
            print("[!] No valid grasps after ranking! ")
            return grasps
        return np.asarray(grasps)[:,1:]

    def grasp_planning(self, img, n):
        """ Grasp planning for img

        Arguments: 
            img {array} -- [H,W,3] 
            n {int} -- # expected grasps

        Returns:
            [array] -- an array of sorted grasps [[x,y,r (degree)], ...]

        """
        height, width, _ = img.shape
        hand_open_mask, hand_close_mask = self.gen_hand_model()
        candidates = self.graspability_map(img,
                hand_open_mask=hand_open_mask, 
                hand_close_mask=hand_close_mask)

        if candidates.size != 0:
            grasps = self.grasp_ranking(candidates, n=n, h=height, w=width, _dismiss=(self.open_w+self.finger_w*2)*math.atan(45), _distance=25)

        return grasps
    def draw_grasp(self, grasps, img, color=(0,255,0), top_only=False):
        # default: draw top grasp as No.0 of grasp list
        # e.g. top_no=3, draw top grasp as No.3
        grasps = np.array(grasps)
        if len(grasps.shape)==1:
            grasps = np.asarray([grasps])
        
        for j in range(len(grasps)):
            i = len(grasps) -1 - j
            x = int(grasps[i][0])
            y = int(grasps[i][1])
            theta = grasps[i][2]
            
            h, w, _ = img.shape

            how = int(self.open_w/2)
            hfh = int(self.finger_h/2)
            fw = int(self.finger_w)
            ho_line = np.zeros((h, w), dtype="uint8") # open with a line as grasp shape
            ho_line[(y-hfh):(y+hfh), (x-how-fw):(x-how)]=255
            ho_line[(y-hfh):(y+hfh), (x+how):(x+how+fw)]=255
            ho_line[(y-1):(y+1), (x-how):(x+how)]=255
            ho_line[(y-3):(y+3), (x-3):(x+3)]=255
            
            ho_line = Image.fromarray(np.int8(ho_line))
            ho_line_ = ho_line.rotate(theta, center=(x,y))
            mask = np.array(ho_line_.convert('L'))
            # ho_line_ = self.rotate_img(ho_line, theta, center=(x,y))

            if i == 0: 
                r,g,b = color
            else: 
                _ci = int(255/len(grasps))
                r,g,b = [c-_ci*i if (c-_ci*i)>0 else 25 for c in color]

            rgbmask = np.ones((h, w), dtype="uint8")
            rgbmask = np.dstack((np.array(rgbmask * r, 'uint8'), np.array(rgbmask * g, 'uint8'),
                            np.array(rgbmask * b, 'uint8')))
            mask_resized = np.resize(mask, (h,w))
            img[:] = np.where(mask_resized[:h, :w, np.newaxis] == 0, img, rgbmask)
            
            if top_only:
                return img

        return img 
