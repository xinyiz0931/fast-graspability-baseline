import os
import math
import sys
import cv2
import timeit
import numpy as np
sys.path.append("./")

from graspability import Graspability

def main():
    start = timeit.default_timer()

    data_dir = os.path.abspath("../data")
    img_path = os.path.join(data_dir, "depth.png")
    ret_path = os.path.join(data_dir, "result.png")
    print(f"Read images from {os.path.abspath(img_path)} ...")
    img = cv2.imread(img_path)

    # gripper parameters unit: pixel
    # it is necessary to transform world coor (unit: mm) to image coor (unit: px)

    finger_h = 40
    finger_w = 16
    open_w = 80
    h_params = (finger_h, finger_w, open_w)

    rotation_step = 45 # [deg]
    depth_step = 20
    down_depth = 25

    n_grasp = 5

    g_params = (rotation_step, depth_step, down_depth)

    method = Graspability(*h_params, *g_params)
    # normalize depth values if necessary
    # img = method.normalize_depth(img)
    grasps = method.grasp_planning(img, n_grasp)
    
    print(f"Success! {len(grasps)} grasps detected! ")
    for i in range(n_grasp):
        print("Grasp #%d: (%d, %d, %.1f)" % (i, *grasps[i]))
    ret = method.draw_grasp(grasps, img, top_only=False)

    cv2.imwrite(ret_path, ret)
    
    end = timeit.default_timer()
    print("Time: %.2f s" % (end - start))
    print(f"Save result to {os.path.abspath(ret_path)} ...")
    
    cv2.imshow("Display", ret)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    main()
