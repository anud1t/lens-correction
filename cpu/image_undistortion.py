import cv2
import numpy as np
import json

def get_undistort_maps(img_shape, camera_param_path):
    with open(camera_param_path, 'r') as f:
        camera_param = json.load(f)
    K = np.array(camera_param['K'])
    D = np.array(camera_param['D'])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (img_shape[1], img_shape[0]), cv2.CV_16SC2
    )
    return map1, map2

def undistortion(img, map1, map2, resize=True):
    if resize:
        img = cv2.resize(img, (map1.shape[1], map1.shape[0]))
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    return undistorted_img
