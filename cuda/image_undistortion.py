import cv2
import os
import numpy as np
import json


def undistortion(img, mapx_gpu, mapy_gpu, resize=True):
    # Upload the image to GPU
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)

    # Perform undistortion on the GPU
    undistorted_gpu_img = cv2.cuda.remap(gpu_img, mapx_gpu, mapy_gpu, interpolation=cv2.INTER_LINEAR)
    undistorted_img = undistorted_gpu_img.download()  # Download back to CPU for further processing or saving
    
    if resize:
        undistorted_img = cv2.resize(undistorted_img, (mapx_gpu.size().width, mapy_gpu.size().height))
    
    return undistorted_img


def get_cuda_maps(width, height, camera_param_path):
    with open(camera_param_path, 'r') as f:
        params = json.load(f)
    
    camera_matrix = np.array(params["K"])
    dist_coeffs = np.array(params["D"]).flatten()
    
    # Use the original camera matrix instead of calculating a new optimal matrix
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, camera_matrix, (width, height), cv2.CV_32FC1
    )
    
    mapx_gpu = cv2.cuda_GpuMat()
    mapy_gpu = cv2.cuda_GpuMat()
    mapx_gpu.upload(mapx)
    mapy_gpu.upload(mapy)
    
    return mapx_gpu, mapy_gpu