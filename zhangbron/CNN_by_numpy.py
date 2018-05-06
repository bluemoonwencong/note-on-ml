# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from CNN_function import *

# load image
img_rgb = mpimg.imread('2.png')
img_gray = rgb2gray(img_rgb)
# plt.imshow(img_gray)
print(len(img_rgb.shape))
print(img_rgb.shape)
print(img_rgb.shape[-1])

print(img_gray.shape)
print(img_gray.shape[-1])
# convert image

# initialize filter
# 创建两个3*3 的滤波器
"""如果是彩色图像，则filter应该设成（2,3,3,3（depth））"""
l1_filter = np.zeros((2, 3, 3))
print(l1_filter.shape)
print(l1_filter.shape[-1])

# init filter to detect horizontal edge
l1_filter[0, :, :] = np.array([[[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]]])

# init filter to detect vertical edge
l1_filter[1, :, :] = np.array([[[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]]])

# convolution operator
l1_feature_map = conv(img_gray, l1_filter)
#
# print(l1_feature_map.shape)
# plt.imshow(l1_feature_map[:, :, 0])
# plt.imshow(l1_feature_map[:, :, 1])

# Relu function
l1_feature_map_relu = relu(l1_feature_map)
# print(l1_feature_map_relu.shape)
# plt.imshow(l1_feature_map_relu[:, :, 0])
# plt.imshow(l1_feature_map_relu[:, :, 1])

# max pooling
l1_feature_map_relu_pool = pooling(l1_feature_map_relu, 2, 2)
# print(l1_feature_map_relu_pool.shape)
# plt.imshow(l1_feature_map_relu_pool[:, :, 0])
# plt.imshow(l1_feature_map_relu_pool[:, :, 1])

l2_filter = np.random.rand(3, 5, 5, l1_feature_map_relu_pool.shape[-1])
l2_filter = np.random.rand(3, 5, 5, l1_feature_map_relu_pool.shape[-1])
print("\n**Working with conv layer 2**")
l2_feature_map = conv(l1_feature_map_relu_pool, l2_filter)
print("\n**ReLU**")
l2_feature_map_relu = relu(l2_feature_map)
print("\n**Pooling**")
l2_feature_map_relu_pool = pooling(l2_feature_map_relu, 2, 2)
print("**End of conv layer 2**\n")
print(l2_feature_map_relu_pool.shape)


# Third conv layer
l3_filter = np.random.rand(1, 7, 7, l2_feature_map_relu_pool.shape[-1])
print("\n**Working with conv layer 3**")
l3_feature_map = conv(l2_feature_map_relu_pool, l3_filter)
print("\n**ReLU**")
l3_feature_map_relu = relu(l3_feature_map)
print("\n**Pooling**")
l3_feature_map_relu_pool = pooling(l3_feature_map_relu, 2, 2)
print("**End of conv layer 3**\n")
plt.imshow(l3_feature_map_relu_pool[:, :, 0])
print(l3_feature_map_relu_pool.shape)


plt.show()

