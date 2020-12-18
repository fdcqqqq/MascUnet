import tensorflow as tf
import numpy as np


# import torch
# import torch.nn as nn
# import random
# from torch.nn import functional as F
# import math
#
# """
# 克罗内积
# """
# r1 = 4
# r2 = 3
#
# # x = np.array([[[1, 2, 3], [3, 2, 1], [1, 2, 3]], [[1, 2, 3], [3, 2, 1], [1, 2, 3]], [[1, 2, 3], [3, 2, 1], [1, 2, 3]]])
# # y = np.zeros((4, 4, 3))
# # y[0:r2, 0:r2] = 1
# t1 = t2 = torch.randn(2, 2)
#
#
# #
# # x = np.eye(3)
# # y = np.ones((3, 3))
# def kronecker_product(t1, t2):
#     """
#     Computes the Kronecker product between two tensors.
#     See https://en.wikipedia.org/wiki/Kronecker_product
#     """
#     t1_height, t1_width = t1.size()
#     t2_height, t2_width = t2.size()
#     out_height = t1_height * t2_height
#     out_width = t1_width * t2_width
#
#     tiled_t2 = t2.repeat(t1_height, t1_width)
#     expanded_t1 = (
#         t1.unsqueeze(2)
#             .unsqueeze(3)
#             .repeat(1, t2_height, t2_width, 1)
#             .view(out_height, out_width)
#     )
#     return expanded_t1 * tiled_t2
#
#
# print(kronecker_product(t1, t2))
#
# # # A = np.kron(np.eye(3), np.ones((3, 3)))
# # A = np.kron(x, y)
# # print(x)
# # print(y)
# # print(A)
#
# # def kronecker_product(mat1, mat2):
# #     return torch.ger(mat1.view(-1), mat2.view(-1))
# # # .reshape(*(mat1.size() + mat2.size())).permute(
# #     #     [0, 2, 1, 3]).reshape(mat1.size(0) * mat2.size(0), mat1.size(1) * mat2.size(1))
# # print(kronecker_product(x,y))
# # def aggregate(gate, D, I, K, sort=False):
# #     if sort:
# #         _, ind = gate.sort(descending=True)
# #         gate = gate[:, ind[0, :]]
# #
# #     U = [(gate[0, i] * D + gate[1, i] * I) for i in range(K)]
# #     while len(U) != 1:
# #         temp = []
# #         for i in range(0, len(U) - 1, 2):
# #             temp.append(kronecker_product(U[i], U[i + 1]))
# #         if len(U) % 2 != 0:
# #             temp.append(U[-1])
# #         del U
# #         U = temp
# #
# #     return U[0], gate
# # D = torch.eye(2)
# # I = torch.ones(2, 2)
# # K = int(math.log2(32))
# # print(D)
# # print(I)
# # print(K)
# # U,gate = kronecker_product()
#
# # results = kronecker_product(x, y)
# # print(results)
# # import pandas as pd
# # import time
# #
# # # 用numpy创建一个 10x5 矩阵
# # # 加到默认图中.
# #
# # a = np.random.random((10, 5))
# # print(a)
# # # b = []
# # # b.append(list(a))
# # matrix1 = tf.convert_to_tensor(a)  # 用于计算相乘和点乘（内积）的矩阵
# # matrix0 = tf.constant(a, shape=[1, 10, 5, 1])  # 用于计算卷积的矩阵，输入一样，但需要指定特殊的shape
# # print(matrix0)
# # print(matrix1)
# # # exit()
# # b = np.random.random((5, 3))  # 计算矩阵相乘的矩阵shape
# # print(b)
# # c = np.random.random((1, 5))  # 计算内积（矩阵点乘）的矩阵shape
# # print(c)
# # matrix2 = tf.convert_to_tensor(b)
# # print(matrix2)
# # matrix3 = tf.convert_to_tensor(c)
# # print(matrix3)
# # kernel_0 = tf.constant(b, shape=[5, 3, 1, 1])  # 用于计算卷积的矩阵，输入一样，但需要指定特殊的shape
# #
# # product_0 = tf.matmul(matrix1, matrix2)  # 矩阵相乘
# # product_1 = tf.multiply(matrix1, matrix3)  # 矩阵点乘（内积）
# # conv2d = tf.nn.conv2d(matrix0, kernel_0, strides=[1, 1, 1, 1], padding='SAME')  # 卷积
# #
# # # numpy的点乘（内积）和tensorflow的点乘（内积）稍有区别，numpy是点乘完后按行求和
# # df = pd.DataFrame(a)
# # score = np.dot(df, c[0])
# # print(score, np.shape(score))
# #
# #
# # # 启动默认图，执行这个乘法操作,并计算耗时
# # with tf.Session() as sess:
# #     tic = time.time()
# #     result_0 = sess.run([product_0])
# #     result_1 = sess.run([product_1])
# #     result_2 = sess.run([conv2d])
# #     print(result_0, np.shape(result_0))
# #     print(result_1, np.shape(result_1))
# #     print(result_2, np.shape(result_2))
# #     toc = time.time()
# #     print('time cost：'+str(1000*(toc - tic)))

# import tensorflow as tf

#
# kernel = tf._variable_with_weight_decay('weights',
#                                          shape=[5, 5, 3, 64],
#                                          stddev=5e-2,
#                                          wd=0.0)
