import numpy as np
from scipy.spatial import cKDTree
from ultralytics import YOLO
from global_vars import cell_metrics
from scipy.spatial.distance import cdist
import cv2
import pandas as pd
from scipy.spatial import distance
from ultralytics.utils.plotting import Annotator
all_thick = []
model = YOLO("best.pt")  # 加载训练好的模型 for G
#model = YOLO("best-old.pt")  #for DG
results = model.predict(source='E:/data/b1--DG',save = True, iou=0.28)
# results = model.predict(source='E:/data/A1-G/0015.jpg',save = True)   #for DG
# results = model.predict(source='E:/data/A1-G',save = True,iou=0.28)
# for result in results:
#     img = result.orig_img
#     annotator = Annotator(img)
#     # 关键：把masks和boxes传给annotator
#     annotator.masks = result.masks
#     annotator.boxes = result.boxes
#     # 调用原生绘图
#     img = annotator.draw_boxes(result.boxes, result.names)
#     # 保存/显示
#     cv2.imwrite("result.jpg", img)
for result in results:
    boxes = result.boxes  # Boxes 对象，用于边界框输出
    masks = result.masks  # Masks 对象，用于分割掩码输出
    names = result.names
    # 新加这4行
    # conf_arr = boxes.conf.cpu().numpy()
    # for val in conf_arr:
    #     if val>0:
    #         all_thick.append(val)
    conf_arr = boxes.conf.cpu().numpy()
    cls_arr = boxes.cls.cpu().numpy()  # 取出每个框的类别ID

    for val, cls in zip(conf_arr, cls_arr):
        # 只收集类别0、1，并且数值>0（有效壁厚）；跳过类别2
        if (cls == 0 or cls == 1) and val > 0:
            all_thick.append(val)

#====【位置3：所有图片循环结束！for缩进外面、整个代码最后】====
# 全部跑完再保存+计算
# with open("壁厚汇总.txt","w",encoding="utf-8") as f:
#     for num in all_thick:
#         print(num)
#         f.write(f"{num:.3f}\n")
# with open("壁厚圆度汇总.txt","w",encoding="utf-8") as f:
#     for thick, roundness in cell_metrics:
#         print(f"壁厚:{thick:.3f} 外环圆度:{roundness:.3f}")
#         f.write(f"{thick:.3f}\t{roundness:.3f}\n") # \t制表符分隔，方便Excel打开
# 统计均值（分开统计壁厚、圆度）
if cell_metrics:

    metric_arr = np.array(cell_metrics)
    # thick_arr = metric_arr[:, 0]
    round_arr = metric_arr
    print("有效细胞总数：", len(cell_metrics))
    # print("全文件夹平均壁厚μm：", np.mean(thick_arr))
    print("全文件夹平均外环圆度：", np.mean(round_arr))
# if all_thick:
#     arr = np.array(all_thick)
#     print("总数：",len(all_thick))
#     print("全文件夹平均壁厚μm：",np.mean(arr))
#print(boxes.data)
    # for box, mask in zip(boxes, masks):
    #     for cls, contour in zip(box.cls, mask.xy):
    #
    #         count[i] =contour
    #         # print("shape:",contour)
    #         class_id = cls.item()  # 获取张量的值
    #         class_name = names[class_id]  # 将类别的序号转化成类别名
    #         # print("name:",class_name)
    #         # print("area:", cv2.contourArea(contour))  # 计算轮廓面积
# def calculate_average_distance(points1, points2, block_size=100):
#     tree = cKDTree(points2)
#     total_distance = 0
#     num_points = len(points1)
#
#     for i in range(0, num_points, block_size):
#         block_points1 = points1[i:i+block_size]
#         _, idx = tree.query(block_points1, k=1)
#         nearest_points = points2[idx]
#         total_distance += np.sum(np.linalg.norm(block_points1 - nearest_points, axis=1))
#
#     average_distance = total_distance / num_points
#     return average_distance
# total_distance = 0
# num_pairs =0
# for point_1 in masks[0]:
#     min_dist = np.inf
#     for point_2 in masks[2]:
#         dist = distance.euclidean(point_1,point_2)
#         if dist < min_dist:
#             min_dist = dist
#     total_distance +=min_dist
#     num_pairs +=1
# average_distance = total_distance / num_pairs
# def select_random_points(points, num_points):
#     if len(points) < num_points:
#         return points
#     else:
#         indices = np.random.choice(len(points), num_points, replace=False)
#         return points[indices]

# 计算两个轮廓之间的距离
# def calculate_contour_distance(contour1, contour2):
#     # 计算两个轮廓中所有点之间的距离
#     distances = cdist(contour1, contour2, 'euclidean')
#
#     # 对于每个点，找到最近轮廓上的点的距离
#     min_distances = np.min(distances, axis=1)
#
#     # 取所有最短距离的平均值作为轮廓之间的距离
#     avg_distance = np.mean(min_distances)
#
#
#     return avg_distance

# # 生成示例数据点集
# contour1 = masks.xy[0]  # 第一个轮廓包含80个点
# contour2 = masks.xy[1] # 第二个轮廓包含120个点
# contour3= masks.xy[2]
# contour4= masks.xy[3]
# contour5= masks.xy[4]
# contour6= masks.xy[5]
# #xy点在图像上的可视化
# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(contour1[:, 0], contour1[:, 1], color='blue',s = 10)  # 绘制点
# plt.scatter(contour2[:, 0], contour2[:, 1], color='yellow',s = 10)
# plt.scatter(contour3[:, 0], contour3[:, 1], color='red',s = 10)
# plt.scatter(contour4[:, 0], contour4[:, 1], color='green',s = 10)
# plt.scatter(contour5[:, 0], contour5[:, 1], color='black',s = 10)
# plt.scatter(contour6[:, 0], contour6[:, 1], color='orange',s = 10)
# plt.title('Scatter Plot of Points')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.xlim(0, 4000)  # 设置x轴范围
# plt.ylim(0, 3000)  # 设置y轴范围
# plt.grid(True)
# plt.show()

# print(contour1)
# 从每个轮廓中随机选择100个数据点，如果数据点数量小于100，则选择所有数据点
# num_points_to_select = 1
# # selected_contour1 = select_random_points(contour1, num_points_to_select)
# selected_contour2 = select_random_points(contour2, num_points_to_select)
#
# # 计算两个子集之间的平均距离
# distance = calculate_contour_distance(contour1, contour2)
# print("Average distance between the two selected subsets: ", distance)



