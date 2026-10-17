import numpy as np
from scipy.spatial import cKDTree
from ultralytics import YOLO
from global_vars import cell_metrics
from scipy.spatial.distance import cdist
import cv2
import pandas as pd
from scipy.spatial import distance
from ultralytics.utils.plotting import Annotator

def smooth_contour(contour_xy, sample_num=200):
    """
    轮廓平滑函数：对锯齿多边形做B样条插值，输出顺滑闭合轮廓N×2坐标
    contour_xy: 原始YOLO轮廓点 (N,2)
    sample_num: 平滑后采样点数，越大越顺滑
    return: 平滑后的轮廓点
    """
    pts = np.array(contour_xy)
    x = pts[:, 0]
    y = pts[:, 1]
    # 闭合轮廓：首尾相连
    x = np.concatenate([x, [x[0]]])
    y = np.concatenate([y, [y[0]]])
    # 累计弧长参数化
    t = np.zeros(len(x))
    t[1:] = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    t_new = np.linspace(t.min(), t.max(), sample_num)
    # B样条插值平滑
    tck_x = interpolate.splrep(t, x, s=0.8)
    tck_y = interpolate.splrep(t, y, s=0.8)
    x_smooth = interpolate.splev(t_new, tck_x)
    y_smooth = interpolate.splev(t_new, tck_y)
    smooth_pts = np.column_stack([x_smooth, y_smooth])
    return smooth_pts

def calc_roundness(contour_xy):
    pts = np.array(contour_xy, dtype=np.int32).reshape((-1, 1, 2))
    area = cv2.contourArea(pts)
    if area < 1e-6:
        return 0.0, area
    peri = cv2.arcLength(pts, closed=True)
    if peri < 1e-6:
        return 0.0, area
    roundness = (4 * np.pi * area) / (peri ** 2)
    return roundness, area
# 轮廓有效性校验
def is_valid_contour(contour_xy, img_w, img_h, min_points=20):
    pts = np.array(contour_xy)
    n_points = len(pts)
    # 过滤1：点数量不足
    if n_points < min_points:
        return False, 0.0, 0.0
    # 平滑轮廓
    smooth_pts = smooth_contour(contour_xy, sample_num=200)
    r_val, area = calc_roundness(smooth_pts)
    # 过滤3：面积过小碎片
    if area < 50:
        return False, r_val, area
    # 过滤4：圆度小于0.5直接剔除
    if r_val < 0.5:
        return False, r_val, area
    return True, r_val, smooth_pts

all_thick = []
model = YOLO("best.pt")  # 加载训练好的模型 for G
#model = YOLO("best-old.pt")  #for DG
results = model.predict(source='E:/data/b1--DG/0011.jpg',save = True, iou=0.28)

for result in results:
    boxes = result.boxes  # Boxes 对象，用于边界框输出
    masks = result.masks  # Masks 对象，用于分割掩码输出
    names = result.names
    img = result.orig_img.copy()  # 获取原图，用于绘制
    annotator = Annotator(img, line_width=2)
    mask_xy = masks.xy  # 所有轮廓边缘xy坐标列表

    conf_arr = boxes.conf.cpu().numpy()
    cls_arr = boxes.cls.cpu().numpy()  # 取出每个框的类别ID

    # 遍历所有检测目标，区分外环cls=0，计算+绘制圆度
    for idx, (box, cls_id) in enumerate(zip(boxes, cls_arr)):
        cls_id = int(cls_id)
        contour_pts = mask_xy[idx]  # 当前目标轮廓xy坐标
        if cls_id == 0:  # 外环 Gill D
            # 1. 计算圆度
            r_val = calc_roundness(contour_pts)
            cell_metrics.append(r_val)  # 存入全局列表
            # 2. 绘制轮廓
            pts_int = np.array(contour_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.drawContours(img, [pts_int], -1, (0, 255, 0), 2)
            # 3. 计算轮廓中心，在中心标注圆度文本
            M = cv2.moments(pts_int)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(img, f"R:{r_val:.2f}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # 原有壁厚收集逻辑不变
        val = box.conf.cpu().numpy()
        if (cls_id == 0 or cls_id == 1) and val > 0:
            all_thick.append(val)

    # 保存带轮廓+圆度标注的图片
    save_name = result.path.split("\\")[-1]
    cv2.imwrite(f"draw_round_{save_name}", img)

# 全部图片跑完后统计圆度
if cell_metrics:
    metric_arr = np.array(cell_metrics)
    round_arr = metric_arr
    print("有效细胞总数：", len(cell_metrics))
    print("全文件夹平均外环圆度：", np.mean(round_arr))