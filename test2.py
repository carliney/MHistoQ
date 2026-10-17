import numpy as np
from scipy.spatial import cKDTree
from ultralytics import YOLO
# from global_vars import cell_metrics
from scipy.spatial.distance import cdist
import cv2


def smooth_contour(contour_xy, window=5):
    """
    滑动窗口均值平滑，专治YOLO输出阶梯直角轮廓
    window: 平滑窗口大小，奇数，越大越圆润，推荐5/7/9/11
    """
    pts = np.array(contour_xy, dtype=np.float32)
    n = len(pts)
    if n < 10:
        return pts
    # 闭合轮廓，首尾循环拼接，处理边界平滑
    pad_pts = np.concatenate([pts[-window//2:], pts, pts[:window//2]])
    smooth_pts = np.zeros_like(pts)
    half_win = window // 2
    # 滑动均值滤波
    for i in range(n):
        seg = pad_pts[i:i+window]
        smooth_pts[i] = seg.mean(axis=0)
    return smooth_pts

def calc_roundness(contour_xy):
    """计算圆度，返回圆度+面积"""
    pts = np.array(contour_xy, dtype=np.int32).reshape((-1, 1, 2))
    area = cv2.contourArea(pts)
    if area < 1e-6:
        return 0.0, area
    peri = cv2.arcLength(pts, closed=True)
    if peri < 1e-6:
        return 0.0, area
    roundness = (4 * np.pi * area) / (peri ** 2)
    return roundness, area

def is_valid_contour(contour_xy, img_w, img_h, min_points=25, edge_gap=15):
    """轮廓有效性校验：点数量、贴边、面积、圆度>=0.5"""
    pts = np.array(contour_xy)
    n_points = len(pts)
    # 过滤1：点太少
    if n_points < min_points:
        return False, 0.0, pts
    # 过滤2：贴图像边缘截断
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    if min_x < edge_gap or min_y < edge_gap or (img_w - max_x) < edge_gap or (img_h - max_y) < edge_gap:
        return False, 0.0, pts
    # 平滑轮廓
    smooth_pts = smooth_contour(contour_xy, window=3)
    r_val, area = calc_roundness(smooth_pts)
    # 过滤3：微小碎片
    if area < 50:
        return False, r_val, smooth_pts
    # 过滤4：圆度小于0.5直接丢弃
    if r_val < 0.5:
        return False, r_val, smooth_pts
    return True, r_val, smooth_pts

all_thick = []
model = YOLO("best-old.pt")
results = model.predict(source='E:/data/C1-DG', save=True, iou=0.28)
cell_metrics=[]
for result in results:
    boxes = result.boxes
    masks = result.masks
    names = result.names
    img = result.orig_img.copy()
    img_h, img_w = img.shape[:2]
    if masks is None:
        print(f"图片 {result.path} 无任何分割掩码，跳过")
        continue
    mask_xy = masks.xy

    cls_arr = boxes.cls.cpu().numpy()
    conf_arr = boxes.conf.cpu().numpy()


    for idx, (box, cls_id) in enumerate(zip(boxes, cls_arr)):
        cls_id = int(cls_id)
        contour_pts = mask_xy[idx]
        val = box.conf.cpu().numpy()
        # 原有壁厚收集
        if (cls_id == 0 or cls_id == 1) and val > 0:
            all_thick.append(val)

        # 仅处理外环类别0
        if cls_id == 0:
            valid, r_val, smooth_pts = is_valid_contour(contour_pts, img_w, img_h, min_points=25, edge_gap=15)
            # 原始轮廓绘图数组
            raw_int = np.array(contour_pts, dtype=np.int32).reshape((-1, 1, 2))
            # 容错：判断平滑后轮廓维度，防止reshape报错
            if len(smooth_pts.shape) == 2 and smooth_pts.shape[1] == 2 and smooth_pts.shape[0] > 3:
                smooth_int = np.array(smooth_pts, dtype=np.int32).reshape((-1, 1, 2))
            else:
                smooth_int = raw_int
                print(f"警告：第{idx}个外环轮廓点数不足，平滑失效，使用原始轮廓绘图")

            # 计算轮廓中心用于标注文字
            M = cv2.moments(smooth_int)
            cX, cY = 0, 0
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

            if valid:
                # 合格：绿色平滑轮廓，写入圆度统计
                # cv2.drawContours(img, [smooth_int], -1, (0, 255, 0), 2)
                # cv2.putText(img, f"R:{r_val:.2f}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cell_metrics.append(r_val)

            #     print(f"有效外环 | 平滑后圆度:{r_val:.3f}")
            # else:
            #     # 不合格：红色原始轮廓，跳过统计
            #     cv2.drawContours(img, [raw_int], -1, (0, 0, 255), 1)
            #     cv2.putText(img, "Skip", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            #     print(f"跳过外环 | 平滑后圆度:{r_val:.3f} < 0.5 / 贴边 / 碎片")

    # 保存可视化图片
    save_name = result.path.split("\\")[-1]
    cv2.imwrite(f"draw_smooth_round_{save_name}", img)

# 全局统计输出
if cell_metrics:
    metric_arr = np.array(cell_metrics)
    round_arr = metric_arr
    print("="*45)
    print(f"有效完整平滑外环总数：{len(cell_metrics)}")
    print(f"全文件夹平均外环圆度：{np.mean(round_arr):.3f}")
else:
    print("无满足条件（圆度≥0.5、不贴边、完整）的外环轮廓")