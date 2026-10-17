import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from ultralytics import YOLO
import os

# ---------------------- 1. 定义骨架提取函数 ----------------------
def skeletonize(img):
    img = img.copy()
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel

all_thick = []
model = YOLO("best.pt")
results = model.predict(source='E:/dataset/G/015.jpg', save=True,iou=0.28)

for result in results:
    img = result.orig_img.copy()
    boxes = result.boxes
    masks = result.masks

    if masks is None or boxes is None:
        continue

    pred_boxes = boxes
    pred_masks = masks
    mask = pred_masks.xy

    # 克隆张量，全程操作副本，禁止直接修改原推理张量
    boxes_data = pred_boxes.data.clone()

    # 筛选类别1
    mask_1 = pred_boxes.data[:, 5] == 1
    indices_1 = torch.nonzero(mask_1).squeeze(dim=1)

    if indices_1.size(0) > 0:
        boxes_data[indices_1, 4] = 0.0

        for index_1 in indices_1:
            # 修复：提取标量索引
            GD_index = index_1.cpu().item()
            single_mask = pred_masks.data[GD_index].cpu().numpy()

            # 将小mask缩放至物体真实框尺寸（原图像素尺度）
            mask_resized = cv2.resize(single_mask, (1024, 540), interpolation=cv2.INTER_NEAREST)
            mask_255 = (mask_resized * 255).astype(np.uint8)

            # 2. 提取骨架
            skel = skeletonize(mask_255)
            sk_y, sk_x = np.where(skel > 0)
            total_points = len(sk_x)
            print(total_points)

            # 3. 容错：骨架点过少，厚度置0
            if len(sk_x) < 200:
                boxes_data[index_1, 4] = 0.0
                continue

            cut_ratio = 0.1
            cut_num = int(total_points * cut_ratio)
            # 截取中间80%：跳过前10%、后10%
            mid_sk_y = sk_y[cut_num: total_points - cut_num]
            mid_sk_x = sk_x[cut_num: total_points - cut_num]

            # 4. 计算厚度
            dist_map = distance_transform_edt(mask_255)

            thickness_points = dist_map[mid_sk_y, mid_sk_x] * 2.0


            avg_px = np.mean(thickness_points)

            # avg_distance = avg_px*4/12.15
            avg_distance = avg_px * 4 / 6

            # 5. 只修改克隆后的张量
            boxes_data[index_1, 4] = avg_distance
            all_thick.append(avg_distance)

            # ===================== 拼接4联可视化图 =====================
            h, w = mask_255.shape
            # 图1：原始Mask (灰度)
            img1 = cv2.cvtColor(mask_255, cv2.COLOR_GRAY2BGR)
            cv2.putText(img1, "Original Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 图2：骨架中心线 (灰度)
            img2 = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
            cv2.putText(img2, "Skeleton", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 图3：距离变换热力图
            dist_norm = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img3 = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)
            cv2.putText(img3, "Distance Heatmap", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 图4：Mask + 绿色骨架叠加
            img4 = cv2.cvtColor(mask_255, cv2.COLOR_GRAY2BGR)
            img4[sk_y[:cut_num], sk_x[:cut_num]] = (0, 0, 255)
            img4[sk_y[total_points - cut_num:], sk_x[total_points - cut_num:]] = (0, 0, 255)
            # 中间有效点标绿
            img4[mid_sk_y, mid_sk_x] = (0, 255, 0)
            # img4[skel > 0] = (0, 255, 0)
            cv2.putText(img4, "Mask + Skeleton", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 横向拼接四张图
            row_top = np.hstack([img1, img2])
            row_bottom = np.hstack([img3, img4])
            concat_img = np.vstack([row_top, row_bottom])

            # 绘制厚度文字在拼接图顶部
            text = f"Avg Thickness: {avg_distance:.2f}"
            cv2.putText(concat_img, text, (w - 80, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 保存4联可视化结果
            file_name = os.path.basename(result.path)
            name_no_ext, ext = os.path.splitext(file_name)
            new_vis_name = f"{name_no_ext}_vis_{GD_index}{ext}"
            vis_save_path = os.path.join(result.save_dir, new_vis_name)
            cv2.imwrite(vis_save_path, concat_img)

        # 循环结束，统一写回原张量（只执行一次）
        pred_boxes.data = boxes_data

# 全局统计
if all_thick:
    print("="*40)
    print(f"所有类别1目标总数: {len(all_thick)}")
    print(f"全局平均厚度: {np.mean(all_thick):.2f}")
else:
    print("未检测到类别1目标")
