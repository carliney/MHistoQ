import numpy as np
from ultralytics import YOLO
import cv2
from scipy.spatial.distance import cdist

def smooth_mask(mask, k=3):
    """mask 平滑：去方角、毛刺"""
    mask = (mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return (mask > 127).astype(np.uint8)

all_thick = []
# 像素→微米 比例，按你实际值修改
scale = 50 / 298

model = YOLO("best.pt")
# save=True 会自动保存绘图结果，mask 以平滑后版本展示
results = model.predict(source='E:/data/A2-DG', save=True)

for res in results:
    boxes = res.boxes
    masks = res.masks
    if boxes is None or masks is None:
        continue

    # ========== 1. 平滑所有mask，并原地替换 ==========
    new_masks = []
    for m in masks.data:
        m = m.cpu().numpy().squeeze()
        m_smooth = smooth_mask(m, k=3)
        new_masks.append(m_smooth)
    # 关键：替换原始mask，后续绘图/保存图片 都会用这版平滑mask
    masks.data = np.stack(new_masks)

    # ========== 2. 区分内外框、计算壁厚 ==========
    cls_np = boxes.cls.cpu().numpy()
    idx_outer = np.where(np.isin(cls_np, [0, 1]))[0]
    idx_inner = np.where(cls_np == 2)[0]

    thickness_list = np.zeros(len(boxes))

    for o_idx in idx_outer:
        outer_mask = masks.data[o_idx]
        outer_mask = (outer_mask * 255).astype(np.uint8)
        outer_contours, _ = cv2.findContours(outer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not outer_contours:
            continue
        outer_contour = outer_contours[0].reshape(-1, 2)

        for i_idx in idx_inner:
            inner_mask = masks.data[i_idx]
            inner_mask = (inner_mask * 255).astype(np.uint8)
            inner_contours, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not inner_contours:
                continue
            inner_contour = inner_contours[0].reshape(-1, 2)

            dist_mat = cdist(outer_contour, inner_contour, "euclidean")
            min_dist = np.min(dist_mat, axis=1)
            avg_px = np.mean(min_dist)
            avg_thick = avg_px * scale
            thickness_list[o_idx] = avg_thick
            break

    # ========== 3. 修改 boxes.data（修复conf只读报错） ==========
    data = boxes.data.clone()
    for i in range(len(data)):
        if int(cls_np[i]) in [0, 1] and thickness_list[i] > 0:
            data[i, 4] = float(thickness_list[i])
    boxes.data = data

    # ========== 4. 收集有效壁厚数据 ==========
    conf_arr = boxes.conf.cpu().numpy()
    for val, cls in zip(conf_arr, cls_np):
        if (cls == 0 or cls == 1) and val > 0:
            all_thick.append(val)

# ========== 5. 导出统计文件 & 打印结果 ==========
with open("壁厚汇总.txt", "w", encoding="utf-8") as f:
    for num in all_thick:
        print(num)
        f.write(f"{num:.3f}\n")

if all_thick:
    arr = np.array(all_thick)
    print("有效腺体总数：", len(all_thick))
    print("全文件夹平均壁厚(μm)：", np.mean(arr))
else:
    print("未检测到有效壁厚数据")