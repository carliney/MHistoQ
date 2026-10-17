import os
import json
import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev

# ===================== 配置区 =====================
json_folder = "E:/data/b2--DG-Label"
img_folder = "E:/data/b2--DG"
# json_folder = r"E:\data\C3-DG-label"
# img_folder = r"E:\data\C3-DG"
vis_save_dir = r"D:\desktop\vis_output-a2"
# output_txt = r"D:\desktop\多细胞高精度壁厚汇总-a2.txt"
pixel2real = 6
sample_num = 2000
draw_sample_count = 50
all_cell_thick = []
color_list = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255)
]
# ==================================================
os.makedirs(vis_save_dir, exist_ok=True)

def resample_polygon(poly_pts, num_sample, scale_factor=0.92):
    # 修正变量名笔误
    poly_pts = np.array(poly_pts)
    n = len(poly_pts)
    if n < 3:
        print(f"【轮廓异常】顶点数量过少：{n}，无法插值")
        return None

    # 1. 生成长度和poly_pts完全匹配的mask，解决维度不匹配问题
    diffs = np.diff(poly_pts, axis=0)
    keep_mask = np.ones(n, dtype=bool)  # 初始全部保留
    # 只有前n-1个点需要判断是否和下一个点重复
    keep_mask[:-1] = np.any(diffs != 0, axis=1)
    # 过滤重复点
    clean_poly = poly_pts[keep_mask]

    # 去重后不足4个点，per=True闭合样条无法运行，直接返回空
    if len(clean_poly) < 4:
        print(f"【轮廓异常】去重后有效顶点仅{len(clean_poly)}个，跳过")
        return None

    # 闭合轮廓（首尾相连）
    pts = np.vstack([clean_poly, clean_poly[0]])
    x = pts[:, 0]
    y = pts[:, 1]

    # 捕获splprep插值异常兜底
    try:
        tck, u = splprep([x, y], s=0, per=True)
    except ValueError as e:
        print(f"【插值失败】splprep报错: {e}")
        return None

    u_new = np.linspace(u.min(), u.max(), num_sample)
    x_new, y_new = splev(u_new, tck)
    dense_pts = np.column_stack([x_new, y_new])

    # 以中心向内缩放
    cx = np.mean(dense_pts[:, 0])
    cy = np.mean(dense_pts[:, 1])
    dense_pts[:, 0] = cx + (dense_pts[:, 0] - cx) * scale_factor
    dense_pts[:, 1] = cy + (dense_pts[:, 1] - cy) * scale_factor

    return dense_pts
def get_polygon_center(pts):
    """计算多边形包围框中心，用于匹配内外环"""
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    return np.array([cx, cy])

def calc_contour_diameter(pts):
    pts = np.array(pts)
    max_x, max_y = pts.max(axis=0)
    min_x, min_y = pts.min(axis=0)
    diag = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
    return diag


def draw_single_cell(vis_img, outer_dense, inner_dense, cell_color, draw_n):
    outer_int = outer_dense.astype(np.int32)
    inner_int = inner_dense.astype(np.int32)
    cv2.polylines(vis_img, [outer_int], True, cell_color, thickness=4)
    cv2.polylines(vis_img, [inner_int], True, cell_color, thickness=4)

    dist_mat = cdist(outer_dense, inner_dense)
    min_dist = np.min(dist_mat, axis=1)
    min_idx = np.argmin(dist_mat, axis=1)
    avg_px = np.mean(min_dist)

    rand_idx = np.random.choice(len(outer_dense), size=draw_n, replace=False)
    for i in rand_idx:
        p_o = tuple(outer_dense[i].astype(int))
        p_i = tuple(inner_dense[min_idx[i]].astype(int))
        cv2.line(vis_img, p_o, p_i, (0, 0, 255), thickness=4)
        cv2.circle(vis_img, p_o, 2, (255, 0, 255), -1)
    return vis_img, avg_px

# 批量遍历所有json
for file_name in os.listdir(json_folder):
    if not file_name.endswith(".json"):
        continue
    json_path = os.path.join(json_folder, file_name)
    base_name = os.path.splitext(file_name)[0]
    img_path = None
    for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
        img_candidate = os.path.join(img_folder, base_name + ext)
        if os.path.exists(img_candidate):
            img_path = img_candidate
            break
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"【损坏跳过】{file_name} | 报错：{e}")
        continue

    # 分开收集所有外环Gill D、所有内环inner
    outer_list = []  # 存储所有Gill D轮廓 + 中心
    inner_list = []  # 存储所有inner轮廓 + 中心

    for shape in data["shapes"]:
        lab = shape["label"].strip()
        pts = np.array(shape["points"], dtype=np.float32)
        if len(pts) < 3:
            continue
        center = get_polygon_center(pts)
        if lab == "Dill G":
            outer_list.append({"pts": pts, "center": center})
        elif lab == "inner" or "INNER":
            inner_list.append({"pts": pts, "center": center})

    if len(outer_list) == 0 or len(inner_list) == 0:
        print(f"【缺少Gill D或inner轮廓】{file_name}，跳过")
        continue
    print(f"\n==== 处理文件：{file_name} | 外环数量:{len(outer_list)} 内环数量:{len(inner_list)} ====")

    # 自动配对：每个外环匹配中心距离最近的内环
    valid_cell_pairs = []
    inner_centers = np.array([item["center"] for item in inner_list])
    for outer_item in outer_list:
        oc = outer_item["center"]
        if len(inner_list) == 0:
            print(f"警告：内环已全部匹配完毕，剩余{len(outer_list)}个外环无对应内环，跳过")
            break

        dists = np.linalg.norm(inner_centers - oc, axis=1)
        best_inner_idx = np.argmin(dists)
        min_center_dist = dists[best_inner_idx]
        matched_inner = inner_list[best_inner_idx]

        # 现场计算内环包围盒对角线（等效内环直径）
        inner_pts = np.array(matched_inner["pts"])
        min_xy = inner_pts.min(axis=0)
        max_xy = inner_pts.max(axis=0)
        inner_diag = np.linalg.norm(max_xy - min_xy)
        # 距离超过内环直径，判定不是同一细胞，跳过当前外环，内环保留不pop
        if min_center_dist > inner_diag:
            print(f"外环中心距离最近内环{min_center_dist:.2f}，大于内环直径{inner_diag:.2f}，舍弃该外环")
            continue
        valid_cell_pairs.append({"outer": outer_item["pts"], "inner": matched_inner["pts"]})
        # 匹配过的内环移除，避免重复配对
        inner_list.pop(best_inner_idx)
        inner_centers = np.array([item["center"] for item in inner_list]) if inner_list else np.empty((0,2))

    if len(valid_cell_pairs) == 0:
        print(f"【无匹配成功的细胞】{file_name}，跳过")
        continue

    vis_img = None
    if img_path and os.path.exists(img_path):
        vis_img = cv2.imread(img_path)

    for idx, cell in enumerate(valid_cell_pairs):
        outer_raw = cell["outer"]
        inner_raw = cell["inner"]
        # print("当前待重采样外环原始轮廓：", outer_raw)
        # print("轮廓点数量：", len(outer_raw))
        outer_dense = resample_polygon(outer_raw, sample_num, scale_factor=0.98)
        if outer_dense is None:
            print("当前外环轮廓残缺，跳过该细胞配对")
            continue
        inner_dense = resample_polygon(inner_raw, sample_num, scale_factor=1 )

        dist_matrix = cdist(outer_dense, inner_dense, metric="euclidean")
        min_distances = np.min(dist_matrix, axis=1)
        avg_px_thick = np.mean(min_distances)
        avg_real = avg_px_thick / pixel2real
        all_cell_thick.append(avg_real)
        print(f"细胞{idx} | 像素厚度:{avg_px_thick:.3f} | 实际壁厚μm:{avg_real:.3f}")

        if vis_img is not None:
            color = color_list[idx % len(color_list)]
            vis_img, _ = draw_single_cell(vis_img, outer_dense, inner_dense, color, draw_sample_count)
            text = f"C{idx}:{avg_real:.2f}um"
            text_pt = (20, 40 + idx*50)
            cv2.putText(vis_img, text, text_pt, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

    if vis_img is not None:
        save_path = os.path.join(vis_save_dir, base_name + "_vis.jpg")
        cv2.imwrite(save_path, vis_img)
        # cv2.imshow("Multi-Cell Thickness Visualization", vis_img)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()

# 输出汇总文件
# with open(output_txt, "w", encoding="utf-8") as f:
#     f.write("【多细胞批量测量】每个细胞独立平均壁厚(μm)\n")
#     f.write("-"*45 + "\n")
#     for val in all_cell_thick:
#         f.write(f"{val:.3f}\n")

# 全局统计
if all_cell_thick:
    arr = np.array(all_cell_thick)
    print("\n================ 全部细胞全局统计 ================")
    print(f"总有效细胞数量：{len(all_cell_thick)}")
    print(f"全部细胞平均壁厚：{np.mean(arr):.3f} μm")
    # print(f"壁厚中位数：{np.median(arr):.3f} μm")
    # print(f"标准差：{np.std(arr):.3f} μm")
    # print(f"最大壁厚：{np.max(arr):.3f} μm")
    # print(f"最小壁厚：{np.min(arr):.3f} μm")

    # with open(output_txt, "a", encoding="utf-8") as f:
    #     f.write("\n================ 全局统计汇总 ================\n")
    #     f.write(f"总有效细胞样本数：{len(all_cell_thick)}\n")
    #     f.write(f"全体细胞平均壁厚：{np.mean(arr):.3f} μm\n")
        # f.write(f"中位数壁厚：{np.median(arr):.3f} μm\n")
        # f.write(f"壁厚标准差：{np.std(arr):.3f} μm\n")
else:
    print("\n无任何有效细胞标注！")