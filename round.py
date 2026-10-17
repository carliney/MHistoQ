import os
import json
import cv2
import numpy as np

# -------------------------- 1. 配置路径（直接改这里） --------------------------
# json_folder = r"E:\data\a1--DG-label"
json_folder = r"E:\data\b2-DG-Label"
output_txt = "GillD_外环圆度汇总.txt"

# -------------------------- 2. 圆度计算函数 --------------------------
def calc_contour_roundness(pts):
    """输入轮廓点列表[[x1,y1],[x2,y2]...]，返回0~1圆度"""
    pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    area = cv2.contourArea(pts)
    # 轮廓面积太小，判定无效
    if area < 1e-6:
        return 0.0
    # closed=True 闭合多边形轮廓
    peri = cv2.arcLength(pts, closed=True)
    if peri < 1e-6:
        return 0.0
    # 标准圆度公式
    roundness = (4 * np.pi * area) / (peri ** 2)
    return roundness

# -------------------------- 3. 批量遍历所有JSON文件 --------------------------
# 先清空/新建输出txt，写入表头
with open(output_txt, "w", encoding="utf-8") as f_write:
    f_write.write("文件名\t轮廓编号\t外环圆度\n")

# 全局存放所有圆度，最后统计均值
all_roundness = []

# 循环文件夹里每一个文件
for file_name in os.listdir(json_folder):
    # 只处理json后缀文件
    if not file_name.endswith(".json"):
        continue
    json_path = os.path.join(json_folder, file_name)
    print(f"\n正在处理文件：{file_name}")

    # 容错：损坏的json直接跳过
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"【文件损坏跳过】{file_name} | 错误信息：{e}")
        continue

    # 提取所有 label = "Gill D" 的外环轮廓
    gill_d_contours = []
    for shape in data["shapes"]:
        lab = shape["label"].strip()
        # 匹配你的外环标签 Gill D
        if lab == "Gill D" or lab == "Dill G":
            contour_points = shape["points"]
            gill_d_contours.append(contour_points)

    # 当前文件没有Gill D外环，直接跳过
    if len(gill_d_contours) == 0:
        print(f"  → {file_name} 无 Gill D 外环")
        continue

    # 遍历当前文件每一条外环，计算圆度并写入txt
    for idx, pts in enumerate(gill_d_contours):
        r_val = calc_contour_roundness(pts)
        all_roundness.append(r_val)
        print(f"  轮廓{idx} | 圆度 = {r_val:.3f}")
        # # 追加写入txt
        # with open(output_txt, "a", encoding="utf-8") as f_write:
        #     f_write.write(f"{file_name}\t{idx}\t{r_val:.3f}\n")

# -------------------------- 4. 全部文件跑完，输出全局统计 --------------------------
print("\n========================================")
if len(all_roundness) > 0:
    round_arr = np.array(all_roundness)
    print(f"有效外环总数量：{len(all_roundness)}")
    print(f"全局平均圆度：{np.mean(round_arr):.3f}")
    print(f"最小圆度：{np.min(round_arr):.3f}")
    print(f"最大圆度：{np.max(round_arr):.3f}")
else:
    print("未找到任何 Gill D 外环轮廓！检查标签名是否正确")
print(f"所有结果已保存至：{output_txt}")
