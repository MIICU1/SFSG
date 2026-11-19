import math
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import norm  # 补充缺失的norm函数


# ===================== 有限元建模核心类 =====================
class FiniteElementStress():
    def __init__(self, number=400, line_points=None, H=640, x=None, is_ILM=False,
                 swit_direction=1., start=130, end=340, force_magnitude=1.0, correct_angle=0.,
                 correct_force=False, transition_width=30):
        # 核心参数初始化
        self.L = number  # 曲线长度（400）
        self.H = H
        self.N = number - 1  # 节点数（399）
        self.E = 1000000  # 杨氏模量
        self.A = 1.0  # 横截面积

        # 节点坐标（强制转为float64避免类型错误）
        self.x = np.linspace(0, self.L, self.N + 1, dtype=np.float64) if x is None else x.astype(np.float64)
        self.y = line_points.astype(np.float64)  # 输入曲线强制转浮点数

        # 刚度矩阵和载荷向量（float64）
        self.K_vertical = np.zeros((self.N + 1, self.N + 1), dtype=np.float64)
        self.F_vertical = np.zeros((self.N + 1,), dtype=np.float64)

        # 关键参数
        self.start = start  # 受力锚点左边界
        self.end = end  # 受力锚点右边界
        self.correct_angle = correct_angle
        self.is_ILM = is_ILM
        self.force_magnitude = float(force_magnitude)
        self.force_weight = 100000.0
        self.transition_width = transition_width  # 过渡带宽度

    def init_unit_K(self):
        # 初始化单元刚度矩阵
        for i in range(self.N):
            dx = self.x[i + 1] - self.x[i]
            k_local = self.A * self.E / dx * np.array([[1, -1], [-1, 1]], dtype=np.float64)
            self.K_vertical[i:i + 2, i:i + 2] += k_local

    def init(self):
        # 重置刚度矩阵和载荷向量
        self.K_vertical = np.zeros((self.N + 1, self.N + 1), dtype=np.float64)
        self.F_vertical = np.zeros((self.N + 1,), dtype=np.float64)
        self.init_unit_K()

    def customize_angle_force(self):
        # 生成正态分布的受力（带中心偏移和角度抖动）
        angle_degrees = self.get_angle()
        angle_radians = np.radians(angle_degrees)

        # 受力中心随机偏移（-20~20像素）
        center_offset = random.uniform(-20, 20) / ((self.end - self.start) / 10)
        x_start_index = self.start
        x_end_index = self.end
        coord = np.linspace(-5 + center_offset, 5 + center_offset,
                            x_end_index - x_start_index + 1, dtype=np.float64)

        # 生成受力分布
        if self.is_ILM:
            mu = 0
            sigma = 2.5
            force = norm.pdf(coord, mu, sigma) - 0.05
        else:
            force = 0.04 * coord ** 2
        force = force.astype(np.float64)

        # 计算竖直方向受力
        force_y = force * np.sin(angle_radians) * self.force_magnitude * self.force_weight * (-1)
        self.F_vertical[x_start_index: x_end_index + 1] += force_y

    def fix_both_ends(self):
        # 边界条件：仅固定过渡带外侧
        def fix_func(F, K, x, l, r):
            # 左侧固定
            fixed_left_end = max(0, l - self.transition_width)
            if fixed_left_end >= 0:
                K[0:fixed_left_end + 1, :] = 0.0
                K[0:fixed_left_end + 1, 0:fixed_left_end + 1] = np.eye(fixed_left_end + 1, dtype=np.float64)
                F[0:fixed_left_end + 1] = 0.0

            # 右侧固定
            fixed_right_start = min(self.N, r + self.transition_width)
            if fixed_right_start <= self.N:
                K[fixed_right_start:self.N + 1, :] = 0.0
                K[fixed_right_start:self.N + 1, fixed_right_start:self.N + 1] = np.eye(self.N - fixed_right_start + 1,
                                                                                       dtype=np.float64)
                F[fixed_right_start:self.N + 1] = 0.0

            return F, K

        self.F_vertical, self.K_vertical = fix_func(self.F_vertical, self.K_vertical, self.x, self.start, self.end)

    def get_angle(self):
        # 角度随机抖动（-15~15度）
        tranx = (self.end - self.start) / (self.y[self.end] - self.y[self.start] + 0.1)
        angle_jitter = round(random.uniform(-15, 15), 1)
        angle_degrees = 90 + math.atan(tranx) + angle_jitter + self.correct_angle
        return angle_degrees

    def solve_offset(self):
        # 求解位移 + 锚点锁定渐变权重
        Uy = np.linalg.solve(self.K_vertical, self.F_vertical)
        deformed_force_y = self.y + Uy  # 纯受力形变曲线
        final_deformed_y = self.y.copy()  # 最终结果初始化

        # 左侧过渡带
        left_trans_start = max(0, self.start - self.transition_width)
        left_trans_end = self.start
        if left_trans_start < left_trans_end:
            trans_length = left_trans_end - left_trans_start
            weight_deformed = 0.9 - 0.3 * (1 + np.cos(np.linspace(0, np.pi, trans_length, dtype=np.float64)))
            weight_original = 1 - weight_deformed
            final_deformed_y[left_trans_start:left_trans_end] = (
                    deformed_force_y[left_trans_start:left_trans_end] * weight_deformed +
                    self.y[left_trans_start:left_trans_end] * weight_original
            )

        # 受力核心区
        final_deformed_y[self.start:self.end + 1] = deformed_force_y[self.start:self.end + 1]

        # 右侧过渡带
        right_trans_start = self.end
        right_trans_end = min(self.N, self.end + self.transition_width)
        if right_trans_start < right_trans_end:
            trans_length = right_trans_end - right_trans_start
            weight_deformed = 1 - 0.3 * (1 - np.cos(np.linspace(0, np.pi, trans_length, dtype=np.float64)))
            weight_original = 1 - weight_deformed
            final_deformed_y[right_trans_start:right_trans_end] = (
                    deformed_force_y[right_trans_start:right_trans_end] * weight_deformed +
                    self.y[right_trans_start:right_trans_end] * weight_original
            )

        return self.x, final_deformed_y

    def __call__(self):
        self.init()
        self.customize_angle_force()
        self.fix_both_ends()
        return self.solve_offset()


# ===================== 核心功能函数（移除随机种子） =====================
def process_finite_element_modeling(
        npy_input_path,
        save_base_path='./output/',
        visualize=True
):
    """
    对npy数组的第2层进行有限元建模，并替换回原数据后保存（仅运行一次）

    Args:
        npy_input_path (np.ndarray): 输入numpy数组（形状(8,400)）
        save_base_path (str): 结果保存基础路径（默认'./output/'）
        visualize (bool): 是否显示可视化图表（默认True）

    Returns:
        tuple: (processed_data, deformed_y)
            - processed_data: 处理后的完整数据（numpy数组）
            - deformed_y: 建模后的第2层数据
    """
    # 移除所有随机种子初始化逻辑
    data_all = npy_input_path.astype(np.float32)

    # 生成随机参数（仅运行一次）
    S_START = random.randint(60, 280)
    S_END = random.randint(S_START + 61, 340)
    gap = S_END - S_START  # 计算锚点区间宽度

    # 动态生成FORCE_SCALE
    if gap <= 60:
        FORCE_SCALE = round(random.uniform(0.1, 0.5), 2)
    elif 60 < gap <= 150:
        upper_limit = 0.5 + (gap - 100) / 100 * 0.4
        upper_limit = max(upper_limit, 0.15)
        FORCE_SCALE = round(random.uniform(0.15, upper_limit), 2)
    else:
        FORCE_SCALE = round(random.uniform(0.2, 0.9), 2)

    # 固定参数
    EVERY_LAYER_POINT_NUMBER = 400
    H = 640
    TRANSITION_WIDTH_MIN = 20
    TRANSITION_WIDTH_MAX = 40

    # 4. 提取并预处理第2层数据
    if len(data_all.shape) == 1:
        raise AssertionError(f"输入数据为一维数组，需至少2层数据（形状应为(8,400)或类似）")
    elif data_all.shape[1] != 400:
        raise AssertionError(f"第2层数据维度需为400，当前为{data_all.shape[1]}")

    layer_2_data = data_all[1]
    curve_data = layer_2_data - layer_2_data[0]
    curve_data = -curve_data
    curve_data = curve_data.squeeze()
    assert curve_data.shape == (400,), f"第2层数据需为(400,)，实际为{curve_data.shape}"

    # 5. 计算过渡带宽度和受力幅值
    slope_trans = (TRANSITION_WIDTH_MAX - TRANSITION_WIDTH_MIN) / (0.9 - 0.1)
    transition_width = TRANSITION_WIDTH_MIN + slope_trans * (FORCE_SCALE - 0.1)
    transition_width = round(transition_width)

    numerator = 100 + 700 * (FORCE_SCALE - 0.1)
    top_force_magnitude_2 = -numerator / (S_END - S_START)

    # 6. 有限元建模
    fes = FiniteElementStress(
        number=EVERY_LAYER_POINT_NUMBER,
        line_points=curve_data,
        H=H,
        start=max(S_START, 1),
        end=min(S_END, EVERY_LAYER_POINT_NUMBER - 1),
        swit_direction=0.,
        is_ILM=True,
        force_magnitude=top_force_magnitude_2,
        correct_angle=round(random.uniform(-10, 10), 1),
        transition_width=transition_width
    )
    deformed_x, deformed_y = fes()

    # 7. 替换原数据的第2层
    array_1x400 = deformed_y.reshape(1, 400)
    array_1x400 = -array_1x400
    array_1x400 = array_1x400 + data_all[1][1]  # 偏移校准

    # 如需启用随机向上移动，取消下面两行注释
    # move_up_range = random.uniform(0, 20)  # 随机移动幅度（0到20之间）
    # array_1x400 = array_1x400 - move_up_range  # 向上移动（y轴反转，减数值=向上）

    data_all[1, :] = array_1x400[0, :]
    processed_data = data_all


    return processed_data, FORCE_SCALE


# ===================== 运行入口 =====================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preview ILM deformation for a boundary NPY.')
    parser.add_argument('npy_file_path')
    npy_file_path = parser.parse_args().npy_file_path
    data = np.load(npy_file_path)
    # 仅运行一次，无随机种子
    processed_data, deformed_y = process_finite_element_modeling(
        npy_input_path=data,
        save_base_path=None,
        visualize=True
    )