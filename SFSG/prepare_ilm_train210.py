"""Prepare ILM NPYs and conditioning metadata for the 210-volume OCTA split."""

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path, PurePosixPath
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

from utils.conditions import index_conditions


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_SPLIT_JSON = PROJECT_DIR / 'jsons/oct6mm_val/train_pathes.json'
SAVE_BASE_PATH = PROJECT_DIR / 'ilm_train210_output'
TARGET_ROWS = (1, 2)  # One-based indices, matching the original detector.
TREND_THRESHOLD = 0.02
MIN_LENGTH = 80
MIN_RISE_RATIO = 0.01
SMOOTH_SIGMA = 1



GLOBAL_SEED = 2026
# Target distribution of the parameter range, for newly deformed samples only.
RANGE_BIN_EDGES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
RANGE_BIN_WEIGHTS = (0.08, 0.15, 0.27, 0.27, 0.15, 0.08)


NORMAL_FOLDERS = [
    "10001", "10007", "10009", "10010", "10013", "10014", "10016", "10017",
    "10021", "10026", "10029", "10034", "10037", "10038", "10039", "10042",
    "10044", "10045", "10049", "10053", "10058", "10061", "10071", "10076",
    "10078", "10080", "10082", "10083", "10085", "10100", "10101", "10104",
    "10108", "10112", "10115", "10120", "10121", "10123", "10124", "10127",
    "10137", "10139", "10143", "10145", "10150", "10153", "10154", "10156",
    "10160", "10167", "10168", "10172", "10173", "10174", "10178", "10179",
    "10185", "10188", "10191", "10194", "10197", "10201", "10207", "10213",
    "10214", "10215", "10217", "10222", "10227", "10229", "10231", "10232",
    "10234"
]
CNV_FOLDERS = [
    "10008", "10018", "10047", "10060", "10107",  # 示例：CNV类别文件夹（可批量添加）
    "10118", "10141", "10200", "10274", "10279", "10291"
]
DR_FOLDERS = [
    "10005", "10024", "10032", "10040", "10050",  # 示例：DR类别文件夹
    "10054", "10056", "10063", "10066", "10068", "10077", "10086", "10091", "10111", "10135", "10151", "10152", "10155", "10157", "10181", "10184", "10189",
    "10196", "10203", "10216", "10221", "10235", "10261", "10262", "10263", "10268", "10281", "10286", "10297", "10300"
]
RVO_FOLDERS = [
    "10027", "10074", "10092", "10149", "10158",  # 示例：RVO类别文件夹
    "10169", "10205", "10289", "10294", "10299"
]

# 2. 三类文件夹的专属参数规则（可按需调整数值）
FOLDER_TYPE_RULES = {
    "CNV": {
        "S_START_RANGE": (60, 280),  # S_START随机范围（更窄，稳定性更高）
        "S_END_MIN_GAP": 60,  # S_END最小间隔（45像素）
        "S_END_MAX": 340,  # S_END最大取值
        "FORCE_SCALE_RULES": 500
    },
    "DR": {
        "S_START_RANGE": (60, 250),  # S_START随机范围
        "S_END_MIN_GAP": 70,  # S_END最小间隔
        "S_END_MAX": 340,  # S_END最大取值
        "FORCE_SCALE_RULES": 500
    },
    "RVO": {
        "S_START_RANGE": (60, 240),  # S_START随机范围（更宽，多样性更高）
        "S_END_MIN_GAP": 80,  # S_END最小间隔
        "S_END_MAX": 340,  # S_END最大取值
        "FORCE_SCALE_RULES": 500
    }
}

# 默认规则（未被归类到CNV/DR/RVO的文件夹使用）
DEFAULT_RULE = {
    "S_START_RANGE": (60, 280),
    "S_END_MIN_GAP": 60,
    "S_END_MAX": 340,
    "FORCE_SCALE_RULES": 600
}


def get_folder_type(folder_name):
    """
    根据文件夹名判断类型（CNV/DR/RVO/默认）
    :param folder_name: 文件夹名（数字字符串，如"10001"）
    :return: 类型字符串（"CNV"/"DR"/"RVO"/"default"）
    """
    if folder_name in NORMAL_FOLDERS:
        return "NORMAL"
    elif folder_name in CNV_FOLDERS:
        return "CNV"
    elif folder_name in DR_FOLDERS:
        return "DR"
    elif folder_name in RVO_FOLDERS:
        return "RVO"
    else:
        return "default"


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


def range_bin_index(value):
    for i, (left, right) in enumerate(zip(RANGE_BIN_EDGES[:-1], RANGE_BIN_EDGES[1:])):
        if left <= value < right or (i == 5 and value == right):
            return i
    raise ValueError(f"Range outside configured bins: {value}")


def rise_limits_for_range(range_value):
    # Continuous upper limit: a narrow range cannot carry a large rise.
    upper = float(np.interp(range_value, RANGE_BIN_EDGES,
                           (0.30, 0.40, 0.50, 0.65, 0.75, 0.85, 0.90)))
    lower = 0.20 if range_value < 0.4 else 0.30 if range_value < 0.5 else 0.40
    return lower, upper


def sample_rise_range_pair(rule, target_bin=None):
    """Sample width first, then compatible rise and feasible anchor coordinates.

    Preserve range=(gap+20*rise/0.9+10)/400 exactly. Range remains an
    estimated control parameter, not measured anatomical protrusion width.
    """
    if target_bin is None:
        target_bin = random.choices(range(6), weights=RANGE_BIN_WEIGHTS, k=1)[0]
    if target_bin not in range(6):
        raise ValueError(f"Invalid range bin {target_bin}")
    left, right = RANGE_BIN_EDGES[target_bin:target_bin + 2]
    start_min, start_max = rule['S_START_RANGE']
    min_gap = rule['S_END_MIN_GAP']
    max_gap = rule['S_END_MAX'] - start_min
    for _ in range(10000):
        desired_range = random.uniform(left, right)
        lower, upper = rise_limits_for_range(desired_range)
        rise = round(random.uniform(lower, upper), 4)
        gap = int(round(400 * desired_range - 20 * rise / 0.9 - 10))
        if not min_gap <= gap <= max_gap:
            continue
        actual_range = round((gap + 20 * rise / 0.9 + 10) / 400, 4)
        if not (left <= actual_range < right or (target_bin == 5 and actual_range == right)):
            continue
        actual_lower, actual_upper = rise_limits_for_range(actual_range)
        if not actual_lower <= rise <= actual_upper:
            continue
        feasible_start_max = min(start_max, rule['S_END_MAX'] - gap)
        if feasible_start_max < start_min:
            continue
        start = random.randint(start_min, feasible_start_max)
        return start, start + gap, rise, actual_range
    raise ValueError(f"No feasible rise/range pair for bin {target_bin}, rule={rule}")


def make_range_schedule(count):
    """Largest-remainder integer quotas, shuffled independently of force RNG."""
    if count <= 0:
        return [], [0] * 6
    raw = np.asarray(RANGE_BIN_WEIGHTS) * count
    quotas = np.floor(raw).astype(int)
    remaining = count - int(quotas.sum())
    order = sorted(range(6), key=lambda i: (-(raw[i] - quotas[i]), i))
    for i in order[:remaining]:
        quotas[i] += 1
    # Keep at least 5% in each edge bin when a batch has >=2 eligible samples.
    # A single sample cannot occupy both edge bins.
    if count >= 2:
        minimum = math.ceil(0.05 * count)
        for edge in (0, 5):
            while quotas[edge] < minimum:
                candidates = [i for i in range(6) if i != edge and
                              quotas[i] > (minimum if i in (0, 5) else 0)]
                donor = max(candidates, key=lambda i: (quotas[i] - raw[i], quotas[i]))
                quotas[donor] -= 1
                quotas[edge] += 1
    schedule = [i for i, quota in enumerate(quotas) for _ in range(int(quota))]
    random.Random(GLOBAL_SEED + 210).shuffle(schedule)
    return schedule, [int(q) for q in quotas]


def process_finite_element_modeling(
        npy_input_path,
        save_base_path='./output/',
        visualize=True,
        input_file_name=None,  # 指定输入文件名（用于拼接保存路径）
        folder_name=None,  # 当前文件所属文件夹名（数字字符串）
        target_range_bin=None
):
    """
    对npy数组的第2层进行有限元建模，按文件夹类型使用对应参数规则
    返回：(processed_data, deformed_y, force_scale, range_param, (rise, range_param))
    """
    data_all = npy_input_path.astype(np.float32)

    # 1. 判断文件夹类型，获取对应参数规则
    folder_type = get_folder_type(folder_name) if folder_name else "default"
    rule = FOLDER_TYPE_RULES.get(folder_type, DEFAULT_RULE)

    # Joint width/rise sampling replaces the former independent anchor draw.
    S_START, S_END, FORCE_SCALE, range_param = sample_rise_range_pair(rule, target_range_bin)
    gap = S_END - S_START
    gap_TRANSITION = gap + 20 * (FORCE_SCALE / 0.9) + 10
    # 组合参数：(rise, range)
    rise_range_tuple = (FORCE_SCALE, range_param)

    # 固定参数
    EVERY_LAYER_POINT_NUMBER = 400
    H = 640
    TRANSITION_WIDTH_MIN = 20
    TRANSITION_WIDTH_MAX = 40

    # 提取并预处理第2层数据
    if len(data_all.shape) == 1:
        raise AssertionError(f"输入数据为一维数组，需至少2层数据（形状应为(8,400)或类似）")
    elif data_all.shape[1] != 400:
        raise AssertionError(f"第2层数据维度需为400，当前为{data_all.shape[1]}")

    layer_2_data = data_all[1]
    curve_data = layer_2_data - layer_2_data[0]
    curve_data = -curve_data
    curve_data = curve_data.squeeze()
    assert curve_data.shape == (400,), f"第2层数据需为(400,)，实际为{curve_data.shape}"

    # 计算过渡带宽度和受力幅值
    slope_trans = (TRANSITION_WIDTH_MAX - TRANSITION_WIDTH_MIN) / (0.9 - 0.1)
    transition_width = TRANSITION_WIDTH_MIN + slope_trans * (FORCE_SCALE - 0.1)
    transition_width = round(transition_width)

    numerator = 100 + rule["FORCE_SCALE_RULES"] * (FORCE_SCALE - 0.1)
    top_force_magnitude_2 = -numerator / (S_END - S_START)

    # 有限元建模
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

    # 替换原数据的第2层
    array_1x400 = deformed_y.reshape(1, 400)
    array_1x400 = -array_1x400
    array_1x400 = array_1x400 + data_all[1][1]  # 偏移校准

    # 按FORCE_SCALE分段设置move_up_range
    if 0.1 <= FORCE_SCALE <= 0.3:
        move_up_range = random.uniform(0, 2)
    elif 0.3 < FORCE_SCALE <= 0.6:
        move_up_range = random.uniform(2, 5)
    elif 0.6 < FORCE_SCALE <= 0.9:
        move_up_range = random.uniform(5, 8)

    array_1x400 = array_1x400 - move_up_range  # 向上移动

    data_all[1, :] = array_1x400[0, :]
    processed_data = data_all
    processed_data = np.round(processed_data, decimals=0)

    # 可视化（可选）
    if visualize:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 8))
        x = np.arange(400)
        y_min = 0
        y_max = 640
        for i in range(processed_data.shape[0]):
            y_data = processed_data[i, :]
            y_data_clamped = np.clip(y_data, y_min, y_max)
            plt.plot(x, y_data_clamped, linewidth=2, label=f'Layer {i + 1}')

        plt.ylim(y_max, y_min)  # 反转y轴
        plt.xlim(0, 399)
        plt.xticks(np.arange(0, 401, 50))
        plt.yticks(np.arange(0, 641, 80))
        plt.title(
            f'Processed Data (Type: {folder_type}, Folder: {folder_name}, rise={FORCE_SCALE}, range={range_param})',
            fontsize=14)
        plt.xlabel('Pixel Index (0-399)', fontsize=14)
        plt.ylabel('Y Position (0=Top, 640=Bottom)', fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(alpha=0.3, linestyle='--')
        plt.show()

    # 保存processed_data（保持原目录结构）
    if input_file_name is not None:
        key_parts = input_file_name.split('/')  # ["10001", "1"]
        if len(key_parts) >= 2:
            save_dir = os.path.join(save_base_path, key_parts[0])
            save_file_name = f"{key_parts[1]}.npy"
        else:
            save_dir = save_base_path
            save_file_name = f"{input_file_name}.npy"
    else:
        save_dir = save_base_path
        save_file_name = f"processed_data_{random.randint(1000, 9999)}.npy"

    os.makedirs(save_dir, exist_ok=True)
    save_full_path = os.path.join(save_dir, save_file_name)
    np.save(save_full_path, processed_data)

    return processed_data, deformed_y, FORCE_SCALE, range_param, rise_range_tuple


def map_rise_cond(rise_cond: float) -> float:

    """Map the measured amplitude to the original discrete stress intervals."""

    if rise_cond == 0:

        return 0.0

    intervals = [

        (0, 15, 0.1),

        (15, 30, 0.2),

        (30, 46, 0.3),

        (46, 59, 0.4),

        (59, 78, 0.5),

        (78, 98, 0.6),

        (98, 119, 0.7),

        (119, 140, 0.8)

    ]

    if rise_cond >= 140:

        return 0.9

    if rise_cond < 0:

        return 0.10

    for start, end, base in intervals:

        if start < rise_cond <= end:

            ratio = (rise_cond - start) / (end - start)

            mapped = base + ratio * 0.09

            return round(mapped, 2)

    return 0.9


def measure_boundary(
    data: np.ndarray, target_row: int, trend_threshold: float = TREND_THRESHOLD,
    min_length: int = MIN_LENGTH, min_rise_ratio: float = MIN_RISE_RATIO,
    smooth_sigma: float = SMOOTH_SIGMA,
) -> Tuple[float, float]:
    """Return the amplitude and normalized width of the largest protrusion."""
    _, cols = data.shape


    row = target_row - 1

    pixel_seq = data[row]

    smoothed = gaussian_filter(pixel_seq, sigma=smooth_sigma)

    slope = np.gradient(smoothed)

    slope_smoothed = gaussian_filter(slope, sigma=2.0)

    descending_mask = slope_smoothed < -trend_threshold

    ascending_mask = slope_smoothed > trend_threshold

    ranges = []

    start_col = None

    bottom_col = None

    for col in range(1, cols - 1):

        if descending_mask[col] and not descending_mask[col - 1] and start_col is None:

            start_col = col

        if start_col is not None and not bottom_col:

            if slope_smoothed[col] >= 0 and slope_smoothed[col - 1] < 0:

                bottom_col = col

        if start_col is not None and bottom_col is not None:

            if ascending_mask[col] and not ascending_mask[col + 1]:

                drop_amplitude = smoothed[start_col] - smoothed[bottom_col]

                rise_amplitude = smoothed[col] - smoothed[bottom_col]

                if drop_amplitude > 0 and (rise_amplitude / drop_amplitude) >= min_rise_ratio:

                    ranges.append((start_col, col))

                start_col = None

                bottom_col = None

    valid_ranges = [(s, e) for s, e in ranges if (e - s + 1) >= min_length]

    if not valid_ranges:

        return 0.0, 0.0

    rise_amp = 0.0

    max_range_width = 0.0

    for s, e in valid_ranges:

        bottom_val = np.min(smoothed[s:e])

        current_rise_amp = np.max(smoothed[s:e]) - bottom_val

        current_range_width = e - s + 1


        if current_rise_amp > rise_amp:

            rise_amp = current_rise_amp

            max_range_width = current_range_width

    normalized_range = max_range_width / 400.0

    return float(rise_amp), float(normalized_range)


def detect_conditions(data: np.ndarray) -> Tuple[float, float]:
    """Measure both configured boundaries and retain the strongest protrusion."""
    if data.shape != (8, 400) or not np.isfinite(data).all():
        raise ValueError('Expected a finite boundary array with shape (8, 400)')
    # max() retains the first row when amplitudes tie, matching the original.
    amplitude, width = max(
        (measure_boundary(data, row) for row in TARGET_ROWS),
        key=lambda result: result[0],
    )
    return map_rise_cond(amplitude), width


def read_json(path: Path):
    with path.open(encoding='utf-8-sig') as handle:
        return json.load(handle)


def write_json(path: Path, data):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=4)
        handle.write('\n')


def load_training_keys(path: Path):
    paths = read_json(path)
    if not isinstance(paths, list) or not all(isinstance(item, str) for item in paths):
        raise ValueError('Training split must be a JSON list of PNG paths')
    keys = []
    for entry in paths:
        item = PurePosixPath(entry.replace(chr(92), '/'))
        if (item.suffix.lower() != '.png' or len(item.parts) != 2
                or not item.parts[0].isascii() or not item.parts[0].isdigit()
                or not item.stem.isascii() or not item.stem.isdigit()):
            raise ValueError(f'Expected a relative volume/slice.png path: {entry!r}')
        keys.append(item.with_suffix('').as_posix())
    if len(set(keys)) != len(keys):
        raise ValueError('Training split contains duplicate paths')
    volume_count = len({key.split('/')[0] for key in keys})
    if volume_count != 210:
        raise ValueError(f'Expected 210 training volumes, found {volume_count}')
    return sorted(keys, key=lambda key: tuple(map(int, key.split('/'))))


def prepare_dataset(npy_root, split_json, conditions_json, output_dir, limit=None, visualize=False):
    """Detect rise/range and prepare the deformed or copied NPYs in two passes."""
    if not npy_root.is_dir():
        raise FileNotFoundError(f'NPY root does not exist: {npy_root}')
    if output_dir.resolve().is_relative_to(npy_root.resolve()):
        raise ValueError('Choose an output directory outside the input NPY root')
    keys = load_training_keys(split_json)
    if limit is not None:
        if limit <= 0:
            raise ValueError('--limit must be positive')
        keys = keys[:limit]

    conditions = index_conditions(read_json(conditions_json))
    for key in keys:
        item = conditions.get(key)
        if not isinstance(item, dict) or 'class' not in item:
            raise ValueError(f'Missing condition object or class for {key}')
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f'Choose a new or empty output directory: {output_dir}')

    measurements = {}
    for index, key in enumerate(keys, start=1):
        path = npy_root / f'{key}.npy'
        try:
            measurements[key] = detect_conditions(np.load(path, allow_pickle=False))
        except (OSError, ValueError) as exc:
            raise ValueError(f'Failed to measure {path}: {exc}') from exc
        if index % 1000 == 0 or index == len(keys):
            print(f'Measured {index}/{len(keys)} boundaries', flush=True)

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    eligible = [key for key in keys if measurements[key][0] == 0.0]
    schedule, planned_counts = make_range_schedule(len(eligible))
    bins = dict(zip(eligible, schedule))
    actual_counts = [0] * len(RANGE_BIN_WEIGHTS)
    metadata = {}
    copied = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_output = output_dir / 'GT_Layers_npy'
    for index, key in enumerate(keys, start=1):
        source = npy_root / f'{key}.npy'
        rise, width = measurements[key]
        if key in bins:
            _, _, _, width, rise_range = process_finite_element_modeling(
                np.load(source, allow_pickle=False),
                save_base_path=str(npy_output),
                visualize=visualize,
                input_file_name=key,
                folder_name=key.split('/')[0],
                target_range_bin=bins[key],
            )
            actual_counts[range_bin_index(width)] += 1
        else:
            destination = npy_output / f'{key}.npy'
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            rise_range = (rise, round(width, 2))
            copied += 1

        record = {
            name: value for name, value in conditions[key].items()
            if name not in {'rise', 'range', 'row', 'mdice', 'folder_type'}
        }
        record['rise_range'] = rise_range
        metadata[key] = record
        if index % 1000 == 0 or index == len(keys):
            print(f'Saved {index}/{len(keys)} NPY files', flush=True)

    total_deformed = sum(actual_counts)
    fractions = [count / total_deformed if total_deformed else 0.0 for count in actual_counts]
    write_json(output_dir / 'range_distribution_train210.json', {
        'range_bin_edges': RANGE_BIN_EDGES,
        'target_fractions': RANGE_BIN_WEIGHTS,
        'eligible_new_deformations': len(eligible),
        'planned_counts': planned_counts,
        'successful_counts': actual_counts,
        'successful_fractions': fractions,
        'edge_at_least_five_percent': min(fractions[0], fractions[-1]) >= 0.05 if total_deformed else None,
        'copied_original_samples_excluded_from_distribution': copied,
        'note': 'The quotas describe the control parameter, not measured protrusion width.',
    })
    metadata_path = output_dir / 'rise_5_train210.json'
    write_json(metadata_path, metadata)
    print(f'Deformed: {total_deformed}, copied: {copied}. Metadata: {metadata_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npy-root', required=True, type=Path, help='Input directory containing volume/slice.npy files')
    parser.add_argument('--train-split', type=Path, default=TRAIN_SPLIT_JSON)
    parser.add_argument('--conditions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=SAVE_BASE_PATH, help='New or empty output directory')
    parser.add_argument('--limit', type=int, help='Process only the first N training slices')
    parser.add_argument('--visualize', action='store_true', help='Display plots during deformation')
    args = parser.parse_args()
    prepare_dataset(args.npy_root, args.train_split, args.conditions, args.output_dir, args.limit, args.visualize)


if __name__ == '__main__':
    main()
