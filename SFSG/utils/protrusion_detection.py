import numpy as np
from scipy.ndimage import gaussian_filter


def detect_proper_protrusions(data,
                              trend_threshold=0.01,  # 趋势判断阈值（斜率绝对值）
                              min_length=50,  # 最小宽度
                              min_rise_ratio=0,  # 上升幅度需达到下降幅度的比例（过滤微弱回升）
                              smooth_sigma=1):  # 平滑系数
    """
    检测严格"先下降后上升"的突起（排除只下降不上升的区域）
    条件：存在明显下降→达到最低点→明显上升（上升幅度需满足比例）
    """
    if data.shape[0] != 8:
        raise ValueError("npy file must contain 8 rows of pixel data")
    rows, cols = data.shape

    target_rows = range(1, 7)  # Rows 2-7 (indexes 1-6)
    protrusion_info = []

    for row in target_rows:
        pixel_seq = data[row]
        smoothed = gaussian_filter(pixel_seq, sigma=smooth_sigma)

        # 计算斜率（平滑后）
        slope = np.gradient(smoothed)
        slope_smoothed = gaussian_filter(slope, sigma=2.0)

        # 定义下降和上升掩码
        descending_mask = slope_smoothed < -trend_threshold  # 显著下降
        ascending_mask = slope_smoothed > trend_threshold  # 显著上升

        # 定位完整的"下降→上升"周期
        ranges = []
        start_col = None  # 下降开始列
        bottom_col = None  # 最低点列（下降转上升的拐点）

        for col in range(1, cols - 1):
            # 1. 检测下降开始
            if descending_mask[col] and not descending_mask[col - 1] and start_col is None:
                start_col = col

            # 2. 检测下降转上升的拐点（最低点）
            if start_col is not None and not bottom_col:
                # 从下降转为上升的瞬间（斜率由负转正）
                if slope_smoothed[col] >= 0 and slope_smoothed[col - 1] < 0:
                    bottom_col = col  # 记录最低点

            # 3. 检测上升结束（必须有明确的上升过程）
            if start_col is not None and bottom_col is not None:
                # 上升趋势结束（进入平稳或下降）
                if ascending_mask[col] and not ascending_mask[col + 1]:
                    # 验证上升幅度：上升部分需达到下降部分的一定比例
                    drop_amplitude = smoothed[start_col] - smoothed[bottom_col]  # 下降幅度
                    rise_amplitude = smoothed[col] - smoothed[bottom_col]  # 上升幅度

                    # 确保上升幅度足够（至少为下降幅度的min_rise_ratio比例）
                    if drop_amplitude > 0 and (rise_amplitude / drop_amplitude) >= min_rise_ratio:
                        ranges.append((start_col, col))  # 记录完整范围

                    # 重置状态，准备下一个周期
                    start_col = None
                    bottom_col = None

        # 过滤过窄区域
        valid_ranges = [(s, e) for s, e in ranges if (e - s + 1) >= min_length]
        if valid_ranges:
            for s, e in valid_ranges:
                protrusion_info.append([s, e])

    # 选择最长的突起范围
    if protrusion_info:
        longest_protrusion = max(protrusion_info, key=lambda x: x[1] - x[0])
        return longest_protrusion  # 返回最长的突起范围
    else:
        return []  # 如果没有检测到任何突起范围，返回空列表

