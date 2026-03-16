import numpy as np
def _sample_points_on_segments(self, segments, total_points):
    """
    segments: list of [(x1,y1),(x2,y2)]
    total_points: landmark数量
    """
    # 计算每条线段长度
    lengths = []
    for (p1, p2) in segments:
        p1 = np.array(p1)
        p2 = np.array(p2)
        lengths.append(np.linalg.norm(p2 - p1))

    lengths = np.array(lengths)
    total_length = lengths.sum()

    # 按比例分配点数
    points_per_segment = np.maximum(
        1,
        np.round(total_points * lengths / total_length).astype(int)
    )

    # 修正数量误差
    diff = total_points - points_per_segment.sum()
    points_per_segment[0] += diff

    # 生成点
    points = []
    for (p1, p2), n in zip(segments, points_per_segment):
        p1 = np.array(p1)
        p2 = np.array(p2)
        for t in np.linspace(0, 1, n, endpoint=False):
            points.append(p1 + t * (p2 - p1))

    return np.array(points[:total_points])

def _get_letter_segments(self, letter):
    """
    所有字母都归一化在 [-1,1] 空间
    """

    if letter == "S":
        return [
            ((-0.8, 0.8), (0.8, 0.8)),
            ((-0.8, 0.8), (-0.8, 0.0)),
            ((-0.8, 0.0), (0.8, 0.0)),
            ((0.8, 0.0), (0.8, -0.8)),
            ((-0.8, -0.8), (0.8, -0.8)),
        ]

    elif letter == "J":
        return [
            ((-0.8, 0.8), (0.8, 0.8)),
            ((0.0, 0.8), (0.0, -0.6)),
            ((0.0, -0.6), (-0.5, -0.8)),
        ]

    elif letter == "T":
        return [
            ((-0.8, 0.8), (0.8, 0.8)),
            ((0.0, 0.8), (0.0, -0.8)),
        ]

    elif letter == "U":
        return [
            ((-0.8, 0.8), (-0.8, -0.8)),
            ((0.8, 0.8), (0.8, -0.8)),
            ((-0.8, -0.8), (0.8, -0.8)),
        ]

    else:
        raise ValueError("Unknown letter")