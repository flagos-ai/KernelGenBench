"""
Layer 6: Statistical Timing Validator (replaces simple threshold checks).
From: triton_competition_anti_cheat_guide.md - Section 8
"""
import statistics
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TimingAnomalyType(Enum):
    """时序异常类型"""
    NORMAL = "normal"
    CONVERGENCE_ISSUE = "convergence_issue"
    HIGH_VARIANCE = "high_variance"
    CACHED_BEHAVIOR = "cached_behavior"
    OUTLIER = "outlier"


@dataclass
class TimingValidationResult:
    """时序校验结果"""
    is_valid: bool
    anomaly_type: TimingAnomalyType
    cv: float  # Coefficient of Variation
    iqr_ratio: float  # IQR ratio
    convergence_score: float
    message: str


class StatisticalTimingValidator:
    """统计时序校验器"""

    def __init__(self,
                 cv_threshold: float = 0.15,  # CV阈值
                 iqr_threshold: float = 0.3,   # IQR阈值
                 convergence_threshold: float = 0.1):
        self.cv_threshold = cv_threshold
        self.iqr_threshold = iqr_threshold
        self.convergence_threshold = convergence_threshold

    def compute_cv(self, times: List[float]) -> float:
        """
        计算变异系数 (Coefficient of Variation)

        CV = std / mean

        工业标准：CV < 0.15 表示稳定
        """
        if not times or len(times) < 2:
            return 0.0

        mean = statistics.mean(times)
        if mean == 0:
            return float('inf')

        std = statistics.stdev(times)
        return std / mean

    def compute_iqr_ratio(self, times: List[float]) -> float:
        """
        计算IQR比率 (Interquartile Range)

        IQR = Q3 - Q1
        IQR_ratio = IQR / median

        工业标准：IQR_ratio < 0.3 表示稳定
        """
        if not times or len(times) < 4:
            return 0.0

        arr = np.array(times)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        median = np.median(arr)

        if median == 0:
            return float('inf')

        return iqr / median

    def compute_convergence_score(self, times: List[float]) -> float:
        """
        计算收敛分数

        检查时间序列是否趋于稳定

        方法：比较前半段和后半段的变异
        """
        if len(times) < 4:
            return 1.0

        mid = len(times) // 2
        first_half = times[:mid]
        second_half = times[mid:]

        cv_first = self.compute_cv(first_half)
        cv_second = self.compute_cv(second_half)

        # 如果后半段比前半段更稳定，说明在收敛
        if cv_first == 0:
            return 1.0

        return cv_second / cv_first

    def detect_outliers_iqr(self, times: List[float]) -> List[int]:
        """
        使用IQR方法检测离群点

        返回离群点的索引
        """
        if len(times) < 4:
            return []

        arr = np.array(times)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = []
        for i, t in enumerate(times):
            if t < lower_bound or t > upper_bound:
                outliers.append(i)

        return outliers

    def validate(self, times: List[float]) -> TimingValidationResult:
        """
        综合校验

        不是看绝对ratio，而是看"是否稳定收敛"
        """
        # 计算统计量
        cv = self.compute_cv(times)
        iqr_ratio = self.compute_iqr_ratio(times)
        convergence = self.compute_convergence_score(times)
        outliers = self.detect_outliers_iqr(times)

        # 判断逻辑
        message_parts = []

        # 1. CV检查
        if cv > self.cv_threshold:
            message_parts.append(f"High CV ({cv:.3f} > {self.cv_threshold})")

        # 2. IQR检查
        if iqr_ratio > self.iqr_threshold:
            message_parts.append(f"High IQR ratio ({iqr_ratio:.3f} > {self.iqr_threshold})")

        # 3. 收敛检查
        if convergence > 1.5:
            message_parts.append(f"Poor convergence ({convergence:.3f})")

        # 4. 离群点检查
        if len(outliers) > len(times) * 0.25:
            message_parts.append(f"Many outliers ({len(outliers)}/{len(times)})")

        # 综合判断
        if not message_parts:
            return TimingValidationResult(
                is_valid=True,
                anomaly_type=TimingAnomalyType.NORMAL,
                cv=cv,
                iqr_ratio=iqr_ratio,
                convergence_score=convergence,
                message="Timing distribution is normal"
            )

        # 判断异常类型
        if convergence > 2.0:
            anomaly_type = TimingAnomalyType.CONVERGENCE_ISSUE
        elif cv > 0.3:
            anomaly_type = TimingAnomalyType.HIGH_VARIANCE
        elif len(outliers) > 0 and times[0] in [times[i] for i in outliers]:
            anomaly_type = TimingAnomalyType.CACHED_BEHAVIOR
        else:
            anomaly_type = TimingAnomalyType.OUTLIER

        return TimingValidationResult(
            is_valid=False,
            anomaly_type=anomaly_type,
            cv=cv,
            iqr_ratio=iqr_ratio,
            convergence_score=convergence,
            message="; ".join(message_parts)
        )

    def retest_comparison(self,
                         times1: List[float],
                         times2: List[float]) -> Tuple[bool, str]:
        """
        比较两次测试的一致性

        用于复测校验
        """
        median1 = statistics.median(times1)
        median2 = statistics.median(times2)

        if median1 == 0 or median2 == 0:
            return False, "Zero median detected"

        ratio = max(median1, median2) / min(median1, median2)

        # 两次测试结果差异不应超过30%
        if ratio > 1.3:
            return False, f"Inconsistent results: {ratio:.2f}x difference"

        return True, f"Consistent: {ratio:.2f}x difference"


class AdvancedTimingValidator:
    """高级时序校验器（支持多种检测方法）"""

    def __init__(self):
        self.statistical_validator = StatisticalTimingValidator()

    def full_validation(self,
                       times: List[float],
                       retest_times: Optional[List[float]] = None) -> dict:
        """完整校验"""
        result = {
            'times': times,
            'statistical': self.statistical_validator.validate(times),
            'retest': None,
        }

        if retest_times:
            is_consistent, msg = self.statistical_validator.retest_comparison(
                times, retest_times
            )
            result['retest'] = {
                'is_consistent': is_consistent,
                'message': msg,
                'retest_times': retest_times
            }

        return result
