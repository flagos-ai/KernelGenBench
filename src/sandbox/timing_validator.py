"""
Statistical Timing Validator for KernelGenBench.

Validates measured kernel times using statistical methods:
- CV (Coefficient of Variation): stability check
- IQR ratio: outlier resistance
- Convergence score: warmup detection
- Outlier detection via IQR method
- Retest consistency check
"""
import statistics
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AnomalyType(Enum):
    NORMAL = "normal"
    CONVERGENCE_ISSUE = "convergence_issue"
    HIGH_VARIANCE = "high_variance"
    CACHED_BEHAVIOR = "cached_behavior"
    OUTLIER = "outlier"


@dataclass
class ValidationResult:
    is_valid: bool
    anomaly_type: AnomalyType
    cv: float              # Coefficient of Variation
    iqr_ratio: float       # IQR / median
    convergence_score: float
    message: str


class StatisticalTimingValidator:
    """Validates timing measurements with statistical methods.

    Uses CV, IQR, and convergence analysis instead of simple threshold
    checks, which avoids false positives from legitimate warmup effects.
    """

    def __init__(self,
                 cv_threshold: float = 0.15,
                 iqr_threshold: float = 0.3,
                 convergence_threshold: float = 1.5,
                 retest_ratio_threshold: float = 1.3):
        self.cv_threshold = cv_threshold
        self.iqr_threshold = iqr_threshold
        self.convergence_threshold = convergence_threshold
        self.retest_ratio_threshold = retest_ratio_threshold

    def cv(self, times: List[float]) -> float:
        """Coefficient of Variation: std / mean. Lower = more stable."""
        if len(times) < 2:
            return 0.0
        mean = statistics.mean(times)
        if mean == 0:
            return float("inf")
        return statistics.stdev(times) / mean

    def iqr_ratio(self, times: List[float]) -> float:
        """IQR / median. Robust measure of spread."""
        if len(times) < 4:
            return 0.0
        # Pure-Python IQR (avoids numpy dependency)
        s = sorted(times)
        n = len(s)
        q1 = self._percentile(s, 25)
        q3 = self._percentile(s, 75)
        iqr = q3 - q1
        median = self._percentile(s, 50)
        if median == 0:
            return float("inf")
        return iqr / median

    @staticmethod
    def _percentile(sorted_vals: List[float], p: int) -> float:
        """Compute p-th percentile from sorted list."""
        n = len(sorted_vals)
        idx = (p / 100.0) * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

    def convergence_score(self, times: List[float]) -> float:
        """Ratio of CV(second_half) / CV(first_half).

        > 1.0 means later runs are less stable (bad convergence).
        < 1.0 means later runs are more stable (good, expected after warmup).
        """
        if len(times) < 4:
            return 0.0
        mid = len(times) // 2
        cv_first = self.cv(times[:mid])
        if cv_first == 0:
            return 0.0
        return self.cv(times[mid:]) / cv_first

    def detect_outliers(self, times: List[float]) -> List[int]:
        """Detect outlier indices using the 1.5*IQR rule."""
        if len(times) < 4:
            return []
        s = sorted(enumerate(times), key=lambda x: x[1])
        vals = [x[1] for x in s]
        q1 = self._percentile(vals, 25)
        q3 = self._percentile(vals, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return [idx for idx, _ in enumerate(times) if times[idx] < lower or times[idx] > upper]

    def validate(self, times: List[float]) -> ValidationResult:
        """Comprehensive timing validation.

        Returns ValidationResult with anomaly details.
        """
        cv_val = self.cv(times)
        iqr_val = self.iqr_ratio(times)
        conv = self.convergence_score(times)
        outliers = self.detect_outliers(times)

        issues = []

        if cv_val > self.cv_threshold:
            issues.append(f"High CV ({cv_val:.3f} > {self.cv_threshold})")
        if iqr_val > self.iqr_threshold:
            issues.append(f"High IQR ratio ({iqr_val:.3f} > {self.iqr_threshold})")
        if conv > self.convergence_threshold:
            issues.append(f"Poor convergence ({conv:.3f} > {self.convergence_threshold})")
        if len(outliers) > len(times) * 0.25:
            issues.append(f"Many outliers ({len(outliers)}/{len(times)})")

        if not issues:
            return ValidationResult(
                is_valid=True, anomaly_type=AnomalyType.NORMAL,
                cv=cv_val, iqr_ratio=iqr_val, convergence_score=conv,
                message="Timing distribution is normal"
            )

        # Classify anomaly
        if conv > 2.0:
            atype = AnomalyType.CONVERGENCE_ISSUE
        elif cv_val > 0.3:
            atype = AnomalyType.HIGH_VARIANCE
        elif outliers and times[0] in [times[i] for i in outliers]:
            atype = AnomalyType.CACHED_BEHAVIOR
        else:
            atype = AnomalyType.OUTLIER

        return ValidationResult(
            is_valid=False, anomaly_type=atype,
            cv=cv_val, iqr_ratio=iqr_val, convergence_score=conv,
            message="; ".join(issues)
        )

    def check_consistency(self,
                          times1: List[float],
                          times2: List[float]) -> Tuple[bool, str]:
        """Compare two measurement sets for consistency.

        Ensures retest results are within acceptable range.
        """
        m1 = statistics.median(times1)
        m2 = statistics.median(times2)
        if m1 == 0 or m2 == 0:
            return False, "Zero median detected"
        ratio = max(m1, m2) / min(m1, m2)
        if ratio > self.retest_ratio_threshold:
            return False, f"Inconsistent: {ratio:.2f}x (max {self.retest_ratio_threshold}x)"
        return True, f"Consistent: {ratio:.2f}x"


class TimingValidator:
    """High-level timing validator with full validation + retest support."""

    def __init__(self):
        self.stats = StatisticalTimingValidator()

    def full_validation(self,
                        times: List[float],
                        retest_times: Optional[List[float]] = None) -> dict:
        result = {
            "times": times,
            "median_ms": statistics.median(times) * 1000 if times else 0,
            "statistical": self.stats.validate(times),
            "retest": None,
        }
        if retest_times:
            is_cons, msg = self.stats.check_consistency(times, retest_times)
            result["retest"] = {
                "is_consistent": is_cons,
                "message": msg,
            }
        return result
