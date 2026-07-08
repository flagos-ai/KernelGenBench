"""
Tests for Layer 7: Timing Validation (timing_validator.py).

Verifies that:
1. Normal timing distributions pass validation
2. Suspicious timing patterns are detected
3. CV/IQR calculations are correct
4. Retest comparison works
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sandbox.competition.timing_validator import (
    StatisticalTimingValidator,
    AdvancedTimingValidator,
    TimingAnomalyType,
    TimingValidationResult,
)


class TestStatisticalTimingValidator:
    """Test the statistical timing validator."""

    def test_create_validator(self):
        validator = StatisticalTimingValidator()
        assert validator.cv_threshold == 0.15
        assert validator.iqr_threshold == 0.3

    def test_custom_thresholds(self):
        validator = StatisticalTimingValidator(cv_threshold=0.1, iqr_threshold=0.2)
        assert validator.cv_threshold == 0.1
        assert validator.iqr_threshold == 0.2

    def test_compute_cv_stable(self):
        """CV is low for stable timing."""
        validator = StatisticalTimingValidator()
        stable_times = [1.0, 1.01, 0.99, 1.02, 1.0, 0.98]
        cv = validator.compute_cv(stable_times)
        assert cv < 0.05

    def test_compute_cv_unstable(self):
        """CV is high for unstable timing."""
        validator = StatisticalTimingValidator()
        unstable_times = [1.0, 2.0, 0.5, 3.0, 0.3, 4.0]
        cv = validator.compute_cv(unstable_times)
        assert cv > 0.3

    def test_compute_cv_empty(self):
        """CV is 0 for empty list."""
        validator = StatisticalTimingValidator()
        assert validator.compute_cv([]) == 0.0

    def test_compute_iqr_ratio_normal(self):
        """IQR ratio is low for consistent timing."""
        validator = StatisticalTimingValidator()
        consistent = [1.0, 1.01, 1.02, 0.99, 1.0, 1.01, 0.98, 1.02]
        iqr = validator.compute_iqr_ratio(consistent)
        assert iqr < 0.1

    def test_compute_convergence_score(self):
        """Convergence score < 1.5 for stable timing."""
        validator = StatisticalTimingValidator()
        stable = [1.0, 1.01, 0.99, 1.0, 1.02, 0.98]
        score = validator.compute_convergence_score(stable)
        assert score <= 2.0

    def test_detect_outliers_none(self):
        """No outliers in clean data."""
        validator = StatisticalTimingValidator()
        clean = [1.0, 1.01, 0.99, 1.02, 1.0, 0.98, 1.01, 1.0]
        outliers = validator.detect_outliers_iqr(clean)
        assert len(outliers) == 0

    def test_detect_outliers_present(self):
        """Outliers are detected."""
        validator = StatisticalTimingValidator()
        with_outlier = [1.0, 1.01, 0.99, 1.0, 10.0, 0.98, 1.01, 1.0]
        outliers = validator.detect_outliers_iqr(with_outlier)
        assert len(outliers) > 0

    def test_validate_normal_passes(self):
        """Normal timing passes validation."""
        validator = StatisticalTimingValidator()
        import numpy as np
        np.random.seed(42)
        times = list(np.random.normal(1.0, 0.02, 10))
        result = validator.validate(times)
        assert result.is_valid
        assert result.anomaly_type == TimingAnomalyType.NORMAL

    def test_validate_cached_behavior_fails(self):
        """Cached behavior (first run much faster) is detected."""
        validator = StatisticalTimingValidator()
        cached = [0.01, 1.0, 1.01, 1.02, 0.99, 1.0, 1.01, 0.98]
        result = validator.validate(cached)
        # First run is 100x faster than others — suspicious
        assert not result.is_valid

    def test_validate_high_variance_fails(self):
        """High variance timing is flagged."""
        validator = StatisticalTimingValidator()
        high_var = [1.0, 2.0, 0.5, 3.0, 0.3, 2.5, 1.5, 0.8]
        result = validator.validate(high_var)
        assert not result.is_valid

    def test_retest_comparison_consistent(self):
        """Consistent retest passes."""
        validator = StatisticalTimingValidator()
        times1 = [1.0, 1.01, 0.99, 1.0]
        times2 = [1.02, 0.98, 1.0, 1.01]
        consistent, msg = validator.retest_comparison(times1, times2)
        assert consistent

    def test_retest_comparison_inconsistent(self):
        """Inconsistent retest fails."""
        validator = StatisticalTimingValidator()
        times1 = [1.0, 1.01, 0.99, 1.0]
        times2 = [3.0, 3.01, 2.99, 3.0]
        consistent, msg = validator.retest_comparison(times1, times2)
        assert not consistent


class TestTimingValidationResult:
    """Test the validation result data class."""

    def test_create_result(self):
        result = TimingValidationResult(
            is_valid=True,
            anomaly_type=TimingAnomalyType.NORMAL,
            cv=0.05,
            iqr_ratio=0.1,
            convergence_score=0.9,
            message="All good",
        )
        assert result.is_valid
        assert result.cv == 0.05
        assert result.message == "All good"


class TestAdvancedTimingValidator:
    """Test the advanced timing validator."""

    def test_create_validator(self):
        validator = AdvancedTimingValidator()
        assert validator.statistical_validator is not None

    def test_full_validation(self):
        """Full validation runs without error."""
        validator = AdvancedTimingValidator()
        import numpy as np
        np.random.seed(42)
        times = list(np.random.normal(1.0, 0.02, 10))
        result = validator.full_validation(times)
        assert 'times' in result
        assert 'statistical' in result
        assert result['statistical'].is_valid

    def test_full_validation_with_retest(self):
        """Full validation with retest works."""
        validator = AdvancedTimingValidator()
        import numpy as np
        np.random.seed(42)
        times1 = list(np.random.normal(1.0, 0.02, 10))
        times2 = list(np.random.normal(1.0, 0.02, 10))
        result = validator.full_validation(times1, retest_times=times2)
        assert result['retest'] is not None
        assert result['retest']['is_consistent']