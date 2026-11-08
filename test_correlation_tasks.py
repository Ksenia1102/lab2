from pathlib import Path

import pytest

from correlation_tasks import analyze_brain_correlations
from grader_contracts.correlation_tasks import BrainDataInput

# Набор (tab-separated, с пропусками NA), чтобы знать
# ожидаемые коэффициенты корреляции и проверить, что:
# - NA корректно распознаются как пропуски (а не как строка "NA");
# - пропуск в Weight/Height не убирает всю строку из остальных расчётов (попарное, а не построчное удаление NaN);
# - strongest_mri_feature ищет максимум по модулю среди обеих групп.
SAMPLE_BRAIN_DATA = (
    "Gender\tFSIQ\tVIQ\tPIQ\tWeight\tHeight\tMRI_Count\n"
    "Male\t100\t90\t110\t150\t70.0\t1000\n"
    "Male\t110\t95\t105\t160\t72.0\t1100\n"
    "Male\t120\t100\t120\t140\t69.0\t1200\n"
    "Male\t130\t98\t115\tNA\t71.0\t1300\n"
    "Female\t90\t88\t95\t120\t64.0\t800\n"
    "Female\t100\t92\t100\t125\t65.0\t850\n"
    "Female\t110\t97\t105\t130\tNA\t900\n"
    "Female\t95\t85\t90\t118\t63.0\t820\n"
)


@pytest.fixture
def sample_brain_path(tmp_path: Path) -> str:
    path = tmp_path / "brainsize_sample.txt"
    path.write_text(SAMPLE_BRAIN_DATA, encoding="utf-8")
    return str(path)


def test_men_women_counts(sample_brain_path):
    summary = analyze_brain_correlations(BrainDataInput(sample_brain_path))
    assert summary.men_count == 4
    assert summary.women_count == 4


def test_men_correlations(sample_brain_path):
    summary = analyze_brain_correlations(BrainDataInput(sample_brain_path))
    assert summary.men_mri_correlation["FSIQ"] == pytest.approx(1.0)
    assert summary.men_mri_correlation["VIQ"] == pytest.approx(0.860795656933511)
    assert summary.men_mri_correlation["Weight"] == pytest.approx(-0.5)


def test_women_correlations(sample_brain_path):
    summary = analyze_brain_correlations(BrainDataInput(sample_brain_path))
    assert summary.women_mri_correlation["FSIQ"] == pytest.approx(0.998488473754967)
    assert summary.women_mri_correlation["Weight"] == pytest.approx(0.9370830501846087)


def test_missing_value_does_not_drop_whole_row(sample_brain_path):
    """У мужчины с NA в Weight всё равно должны посчитаться FSIQ/VIQ/PIQ/Height."""
    summary = analyze_brain_correlations(BrainDataInput(sample_brain_path))
    assert summary.men_mri_correlation["FSIQ"] == pytest.approx(1.0)  # использует все 4 строки
    assert summary.men_mri_correlation["Weight"] == pytest.approx(-0.5)  # использует только 3


def test_strongest_mri_feature_is_max_abs_across_both_groups(sample_brain_path):
    summary = analyze_brain_correlations(BrainDataInput(sample_brain_path))
    assert summary.strongest_mri_feature == "FSIQ"


def test_real_brainsize_dataset_if_available():
    """Smoke-тест на настоящем brainsize.txt — пропускается, если файла нет рядом."""
    real_path = Path(__file__).parent / "brainsize.txt"
    if not real_path.exists():
        pytest.skip("brainsize.txt не найден рядом с тестом")

    summary = analyze_brain_correlations(BrainDataInput(str(real_path)))
    assert summary.men_count + summary.women_count == 40
    assert summary.strongest_mri_feature in {"FSIQ", "VIQ", "PIQ", "Weight", "Height"}
    for corr in {**summary.men_mri_correlation, **summary.women_mri_correlation}.values():
        assert -1.0 <= corr <= 1.0
