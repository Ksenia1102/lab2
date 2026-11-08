import math
from pathlib import Path

import pytest

from grader_contracts.pandas_tasks import TitanicInput
from pandas_tasks import analyze_titanic

# Датасет с теми же столбцами, что и titanic.csv — чтобы точно знать ожидаемые числа и
# проверить граничные случаи (пропуски возраста, пропуски во всех столбцах, сортировку тарифов)
SAMPLE_CSV = """PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"A",male,22,1,0,T1,7.25,,S
2,1,1,"B",female,38,1,0,T2,71.28,C85,C
3,1,3,"C",female,26,0,0,T3,7.93,,S
4,1,1,"D",female,35,1,0,T4,53.1,C123,S
5,0,3,"E",male,,0,0,T5,8.05,,S
6,0,1,"F",male,54,0,0,T6,51.86,E46,S
7,0,3,"G",male,2,3,1,T7,21.08,,S
8,1,1,"H",female,,0,0,T8,512.33,,C
"""


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> str:
    csv_path = tmp_path / "titanic_sample.csv"
    csv_path.write_text(SAMPLE_CSV, encoding="utf-8")
    return str(csv_path)


def test_row_count(sample_csv_path):
    summary = analyze_titanic(TitanicInput(sample_csv_path))
    assert summary.row_count == 8


def test_missing_by_column(sample_csv_path):
    summary = analyze_titanic(TitanicInput(sample_csv_path))
    assert summary.missing_by_column["Age"] == 2
    assert summary.missing_by_column["Cabin"] == 5
    assert summary.missing_by_column["Survived"] == 0


def test_adults_over_30_count_excludes_missing_age(sample_csv_path):
    summary = analyze_titanic(TitanicInput(sample_csv_path))
    assert summary.adults_over_30_count == 3


def test_mean_age_by_pclass_ignores_nan(sample_csv_path):
    summary = analyze_titanic(TitanicInput(sample_csv_path))
    assert summary.mean_age_by_pclass[1] == pytest.approx((38 + 35 + 54) / 3)
    assert summary.mean_age_by_pclass[3] == pytest.approx((22 + 26 + 2) / 3)
    assert not math.isnan(summary.mean_age_by_pclass[1])


def test_mean_age_by_pclass_keys_are_int(sample_csv_path):
    summary = analyze_titanic(TitanicInput(sample_csv_path))
    assert all(isinstance(key, int) for key in summary.mean_age_by_pclass)


def test_survival_rate_by_pclass(sample_csv_path):
    summary = analyze_titanic(TitanicInput(sample_csv_path))
    assert summary.survival_rate_by_pclass[1] == pytest.approx(3 / 4)
    assert summary.survival_rate_by_pclass[3] == pytest.approx(1 / 4)


def test_highest_fares_sorted_descending(sample_csv_path):
    summary = analyze_titanic(TitanicInput(sample_csv_path))
    assert summary.highest_fares == sorted(summary.highest_fares, reverse=True)
    assert summary.highest_fares[0] == pytest.approx(512.33)
    assert len(summary.highest_fares) == 5


def test_real_titanic_dataset_if_available():
    """Smoke-тест на настоящем titanic.csv — пропускается, если файла нет рядом."""
    real_path = Path(__file__).parent / "titanic.csv"
    if not real_path.exists():
        pytest.skip("titanic.csv не найден рядом с тестом")

    summary = analyze_titanic(TitanicInput(str(real_path)))
    assert summary.row_count == 891
    assert set(summary.mean_age_by_pclass) == {1, 2, 3}
    assert all(0 <= rate <= 1 for rate in summary.survival_rate_by_pclass.values())
    assert len(summary.highest_fares) == 5
