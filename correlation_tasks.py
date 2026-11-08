"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]

def _mri_correlations(subset: pd.DataFrame) -> dict[str, float]:
    """Корреляция Пирсона каждого признака из FEATURES с MRI_Count.

    Series.corr() по умолчанию пересчитывается "парно": строки, где у одного
    из двух столбцов NaN (например, пропуски в Weight/Height), просто
    выбрасываются из конкретного расчёта, а не из всей таблицы разом.
    """
    return {
        feature: float(subset[feature].corr(subset["MRI_Count"]))
        for feature in FEATURES
    }

def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])

    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]

    men_mri_correlation = _mri_correlations(men)
    women_mri_correlation = _mri_correlations(women)

    strongest_mri_feature = max(
        FEATURES,
        key=lambda feature: max(
            abs(men_mri_correlation[feature]),
            abs(women_mri_correlation[feature]),
        ),
    )

    return BrainCorrelationSummary(
        men_count=len(men),
        women_count=len(women),
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature,
    )
