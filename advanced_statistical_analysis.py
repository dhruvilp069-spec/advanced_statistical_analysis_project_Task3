"""
Advanced Statistical Analysis & Hypothesis Testing
===================================================

Project goals:
1. Shapiro-Wilk and Kolmogorov-Smirnov normality tests
2. Independent two-sample t-test
3. Mann-Whitney U test
4. One-Way ANOVA + Tukey HSD
5. Two-Way ANOVA + Tukey HSD
6. 95% confidence intervals
7. Distribution plots and findings

Dataset:
    statistical_dataset.csv

Install:
    pip install pandas numpy scipy statsmodels matplotlib seaborn
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "statistical_dataset.csv"

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("ADVANCED STATISTICAL ANALYSIS & HYPOTHESIS TESTING")
print("=" * 70)
print("\nDataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nGroup counts:")
print(df["Group"].value_counts())

alpha = 0.05
outcome = df["Outcome_Score"]


def decision(p_value):
    return "Reject H0 (statistically significant)" if p_value < alpha else "Fail to reject H0 (not statistically significant)"


# -------------------------------------------------------------------
# 2. Descriptive statistics
# -------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. DESCRIPTIVE STATISTICS")
print("=" * 70)

desc = df.groupby("Group")["Outcome_Score"].agg(
    ["count", "mean", "std", "median", "min", "max"]
)
print(desc.round(3))

# 95% CI for each group mean
print("\n95% Confidence Intervals for group means:")
for group, values in df.groupby("Group")["Outcome_Score"]:
    values = values.dropna()
    mean = values.mean()
    sem = stats.sem(values)
    ci_low, ci_high = stats.t.interval(
        confidence=0.95,
        df=len(values) - 1,
        loc=mean,
        scale=sem
    )
    print(f"{group:12s}: mean={mean:.3f}, 95% CI=({ci_low:.3f}, {ci_high:.3f})")


# -------------------------------------------------------------------
# 3. Distribution plots
# -------------------------------------------------------------------
print("\nCreating distribution plots...")

plt.figure(figsize=(10, 6))
sns.histplot(
    data=df,
    x="Outcome_Score",
    hue="Group",
    kde=True,
    element="step",
    stat="density",
    common_norm=False
)
plt.title("Outcome Score Distribution by Group")
plt.xlabel("Outcome Score")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(BASE_DIR / "distribution_by_group.png", dpi=150)
plt.show()

plt.figure(figsize=(9, 6))
sns.boxplot(data=df, x="Group", y="Outcome_Score")
sns.stripplot(data=df, x="Group", y="Outcome_Score", color="black", alpha=0.35)
plt.title("Outcome Score by Group")
plt.tight_layout()
plt.savefig(BASE_DIR / "boxplot_by_group.png", dpi=150)
plt.show()


# -------------------------------------------------------------------
# 4. Normality tests
# -------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. NORMALITY TESTS")
print("=" * 70)
print("H0: The sample comes from a normal distribution.")
print("H1: The sample does not come from a normal distribution.")
print("Decision rule: p < 0.05 => reject H0.\n")

normality_results = []

for group, values in df.groupby("Group")["Outcome_Score"]:
    values = values.dropna()

    # Shapiro-Wilk
    shapiro_stat, shapiro_p = stats.shapiro(values)

    # One-sample KS against fitted normal distribution
    standardized = (values - values.mean()) / values.std(ddof=1)
    ks_stat, ks_p = stats.kstest(standardized, "norm")

    normality_results.append(
        [group, shapiro_stat, shapiro_p, ks_stat, ks_p]
    )

normality_df = pd.DataFrame(
    normality_results,
    columns=[
        "Group",
        "Shapiro_W",
        "Shapiro_p",
        "KS_D",
        "KS_p"
    ]
)

print(normality_df.round(4))
print("\nInterpretation:")
for _, row in normality_df.iterrows():
    print(
        f"{row['Group']}: "
        f"Shapiro -> {decision(row['Shapiro_p'])}; "
        f"KS -> {decision(row['KS_p'])}"
    )


# -------------------------------------------------------------------
# 5. Two-sample t-test and Mann-Whitney U
# -------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. TWO-SAMPLE TESTS")
print("=" * 70)

# Compare Control vs Treatment_A
control = df.loc[df["Group"] == "Control", "Outcome_Score"]
treatment_a = df.loc[df["Group"] == "Treatment_A", "Outcome_Score"]

print("\nComparison: Control vs Treatment_A")
print("H0: The two population means are equal.")
print("H1: The two population means are different.")

# Welch's independent two-sample t-test
t_stat, t_p = stats.ttest_ind(
    control,
    treatment_a,
    equal_var=False
)

print(f"\nWelch two-sample t-test:")
print(f"t-statistic = {t_stat:.4f}")
print(f"p-value     = {t_p:.6f}")
print(decision(t_p))

# Mann-Whitney U
print("\nH0: The two groups have the same distribution.")
print("H1: The distributions differ.")

u_stat, u_p = stats.mannwhitneyu(
    control,
    treatment_a,
    alternative="two-sided"
)

print("\nMann-Whitney U test:")
print(f"U-statistic = {u_stat:.4f}")
print(f"p-value     = {u_p:.6f}")
print(decision(u_p))

# CI for difference in means
mean_diff = treatment_a.mean() - control.mean()
se_diff = np.sqrt(
    treatment_a.var(ddof=1) / len(treatment_a)
    + control.var(ddof=1) / len(control)
)

df_welch = (
    (treatment_a.var(ddof=1) / len(treatment_a)
     + control.var(ddof=1) / len(control)) ** 2
    /
    (
        (treatment_a.var(ddof=1) / len(treatment_a)) ** 2
        / (len(treatment_a) - 1)
        +
        (control.var(ddof=1) / len(control)) ** 2
        / (len(control) - 1)
    )
)

critical = stats.t.ppf(0.975, df_welch)
ci_low = mean_diff - critical * se_diff
ci_high = mean_diff + critical * se_diff

print(
    f"\nMean difference (Treatment_A - Control) = {mean_diff:.4f}"
)
print(
    f"95% CI for mean difference = ({ci_low:.4f}, {ci_high:.4f})"
)


# -------------------------------------------------------------------
# 6. One-Way ANOVA
# -------------------------------------------------------------------
print("\n" + "=" * 70)
print("6. ONE-WAY ANOVA")
print("=" * 70)

print("H0: All group means are equal.")
print("H1: At least one group mean is different.")

groups_data = [
    df.loc[df["Group"] == g, "Outcome_Score"]
    for g in groups
]

f_stat, anova_p = stats.f_oneway(*groups_data)

print(f"\nF-statistic = {f_stat:.4f}")
print(f"p-value     = {anova_p:.6f}")
print(decision(anova_p))

# ANOVA table via statsmodels
model_1 = ols("Outcome_Score ~ C(Group)", data=df).fit()
anova_table_1 = sm.stats.anova_lm(model_1, typ=2)

print("\nStatsmodels One-Way ANOVA table:")
print(anova_table_1.round(4))


# -------------------------------------------------------------------
# 7. Tukey HSD post-hoc
# -------------------------------------------------------------------
print("\n" + "=" * 70)
print("7. TUKEY HSD POST-HOC ANALYSIS")
print("=" * 70)

print(
    "Used after ANOVA to identify which specific group pairs "
    "have significantly different means."
)

tukey = pairwise_tukeyhsd(
    endog=df["Outcome_Score"],
    groups=df["Group"],
    alpha=0.05
)

print(tukey)

tukey_df = pd.DataFrame(
    data=tukey._results_table.data[1:],
    columns=tukey._results_table.data[0]
)

tukey_df.to_csv(BASE_DIR / "tukey_hsd_results.csv", index=False)


# -------------------------------------------------------------------
# 8. Two-Way ANOVA
# -------------------------------------------------------------------
print("\n" + "=" * 70)
print("8. TWO-WAY ANOVA")
print("=" * 70)

print("Factors: Group and Region")
print("H0 (Group): no group effect.")
print("H0 (Region): no region effect.")
print("H0 (Interaction): no Group × Region interaction.")
print("H1: corresponding effect exists.")

model_2 = ols(
    "Outcome_Score ~ C(Group) * C(Region)",
    data=df
).fit()

anova_table_2 = sm.stats.anova_lm(model_2, typ=2)

print("\nTwo-Way ANOVA table:")
print(anova_table_2.round(4))


# -------------------------------------------------------------------
# 9. Tukey HSD for Group levels in two-way analysis
# -------------------------------------------------------------------
print("\nTukey HSD for Group levels:")
tukey_two_way = pairwise_tukeyhsd(
    endog=df["Outcome_Score"],
    groups=df["Group"],
    alpha=0.05
)
print(tukey_two_way)


# -------------------------------------------------------------------
# 10. Interaction visualization
# -------------------------------------------------------------------
plt.figure(figsize=(9, 6))
sns.pointplot(
    data=df,
    x="Group",
    y="Outcome_Score",
    hue="Region",
    errorbar=("ci", 95)
)
plt.title("Group × Region Interaction with 95% CI")
plt.ylabel("Mean Outcome Score")
plt.tight_layout()
plt.savefig(BASE_DIR / "two_way_interaction.png", dpi=150)
plt.show()


# -------------------------------------------------------------------
# 11. Automated findings
# -------------------------------------------------------------------
print("\n" + "=" * 70)
print("11. KEY FINDINGS")
print("=" * 70)

highest_group = desc["mean"].idxmax()
lowest_group = desc["mean"].idxmin()

print(
    f"1. Highest average outcome: {highest_group} "
    f"({desc.loc[highest_group, 'mean']:.2f})"
)
print(
    f"2. Lowest average outcome: {lowest_group} "
    f"({desc.loc[lowest_group, 'mean']:.2f})"
)
print(
    f"3. One-Way ANOVA p-value = {anova_p:.6f}; "
    f"{'group differences are statistically significant.' if anova_p < alpha else 'no statistically significant group difference was detected.'}"
)
print(
    f"4. Welch t-test p-value = {t_p:.6f}; "
    f"{'Control and Treatment_A differ significantly.' if t_p < alpha else 'no significant difference was detected between Control and Treatment_A.'}"
)
print(
    f"5. Mann-Whitney p-value = {u_p:.6f}; "
    f"{'the distributions differ significantly.' if u_p < alpha else 'no significant distribution difference was detected.'}"
)

print("\nProject completed successfully.")
print("Generated files:")
print("- statistical_dataset.csv")
print("- distribution_by_group.png")
print("- boxplot_by_group.png")
print("- two_way_interaction.png")
print("- tukey_hsd_results.csv")
