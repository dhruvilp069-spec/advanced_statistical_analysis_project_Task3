# Advanced Statistical Analysis & Hypothesis Testing

## Objective
Perform rigorous inferential statistics on a reproducible business/scientific-style dataset.

## Tests Included
- Shapiro-Wilk normality test
- Kolmogorov-Smirnov normality test
- Welch independent two-sample t-test
- Mann-Whitney U test
- One-Way ANOVA
- Tukey HSD post-hoc analysis
- Two-Way ANOVA
- 95% confidence intervals
- Distribution, box, and interaction plots

## Dataset
The dataset contains:
- Group: Control, Treatment_A, Treatment_B
- Region: North, South
- Age
- Baseline_Score
- Outcome_Score

## Hypotheses

### Normality
- H0: Data are normally distributed.
- H1: Data are not normally distributed.

### Two-Sample Tests
- H0: The two population means/distributions are equal.
- H1: They are different.

### One-Way ANOVA
- H0: All group means are equal.
- H1: At least one group mean differs.

### Two-Way ANOVA
- H0: No Group effect, no Region effect, and no Group × Region interaction.
- H1: At least one corresponding effect exists.

## Significance Level
alpha = 0.05

## Run
```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn
python advanced_statistical_analysis.py
```

For mentor review, the script can also be copied cell-by-cell into a Jupyter Notebook and saved as:
`Advanced_Statistical_Analysis.ipynb`.
