"""
FIRE-UdeA — Exploratory Data Analysis (EDA) completo
=====================================================
Genera un reporte detallado del dataset realista para validar
el implementation plan antes de entrenar modelos.

Salida: results/eda/ con CSV, PNG y reporte de texto.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
from scipy import stats
from scipy.interpolate import griddata

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'dataset_sintetico_FIRE_UdeA_realista.csv')
EDA_DIR = os.path.join(SCRIPT_DIR, 'results', 'eda')
os.makedirs(EDA_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.1)
COLORS = {'0': '#4CAF50', '1': '#F44336'}
REPORT_LINES = []


def log(msg):
    """Print and save to report."""
    print(msg)
    REPORT_LINES.append(msg)


# ============================================================
# 1. DATA OVERVIEW
# ============================================================
def section_overview(df):
    log("=" * 70)
    log("1. DATA OVERVIEW")
    log("=" * 70)
    log(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    log(f"Dtypes:\n{df.dtypes.to_string()}")
    log(f"\nYears: {sorted(df['anio'].unique())} ({df['anio'].nunique()} unique)")
    log(f"Units: {sorted(df['unidad'].unique())} ({df['unidad'].nunique()} unique)")
    log(f"\nFirst 5 rows:\n{df.head().to_string()}")
    log(f"\nBasic stats:\n{df.describe().T.to_string()}")


# ============================================================
# 2. TARGET VARIABLE ANALYSIS
# ============================================================
def section_target(df):
    log("\n" + "=" * 70)
    log("2. TARGET VARIABLE ANALYSIS (label)")
    log("=" * 70)

    # Overall distribution
    vc = df['label'].value_counts().sort_index()
    total = len(df)
    log(f"\nOverall distribution:")
    log(f"  label=0 (sano):   {vc.get(0,0)} ({vc.get(0,0)/total*100:.1f}%)")
    log(f"  label=1 (riesgo): {vc.get(1,0)} ({vc.get(1,0)/total*100:.1f}%)")
    log(f"  Prevalence (P=1): {df['label'].mean():.3f}")

    # Per unit
    log(f"\nLabel distribution by unit:")
    pivot_unit = df.pivot_table(index='unidad', columns='label', values='anio', aggfunc='count', fill_value=0)
    pivot_unit.columns = [f'label_{c}' for c in pivot_unit.columns]
    pivot_unit['total'] = pivot_unit.sum(axis=1)
    pivot_unit['prevalence'] = pivot_unit.get('label_1', 0) / pivot_unit['total']
    log(pivot_unit.to_string())

    # Per year
    log(f"\nLabel distribution by year:")
    pivot_year = df.pivot_table(index='anio', columns='label', values='unidad', aggfunc='count', fill_value=0)
    pivot_year.columns = [f'label_{c}' for c in pivot_year.columns]
    pivot_year['prevalence'] = pivot_year.get('label_1', 0) / pivot_year.sum(axis=1)
    log(pivot_year.to_string())

    # Plot: label by unit
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    unit_prev = df.groupby('unidad')['label'].mean().sort_values()
    unit_prev.plot(kind='barh', ax=ax, color='#42A5F5', edgecolor='white')
    ax.set_xlabel('Prevalence (P=1)')
    ax.set_title('Risk Prevalence by Unit')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)

    ax = axes[1]
    year_prev = df.groupby('anio')['label'].mean()
    ax.plot(year_prev.index, year_prev.values, 'o-', color='#F44336', linewidth=2, markersize=8)
    ax.set_xlabel('Year')
    ax.set_ylabel('Prevalence (P=1)')
    ax.set_title('Risk Prevalence by Year')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(-0.05, 1.05)

    plt.suptitle('FIRE-UdeA EDA — Target Distribution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '01_target_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot: heatmap unit × year
    fig, ax = plt.subplots(figsize=(12, 6))
    heat_data = df.pivot_table(index='unidad', columns='anio', values='label', aggfunc='first')
    sns.heatmap(heat_data, annot=True, fmt='.0f', cmap=['#4CAF50', '#F44336'],
                linewidths=1, linecolor='white', ax=ax, cbar=False,
                vmin=0, vmax=1)
    ax.set_title('Risk Label Heatmap (Unit × Year)', fontsize=14, fontweight='bold')
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '02_label_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# 3. MISSING VALUES ANALYSIS
# ============================================================
def section_missing(df):
    log("\n" + "=" * 70)
    log("3. MISSING VALUES ANALYSIS")
    log("=" * 70)

    # Overall missing
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    log(f"\nColumns with missing values:")
    if len(miss) == 0:
        log("  None")
        return
    for col, cnt in miss.items():
        log(f"  {col}: {cnt}/{len(df)} ({cnt/len(df)*100:.1f}%)")

    # Missing by unit
    log(f"\nMissing values by unit:")
    miss_cols = miss.index.tolist()
    miss_by_unit = df.groupby('unidad')[miss_cols].apply(lambda x: x.isnull().sum())
    log(miss_by_unit.to_string())

    # Missing by year
    log(f"\nMissing values by year:")
    miss_by_year = df.groupby('anio')[miss_cols].apply(lambda x: x.isnull().sum())
    log(miss_by_year.to_string())

    # Pattern analysis: are missing values random (MCAR) or systematic?
    log(f"\nMissingness pattern analysis:")
    for col in miss_cols:
        is_miss = df[col].isnull()
        # Correlation with label
        if is_miss.sum() > 0 and is_miss.sum() < len(df):
            label_when_miss = df.loc[is_miss, 'label'].mean()
            label_when_not = df.loc[~is_miss, 'label'].mean()
            log(f"  {col}:")
            log(f"    P(label=1 | missing):     {label_when_miss:.3f}")
            log(f"    P(label=1 | not missing): {label_when_not:.3f}")
            diff = abs(label_when_miss - label_when_not)
            if diff > 0.15:
                log(f"    ⚠ POTENTIAL MNAR — missingness correlated with label (diff={diff:.3f})")
            else:
                log(f"    ✓ Likely MAR/MCAR (diff={diff:.3f})")

    # Plot missing pattern
    fig, ax = plt.subplots(figsize=(12, 6))
    miss_matrix = df[miss_cols].isnull().astype(int)
    miss_matrix.index = [f"{r['anio']}-{r['unidad'][:3]}" for _, r in df.iterrows()]
    sns.heatmap(miss_matrix, cmap=['#FFFFFF', '#F44336'], linewidths=0.5,
                ax=ax, cbar=False, yticklabels=True)
    ax.set_title('Missing Values Pattern (Red = Missing)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '03_missing_pattern.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# 4. FEATURE DISTRIBUTIONS
# ============================================================
def section_distributions(df):
    log("\n" + "=" * 70)
    log("4. FEATURE DISTRIBUTIONS BY LABEL")
    log("=" * 70)

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ['anio', 'label']]

    # Stats by label
    log(f"\nMean by label:")
    means = df.groupby('label')[numeric_cols].mean()
    log(means.T.to_string())

    log(f"\nStd by label:")
    stds = df.groupby('label')[numeric_cols].std()
    log(stds.T.to_string())

    # Statistical tests (Mann-Whitney U for each feature)
    log(f"\nMann-Whitney U tests (label=0 vs label=1):")
    log(f"  {'Feature':<30} {'U-stat':>10} {'p-value':>10} {'Significant':>12}")
    log(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*12}")

    test_results = []
    for col in numeric_cols:
        g0 = df.loc[df['label'] == 0, col].dropna()
        g1 = df.loc[df['label'] == 1, col].dropna()
        if len(g0) >= 3 and len(g1) >= 3:
            stat, pval = stats.mannwhitneyu(g0, g1, alternative='two-sided')
            sig = '✓ Yes' if pval < 0.05 else '  No'
            log(f"  {col:<30} {stat:>10.1f} {pval:>10.4f} {sig:>12}")
            test_results.append({'feature': col, 'U_stat': stat, 'p_value': pval, 'significant': pval < 0.05})
        else:
            log(f"  {col:<30} {'N/A':>10} {'N/A':>10} {'N/A':>12}  (insufficient data)")

    test_df = pd.DataFrame(test_results)
    test_df.to_csv(os.path.join(EDA_DIR, 'feature_tests.csv'), index=False)

    # Plot distributions
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.ravel() if n_rows > 1 else (axes if hasattr(axes, '__len__') else [axes])

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        for label_val in [0, 1]:
            data = df.loc[df['label'] == label_val, col].dropna()
            ax.hist(data, bins=12, alpha=0.6, color=COLORS[str(label_val)],
                    label=f'Label {label_val}', edgecolor='white')
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    # Hide empty subplots
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Feature Distributions by Label', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '04_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Boxplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.ravel() if n_rows > 1 else (axes if hasattr(axes, '__len__') else [axes])

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        df_plot = df[['label', col]].dropna()
        df_plot['label'] = df_plot['label'].astype(str)
        sns.boxplot(data=df_plot, x='label', y=col, ax=ax,
                    palette=COLORS, width=0.5)
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(labelsize=8)

    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Feature Boxplots by Label', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '05_boxplots.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# 5. CORRELATION ANALYSIS
# ============================================================
def section_correlations(df):
    log("\n" + "=" * 70)
    log("5. CORRELATION ANALYSIS")
    log("=" * 70)

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ['anio']]

    corr = df[numeric_cols].corr()

    # Correlation with label
    label_corr = corr['label'].drop('label').sort_values(key=abs, ascending=False)
    log(f"\nCorrelation with label (Pearson):")
    for feat, val in label_corr.items():
        marker = '★' if abs(val) > 0.2 else ' '
        log(f"  {marker} {feat:<30} {val:>8.4f}")

    # High inter-feature correlations
    log(f"\nHigh inter-feature correlations (|r| > 0.7):")
    feat_cols = [c for c in numeric_cols if c != 'label']
    found = False
    for i, c1 in enumerate(feat_cols):
        for c2 in feat_cols[i+1:]:
            r = corr.loc[c1, c2]
            if abs(r) > 0.7:
                log(f"  {c1} ↔ {c2}: r={r:.4f}")
                found = True
    if not found:
        log("  None found.")

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                annot_kws={'size': 7})
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '06_correlation_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# 6. TEMPORAL PATTERNS
# ============================================================
def section_temporal(df):
    log("\n" + "=" * 70)
    log("6. TEMPORAL PATTERNS")
    log("=" * 70)

    # Key features over time by unit
    key_features = ['liquidez', 'gp_ratio', 'endeudamiento', 'tendencia_ingresos',
                    'cfo', 'hhi_fuentes']
    key_features = [f for f in key_features if f in df.columns]

    log(f"\nKey feature means by year:")
    yearly = df.groupby('anio')[key_features].mean()
    log(yearly.to_string())

    # Temporal autocorrelation within units
    log(f"\nTemporal autocorrelation within units (lag-1):")
    for col in key_features:
        autocorrs = []
        for unit in df['unidad'].unique():
            series = df.loc[df['unidad'] == unit, col].dropna()
            if len(series) >= 4:
                ac = series.autocorr(lag=1)
                if not np.isnan(ac):
                    autocorrs.append(ac)
        if autocorrs:
            mean_ac = np.mean(autocorrs)
            log(f"  {col:<30} mean autocorr = {mean_ac:.3f}")

    # Plot time series per unit for key features
    n_feats = min(len(key_features), 6)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for i, col in enumerate(key_features[:6]):
        ax = axes[i]
        for unit in df['unidad'].unique():
            unit_data = df[df['unidad'] == unit].sort_values('anio')
            ax.plot(unit_data['anio'], unit_data[col], 'o-', alpha=0.6,
                    linewidth=1.5, markersize=4, label=unit[:12])
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=6, loc='upper right', ncol=2)

    for j in range(n_feats, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Key Features Over Time by Unit', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '07_temporal_patterns.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Label transitions analysis
    log(f"\nLabel transitions (year-to-year within units):")
    df_sorted = df.sort_values(['unidad', 'anio'])
    df_sorted['label_prev'] = df_sorted.groupby('unidad')['label'].shift(1)
    transitions = df_sorted.dropna(subset=['label_prev'])
    transitions['label_prev'] = transitions['label_prev'].astype(int)
    
    trans_matrix = pd.crosstab(transitions['label_prev'], transitions['label'],
                                rownames=['From'], colnames=['To'])
    log(f"\n  Transition matrix:")
    log(f"  {trans_matrix.to_string()}")
    
    total_trans = len(transitions)
    stay_0 = ((transitions['label_prev'] == 0) & (transitions['label'] == 0)).sum()
    to_risk = ((transitions['label_prev'] == 0) & (transitions['label'] == 1)).sum()
    stay_1 = ((transitions['label_prev'] == 1) & (transitions['label'] == 1)).sum()
    to_safe = ((transitions['label_prev'] == 1) & (transitions['label'] == 0)).sum()
    
    log(f"\n  Transition rates:")
    if stay_0 + to_risk > 0:
        log(f"    P(stay sano | was sano):     {stay_0/(stay_0+to_risk):.3f}")
        log(f"    P(become risk | was sano):   {to_risk/(stay_0+to_risk):.3f}")
    if stay_1 + to_safe > 0:
        log(f"    P(stay risk | was risk):     {stay_1/(stay_1+to_safe):.3f}")
        log(f"    P(recovery | was risk):      {to_safe/(stay_1+to_safe):.3f}")


# ============================================================
# 7. OUTLIER ANALYSIS
# ============================================================
def section_outliers(df):
    log("\n" + "=" * 70)
    log("7. OUTLIER ANALYSIS")
    log("=" * 70)

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ['anio', 'label']]

    log(f"\nOutliers detection (IQR method, 1.5×IQR):")
    log(f"  {'Feature':<30} {'N outliers':>10} {'% outliers':>10} {'Range':>30}")
    log(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*30}")

    total_outliers = {}
    for col in numeric_cols:
        data = df[col].dropna()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = data[(data < lower) | (data > upper)]
        n_out = len(outliers)
        pct = n_out / len(data) * 100 if len(data) > 0 else 0
        total_outliers[col] = n_out
        if n_out > 0:
            log(f"  {col:<30} {n_out:>10} {pct:>9.1f}% [{lower:.2f}, {upper:.2f}]")

    if sum(total_outliers.values()) == 0:
        log("  No outliers found.")

    # Extreme values for CFO (can be very negative)
    log(f"\n  CFO extreme values:")
    if 'cfo' in df.columns:
        cfo_sorted = df[['anio', 'unidad', 'cfo']].dropna().sort_values('cfo')
        log(f"    Bottom 5:\n{cfo_sorted.head(5).to_string(index=False)}")
        log(f"    Top 5:\n{cfo_sorted.tail(5).to_string(index=False)}")


# ============================================================
# 8. FEATURE SCALE ANALYSIS
# ============================================================
def section_scales(df):
    log("\n" + "=" * 70)
    log("8. FEATURE SCALE ANALYSIS")
    log("=" * 70)

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ['anio', 'label']]

    log(f"\n  {'Feature':<30} {'Min':>15} {'Max':>15} {'Scale':>12}")
    log(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*12}")

    for col in numeric_cols:
        data = df[col].dropna()
        mn = data.min()
        mx = data.max()
        scale = np.log10(max(abs(mn), abs(mx), 1))
        log(f"  {col:<30} {mn:>15.2f} {mx:>15.2f} {f'10^{scale:.0f}':>12}")

    log(f"\n  NOTE: Tree-based models are scale-invariant, but this matters for")
    log(f"  linear models (LogReg, ElasticNet) which need StandardScaler.")


# ============================================================
# 9. CLASS SEPARABILITY ANALYSIS
# ============================================================
def section_separability(df):
    log("\n" + "=" * 70)
    log("9. CLASS SEPARABILITY — Key Financial Indicators")
    log("=" * 70)

    # Cohen's d for each feature
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ['anio', 'label']]

    log(f"\n  Cohen's d effect size (label=0 vs label=1):")
    log(f"  {'Feature':<30} {'Cohen d':>10} {'Effect':>12}")
    log(f"  {'-'*30} {'-'*10} {'-'*12}")

    effect_sizes = []
    for col in numeric_cols:
        g0 = df.loc[df['label'] == 0, col].dropna()
        g1 = df.loc[df['label'] == 1, col].dropna()
        if len(g0) >= 3 and len(g1) >= 3:
            pooled_std = np.sqrt((g0.std()**2 + g1.std()**2) / 2)
            if pooled_std > 0:
                d = (g1.mean() - g0.mean()) / pooled_std
            else:
                d = 0.0
            effect = 'Large' if abs(d) > 0.8 else 'Medium' if abs(d) > 0.5 else 'Small' if abs(d) > 0.2 else 'Negligible'
            log(f"  {col:<30} {d:>10.4f} {effect:>12}")
            effect_sizes.append({'feature': col, 'cohens_d': d, 'abs_d': abs(d), 'effect': effect})

    es_df = pd.DataFrame(effect_sizes).sort_values('abs_d', ascending=False)
    es_df.to_csv(os.path.join(EDA_DIR, 'effect_sizes.csv'), index=False)

    log(f"\n  Top discriminative features (|d| > 0.2):")
    top = es_df[es_df['abs_d'] > 0.2]
    if len(top) > 0:
        for _, row in top.iterrows():
            log(f"    ★ {row['feature']}: d={row['cohens_d']:.4f} ({row['effect']})")
    else:
        log("    ⚠ No features with meaningful effect size! Classes may overlap heavily.")

    # Scatter of top 2 features
    if len(es_df) >= 2:
        top2 = es_df.head(2)['feature'].tolist()
        fig, ax = plt.subplots(figsize=(8, 6))
        for label_val in [0, 1]:
            mask = df['label'] == label_val
            ax.scatter(df.loc[mask, top2[0]], df.loc[mask, top2[1]],
                      c=COLORS[str(label_val)], label=f'Label {label_val}',
                      alpha=0.6, s=60, edgecolors='white', linewidth=0.5)
        ax.set_xlabel(top2[0])
        ax.set_ylabel(top2[1])
        ax.set_title(f'Top 2 Discriminative Features', fontsize=14, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, '08_class_separability.png'), dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================
# 10. IMPLICATIONS FOR MODELING
# ============================================================
def section_implications(df):
    log("\n" + "=" * 70)
    log("10. EDA FINDINGS — IMPLICATIONS FOR IMPLEMENTATION PLAN")
    log("=" * 70)

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ['anio', 'label']]

    # 1. Sample size
    log(f"\n  [1] SAMPLE SIZE: {len(df)} observations")
    log(f"      After removing 2016 (for lags): ~{len(df) - df['unidad'].nunique()} observations")
    log(f"      →  Confirms: simple models should be prioritized")

    # 2. Missing values severity
    miss_pct = df[numeric_cols].isnull().sum().sum() / (len(df) * len(numeric_cols)) * 100
    log(f"\n  [2] MISSING VALUES: {miss_pct:.1f}% overall")
    log(f"      →  Confirms: imputation needed, missing indicators useful")

    # 3. Class balance
    prev = df['label'].mean()
    log(f"\n  [3] CLASS BALANCE: prevalence={prev:.3f}")
    if 0.3 <= prev <= 0.7:
        log(f"      →  Classes reasonably balanced, no SMOTE needed")
    else:
        log(f"      →  Consider class weighting or oversampling")

    # 4. Feature correlations
    corr_with_label = df[numeric_cols + ['label']].corr()['label'].drop('label').abs().sort_values(ascending=False)
    n_correlated = (corr_with_label > 0.2).sum()
    log(f"\n  [4] FEATURES CORRELATED WITH LABEL (|r|>0.2): {n_correlated}/{len(numeric_cols)}")
    if n_correlated < 3:
        log(f"      ⚠ Few features have linear correlation with label")
        log(f"      →  Non-linear models (trees) may help capture interactions")
    else:
        log(f"      →  Linear models may work well as baseline")

    # 5. Inter-feature correlation
    feat_corr = df[numeric_cols].corr()
    high_corr_pairs = []
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i+1:]:
            if abs(feat_corr.loc[c1, c2]) > 0.7:
                high_corr_pairs.append((c1, c2, feat_corr.loc[c1, c2]))
    log(f"\n  [5] HIGH INTER-FEATURE CORRELATIONS (|r|>0.7): {len(high_corr_pairs)} pairs")
    if high_corr_pairs:
        log(f"      →  Confirms: ElasticNet (L1) can help with multicollinearity")
        for c1, c2, r in high_corr_pairs:
            log(f"         {c1} ↔ {c2}: r={r:.3f}")

    # 6. Scale differences
    scales = {}
    for col in numeric_cols:
        data = df[col].dropna()
        scales[col] = max(abs(data.min()), abs(data.max()))
    max_scale = max(scales.values())
    min_scale = min(v for v in scales.values() if v > 0)
    log(f"\n  [6] SCALE RANGE: {min_scale:.2f} to {max_scale:.2f} (ratio: {max_scale/min_scale:.0f}x)")
    log(f"      →  Confirms: StandardScaler essential for linear models")

    # 7. Temporal patterns
    log(f"\n  [7] TEMPORAL STRUCTURE:")
    log(f"      Panel: {df['unidad'].nunique()} units × {df['anio'].nunique()} years")
    log(f"      →  Confirms: LOYO validation is correct approach")

    log(f"\n" + "=" * 70)
    log(f"EDA COMPLETE — All outputs saved to {EDA_DIR}")
    log(f"=" * 70)


# ============================================================
# 11. 3D VISUALIZATIONS
# ============================================================
def section_3d(df):
    log("\n" + "=" * 70)
    log("11. 3D VISUALIZATIONS")
    log("=" * 70)

    # --- Plot 1: 3D scatter of top 3 discriminative features ---
    top3 = ['gp_ratio', 'cfo', 'dias_efectivo']
    df_plot = df.dropna(subset=top3).copy()

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    for label_val in [0, 1]:
        mask = df_plot['label'] == label_val
        color = '#4CAF50' if label_val == 0 else '#F44336'
        label_text = 'Sano (0)' if label_val == 0 else 'Riesgo (1)'
        ax.scatter(
            df_plot.loc[mask, top3[0]],
            df_plot.loc[mask, top3[1]] / 1e9,  # Scale CFO to billions
            df_plot.loc[mask, top3[2]],
            c=color, label=label_text, s=80, alpha=0.75,
            edgecolors='white', linewidth=0.5, depthshade=True
        )

    ax.set_xlabel('gp_ratio', fontsize=11, labelpad=10)
    ax.set_ylabel('CFO (miles de millones COP)', fontsize=11, labelpad=10)
    ax.set_zlabel('dias_efectivo', fontsize=11, labelpad=10)
    ax.set_title('3D Class Separability — Top 3 Features', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left')
    ax.view_init(elev=25, azim=135)

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '09_3d_top3_features.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log("  ✓ 09_3d_top3_features.png — 3D scatter of gp_ratio × CFO × dias_efectivo")

    # --- Plot 2: 3D temporal mapping — feature evolution over unit × year ---
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')

    units = sorted(df['unidad'].unique())
    unit_to_num = {u: i for i, u in enumerate(units)}
    df_t = df.dropna(subset=['gp_ratio']).copy()
    df_t['unit_num'] = df_t['unidad'].map(unit_to_num)

    for label_val in [0, 1]:
        mask = df_t['label'] == label_val
        color = '#4CAF50' if label_val == 0 else '#F44336'
        label_text = 'Sano (0)' if label_val == 0 else 'Riesgo (1)'
        ax.scatter(
            df_t.loc[mask, 'anio'],
            df_t.loc[mask, 'unit_num'],
            df_t.loc[mask, 'gp_ratio'],
            c=color, label=label_text, s=80, alpha=0.75,
            edgecolors='white', linewidth=0.5, depthshade=True
        )

    # Connect points within each unit with lines
    for unit in units:
        unit_data = df_t[df_t['unidad'] == unit].sort_values('anio')
        if len(unit_data) >= 2:
            ax.plot(
                unit_data['anio'], unit_data['unit_num'], unit_data['gp_ratio'],
                color='#90A4AE', alpha=0.4, linewidth=1
            )

    ax.set_xlabel('Año', fontsize=11, labelpad=10)
    ax.set_ylabel('Unidad', fontsize=11, labelpad=10)
    ax.set_zlabel('gp_ratio', fontsize=11, labelpad=10)
    ax.set_yticks(range(len(units)))
    ax.set_yticklabels([u[:10] for u in units], fontsize=7, rotation=-15)
    ax.set_title('3D Temporal Mapping — gp_ratio by Unit & Year', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '10_3d_temporal_gp_ratio.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log("  ✓ 10_3d_temporal_gp_ratio.png — 3D temporal mapping (year × unit × gp_ratio)")

    # --- Plot 3: 3D scatter — cfo × liquidez × endeudamiento colored by label ---
    feats_3 = ['liquidez', 'endeudamiento', 'cfo']
    df_plot3 = df.dropna(subset=feats_3).copy()

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    for label_val in [0, 1]:
        mask = df_plot3['label'] == label_val
        color = '#4CAF50' if label_val == 0 else '#F44336'
        label_text = 'Sano (0)' if label_val == 0 else 'Riesgo (1)'
        ax.scatter(
            df_plot3.loc[mask, 'liquidez'],
            df_plot3.loc[mask, 'endeudamiento'],
            df_plot3.loc[mask, 'cfo'] / 1e9,
            c=color, label=label_text, s=80, alpha=0.75,
            edgecolors='white', linewidth=0.5, depthshade=True
        )

    ax.set_xlabel('Liquidez', fontsize=11, labelpad=10)
    ax.set_ylabel('Endeudamiento', fontsize=11, labelpad=10)
    ax.set_zlabel('CFO (miles de millones COP)', fontsize=11, labelpad=10)
    ax.set_title('3D Financial Risk Space — Liquidez × Endeudamiento × CFO',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left')
    ax.view_init(elev=20, azim=225)

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '11_3d_financial_risk_space.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log("  ✓ 11_3d_financial_risk_space.png — 3D scatter of liquidez × endeudamiento × cfo")

    # --- Plot 4: 3D surface — risk density over gp_ratio × dias_efectivo ---
    feat_x = 'gp_ratio'
    feat_y = 'dias_efectivo'
    df_surf = df.dropna(subset=[feat_x, feat_y]).copy()

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Create grid
    x_range = np.linspace(df_surf[feat_x].min() - 0.02, df_surf[feat_x].max() + 0.02, 40)
    y_range = np.linspace(df_surf[feat_y].min() - 5, df_surf[feat_y].max() + 5, 40)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    # Estimate risk probability using gaussian kernel
    from scipy.stats import gaussian_kde
    risk_data = df_surf[df_surf['label'] == 1]
    safe_data = df_surf[df_surf['label'] == 0]

    positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])

    try:
        kde_risk = gaussian_kde(risk_data[[feat_x, feat_y]].dropna().T.values, bw_method=0.4)
        kde_safe = gaussian_kde(safe_data[[feat_x, feat_y]].dropna().T.values, bw_method=0.4)

        density_risk = kde_risk(positions).reshape(X_grid.shape)
        density_safe = kde_safe(positions).reshape(X_grid.shape)

        # P(risk) = density_risk / (density_risk + density_safe)
        total_density = density_risk + density_safe
        total_density = np.where(total_density > 1e-10, total_density, 1e-10)
        risk_prob = density_risk / total_density

        # Plot surface
        from matplotlib.colors import LinearSegmentedColormap
        cmap_risk = LinearSegmentedColormap.from_list('risk', ['#4CAF50', '#FFEB3B', '#F44336'])
        surf = ax.plot_surface(X_grid, Y_grid, risk_prob, cmap=cmap_risk,
                               alpha=0.7, edgecolor='none', antialiased=True)

        # Overlay actual points
        for label_val in [0, 1]:
            mask = df_surf['label'] == label_val
            color = '#2E7D32' if label_val == 0 else '#C62828'
            marker = 'o' if label_val == 0 else '^'
            z_vals = np.full(mask.sum(), -0.05 if label_val == 0 else 1.05)
            ax.scatter(
                df_surf.loc[mask, feat_x], df_surf.loc[mask, feat_y], z_vals,
                c=color, marker=marker, s=50, alpha=0.8, edgecolors='white', linewidth=0.5,
                label=f"{'Sano' if label_val == 0 else 'Riesgo'} ({label_val})"
            )

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='P(Risk)', pad=0.1)
        ax.set_xlabel(feat_x, fontsize=11, labelpad=10)
        ax.set_ylabel(feat_y, fontsize=11, labelpad=10)
        ax.set_zlabel('P(Risk)', fontsize=11, labelpad=10)
        ax.set_title('3D Risk Probability Surface — gp_ratio × dias_efectivo',
                     fontsize=13, fontweight='bold', pad=20)
        ax.legend(fontsize=10, loc='upper right')
        ax.view_init(elev=30, azim=135)

    except Exception as e:
        log(f"  ⚠ Risk surface plot failed: {e}")
        ax.text(0.5, 0.5, 0.5, f'Surface failed: {e}', fontsize=12, ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '12_3d_risk_surface.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log("  ✓ 12_3d_risk_surface.png — 3D risk probability surface (KDE-based)")


# ============================================================
# 12. UMAP VISUALIZATIONS
# ============================================================
def section_umap(df):
    log("\n" + "=" * 70)
    log("12. UMAP DIMENSIONALITY REDUCTION")
    log("=" * 70)

    try:
        import umap
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        log("  ⚠ umap-learn not installed. Skipping UMAP.")
        return

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ['anio', 'label']]

    # Prepare data — drop rows with any NaN in features
    df_umap = df.dropna(subset=numeric_cols).copy()
    X = df_umap[numeric_cols].values
    y_label = df_umap['label'].values
    y_year = df_umap['anio'].values
    y_unit = df_umap['unidad'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    log(f"  UMAP input: {X_scaled.shape[0]} samples × {X_scaled.shape[1]} features")

    # --- 2D UMAP ---
    reducer_2d = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.3,
                            metric='euclidean', random_state=42)
    embedding_2d = reducer_2d.fit_transform(X_scaled)
    log(f"  2D UMAP embedding computed.")

    # --- 3D UMAP ---
    reducer_3d = umap.UMAP(n_components=3, n_neighbors=10, min_dist=0.3,
                            metric='euclidean', random_state=42)
    embedding_3d = reducer_3d.fit_transform(X_scaled)
    log(f"  3D UMAP embedding computed.")

    # --- Plot 1: UMAP colored by label ---
    fig, ax = plt.subplots(figsize=(10, 8))
    for label_val in [0, 1]:
        mask = y_label == label_val
        color = '#4CAF50' if label_val == 0 else '#F44336'
        label_text = 'Sano (0)' if label_val == 0 else 'Riesgo (1)'
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   c=color, label=label_text, s=90, alpha=0.75,
                   edgecolors='white', linewidth=0.8)
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('UMAP — Colored by Risk Label', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '13_umap_by_label.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log("  ✓ 13_umap_by_label.png")

    # --- Plot 2: UMAP colored by year ---
    fig, ax = plt.subplots(figsize=(10, 8))
    years = sorted(df_umap['anio'].unique())
    cmap_years = plt.cm.viridis(np.linspace(0.1, 0.95, len(years)))
    for i, year in enumerate(years):
        mask = y_year == year
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   c=[cmap_years[i]], label=str(year), s=80, alpha=0.75,
                   edgecolors='white', linewidth=0.8)
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('UMAP — Colored by Year', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '14_umap_by_year.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log("  ✓ 14_umap_by_year.png")

    # --- Plot 3: UMAP colored by unit ---
    fig, ax = plt.subplots(figsize=(11, 8))
    units = sorted(df_umap['unidad'].unique())
    cmap_units = plt.cm.tab10(np.linspace(0, 1, len(units)))
    for i, unit in enumerate(units):
        mask = y_unit == unit
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   c=[cmap_units[i]], label=unit[:18], s=80, alpha=0.75,
                   edgecolors='white', linewidth=0.8, marker='o')
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('UMAP — Colored by Academic Unit', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '15_umap_by_unit.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log("  ✓ 15_umap_by_unit.png")

    # --- Plot 4: 3D UMAP colored by label ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    for label_val in [0, 1]:
        mask = y_label == label_val
        color = '#4CAF50' if label_val == 0 else '#F44336'
        label_text = 'Sano (0)' if label_val == 0 else 'Riesgo (1)'
        ax.scatter(embedding_3d[mask, 0], embedding_3d[mask, 1], embedding_3d[mask, 2],
                   c=color, label=label_text, s=80, alpha=0.75,
                   edgecolors='white', linewidth=0.5, depthshade=True)
    ax.set_xlabel('UMAP 1', fontsize=11, labelpad=8)
    ax.set_ylabel('UMAP 2', fontsize=11, labelpad=8)
    ax.set_zlabel('UMAP 3', fontsize=11, labelpad=8)
    ax.set_title('3D UMAP — Colored by Risk Label', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '16_umap_3d_label.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log("  ✓ 16_umap_3d_label.png")

    # --- Plot 5: Combined UMAP dashboard ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Top-left: by label
    ax = axes[0, 0]
    for label_val in [0, 1]:
        mask = y_label == label_val
        color = '#4CAF50' if label_val == 0 else '#F44336'
        label_text = 'Sano (0)' if label_val == 0 else 'Riesgo (1)'
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   c=color, label=label_text, s=70, alpha=0.75,
                   edgecolors='white', linewidth=0.6)
    ax.set_title('By Label', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    # Top-right: by year
    ax = axes[0, 1]
    for i, year in enumerate(years):
        mask = y_year == year
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   c=[cmap_years[i]], label=str(year), s=70, alpha=0.75,
                   edgecolors='white', linewidth=0.6)
    ax.set_title('By Year', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)

    # Bottom-left: by unit
    ax = axes[1, 0]
    for i, unit in enumerate(units):
        mask = y_unit == unit
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   c=[cmap_units[i]], label=unit[:15], s=70, alpha=0.75,
                   edgecolors='white', linewidth=0.6)
    ax.set_title('By Academic Unit', fontsize=13, fontweight='bold')
    ax.legend(fontsize=7, loc='best')

    # Bottom-right: annotated with unit+year
    ax = axes[1, 1]
    ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
               c=['#F44336' if l == 1 else '#4CAF50' for l in y_label],
               s=30, alpha=0.4, edgecolors='none')
    for idx in range(len(df_umap)):
        row = df_umap.iloc[idx]
        ax.annotate(f"{int(row['anio'])%100}-{row['unidad'][:3]}",
                    (embedding_2d[idx, 0], embedding_2d[idx, 1]),
                    fontsize=5, alpha=0.7, ha='center')
    ax.set_title('Annotated (YY-Unit)', fontsize=13, fontweight='bold')

    for ax in axes.ravel():
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)

    plt.suptitle('FIRE-UdeA — UMAP Dashboard', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '17_umap_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log("  ✓ 17_umap_dashboard.png — Combined UMAP dashboard")


# ============================================================
# 13. PCA VISUALIZATIONS
# ============================================================
def section_pca(df):
    log("\n" + "=" * 70)
    log("13. PCA DIMENSIONALITY REDUCTION & LOADINGS")
    log("=" * 70)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ['anio', 'label']]

    # Prepare data
    df_pca = df.dropna(subset=numeric_cols).copy()
    X = df_pca[numeric_cols].values
    y_label = df_pca['label'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA
    pca = PCA(random_state=42)
    embedding_pca = pca.fit_transform(X_scaled)
    
    log(f"  PCA fitted: {pca.n_components_} components")

    # --- Plot 1: 2D PCA colored by label ---
    fig, ax = plt.subplots(figsize=(10, 8))
    for label_val in [0, 1]:
        mask = y_label == label_val
        color = '#4CAF50' if label_val == 0 else '#F44336'
        label_text = 'Sano (0)' if label_val == 0 else 'Riesgo (1)'
        ax.scatter(embedding_pca[mask, 0], embedding_pca[mask, 1],
                   c=color, label=label_text, s=90, alpha=0.75,
                   edgecolors='white', linewidth=0.8)
    
    ev_1 = pca.explained_variance_ratio_[0] * 100
    ev_2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f'PC1 ({ev_1:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({ev_2:.1f}% variance)', fontsize=12)
    ax.set_title('PCA — First two Principal Components', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '18_pca_2d.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log("  ✓ 18_pca_2d.png")

    # --- Plot 2: Explained Variance Ratio ---
    fig, ax = plt.subplots(figsize=(10, 6))
    var_ratio = pca.explained_variance_ratio_ * 100
    cum_var_ratio = np.cumsum(var_ratio)
    
    x_ticks = np.arange(1, len(var_ratio) + 1)
    ax.bar(x_ticks, var_ratio, alpha=0.7, color='#2196F3',
           edgecolor='white', label='Individual')
    ax.plot(x_ticks, cum_var_ratio, 'r-o', linewidth=2, label='Cumulative')
    
    ax.axhline(80, color='gray', linestyle='--', alpha=0.5, label='80% variance')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance (%)', fontsize=12)
    ax.set_title('PCA — Explained Variance', fontsize=14, fontweight='bold')
    ax.set_xticks(x_ticks)
    ax.legend(fontsize=11)
    
    for i, v in enumerate(var_ratio):
        ax.text(i + 1, v + 1, f'{v:.1f}%', ha='center', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '19_pca_variance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log("  ✓ 19_pca_variance.png")

    # --- Plot 3: PCA Loadings (Feature Importance for PC1 & PC2) ---
    fig, ax = plt.subplots(figsize=(12, 10))
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    ax.scatter(loadings[:, 0], loadings[:, 1], alpha=0)
    for i, feature in enumerate(numeric_cols):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                 color='#F44336', alpha=0.5, head_width=0.04, head_length=0.06)
        ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, feature,
                color='black', ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', alpha=0.3)
    ax.add_artist(circle)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel(f'PC1 ({ev_1:.1f}%) Correlation', fontsize=12)
    ax.set_ylabel(f'PC2 ({ev_2:.1f}%) Correlation', fontsize=12)
    ax.set_title('PCA Loadings Plot (Correlation Circle)', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '20_pca_loadings.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log("  ✓ 20_pca_loadings.png")


# ============================================================
# MAIN
# ============================================================
def main():
    df = pd.read_csv(DATA_PATH)

    section_overview(df)
    section_target(df)
    section_missing(df)
    section_distributions(df)
    section_correlations(df)
    section_temporal(df)
    section_outliers(df)
    section_scales(df)
    section_separability(df)
    section_3d(df)
    section_umap(df)
    section_pca(df)
    section_implications(df)

    # Save full report
    report_path = os.path.join(EDA_DIR, 'eda_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(REPORT_LINES))
    print(f"\n  Full report saved to: {report_path}")


if __name__ == '__main__':
    main()

