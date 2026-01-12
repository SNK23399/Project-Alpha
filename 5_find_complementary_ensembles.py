"""
Find Complementary Feature Ensembles

This script identifies sets of features that:
1. Each have strong individual predictive power
2. Have low correlation with each other (make different mistakes)
3. Combined, form robust ensemble predictors

Strategy:
- Start with top N features by robustness
- Calculate pairwise correlations between their predictions
- Use greedy selection to build ensemble of low-correlation features
- Validate ensemble performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

print("=" * 80)
print("FINDING COMPLEMENTARY FEATURE ENSEMBLES")
print("=" * 80)

# Configuration
FORWARD_PERIOD = 63  # Use 63-day horizon (best balance of signal strength and frequency)
TOP_N_CANDIDATES = 200  # Consider top 200 features
MIN_ROBUSTNESS = 0.008  # Minimum robustness score
MAX_CORRELATION = 0.50  # Maximum correlation between ensemble members

# Load robustness results
robustness_file = Path("data/feature_analysis/robustness_comparison.parquet")
df_robustness = pd.read_parquet(robustness_file)

# Filter for strong features
strong_features = df_robustness[
    (df_robustness['robustness_score'] > MIN_ROBUSTNESS) &
    (df_robustness['corr_consistent'] == True)
].nlargest(TOP_N_CANDIDATES, 'robustness_score')

print(f"\nAnalyzing top {len(strong_features)} features")
print(f"Min robustness: {strong_features['robustness_score'].min():.6f}")
print(f"Max robustness: {strong_features['robustness_score'].max():.6f}")

# Load actual feature values to calculate correlations
print("\nLoading feature data...")
feature_data = {}

for idx, row in tqdm(strong_features.iterrows(), total=len(strong_features), desc="Loading features"):
    feature_name = row['feature']

    # Parse feature name: signal_filter__indicator
    parts = feature_name.rsplit('__', 1)
    if len(parts) != 2:
        continue

    signal_filter, indicator = parts

    # Load feature file
    feature_file = Path(f"data/features/{signal_filter}.parquet")
    if not feature_file.exists():
        continue

    try:
        df_feature = pd.read_parquet(feature_file)

        # Get specific indicator column
        col_name = f"{signal_filter}__{indicator}"
        if col_name in df_feature.columns:
            feature_data[feature_name] = df_feature[col_name]
    except Exception as e:
        print(f"  Warning: Could not load {feature_name}: {e}")
        continue

print(f"\nSuccessfully loaded {len(feature_data)} features")

# Convert to DataFrame
df_features = pd.DataFrame(feature_data)

# Remove features with too many NaNs
min_valid_pct = 0.80
valid_counts = df_features.notna().sum() / len(df_features)
df_features = df_features.loc[:, valid_counts >= min_valid_pct]

print(f"After filtering NaNs: {len(df_features.columns)} features")

# Calculate pairwise correlations
print("\nCalculating pairwise feature correlations...")
feature_corr_matrix = df_features.corr()

# Save correlation matrix
corr_output = Path("data/feature_analysis/feature_correlation_matrix.parquet")
feature_corr_matrix.to_parquet(corr_output)
print(f"Saved correlation matrix to {corr_output}")

# Analyze correlation structure
print("\n" + "=" * 80)
print("CORRELATION STRUCTURE ANALYSIS")
print("=" * 80)

# Get upper triangle (avoid double-counting)
mask = np.triu(np.ones_like(feature_corr_matrix), k=1).astype(bool)
upper_triangle = feature_corr_matrix.where(mask)
correlations = upper_triangle.stack()

print(f"\nPairwise correlation statistics:")
print(f"  Mean absolute correlation: {abs(correlations).mean():.4f}")
print(f"  Median absolute correlation: {abs(correlations).median():.4f}")
print(f"  Max correlation: {correlations.max():.4f}")
print(f"  Min correlation: {correlations.min():.4f}")

print(f"\nDistribution:")
print(f"  |corr| < 0.10 (very low): {(abs(correlations) < 0.10).sum()} pairs ({(abs(correlations) < 0.10).mean():.1%})")
print(f"  |corr| < 0.30 (low): {(abs(correlations) < 0.30).sum()} pairs ({(abs(correlations) < 0.30).mean():.1%})")
print(f"  |corr| < 0.50 (moderate): {(abs(correlations) < 0.50).sum()} pairs ({(abs(correlations) < 0.50).mean():.1%})")
print(f"  |corr| > 0.70 (high): {(abs(correlations) > 0.70).sum()} pairs ({(abs(correlations) > 0.70).mean():.1%})")
print(f"  |corr| > 0.90 (very high): {(abs(correlations) > 0.90).sum()} pairs ({(abs(correlations) > 0.90).mean():.1%})")

# Find highly correlated pairs (redundant features)
high_corr_pairs = correlations[abs(correlations) > 0.90].sort_values(ascending=False)
print(f"\nHighly correlated pairs (|r| > 0.90): {len(high_corr_pairs)}")
if len(high_corr_pairs) > 0:
    print("\nTop 10 redundant feature pairs:")
    for (feat1, feat2), corr in high_corr_pairs.head(10).items():
        print(f"  {feat1} <-> {feat2}: {corr:+.4f}")

# Greedy ensemble selection
print("\n" + "=" * 80)
print("GREEDY ENSEMBLE SELECTION")
print("=" * 80)

def greedy_ensemble_selection(robustness_df, corr_matrix, max_features=20, max_corr=0.50):
    """
    Greedy algorithm to select diverse ensemble:
    1. Start with most robust feature
    2. Add features that have low correlation with selected set
    3. Continue until max_features or no suitable candidates
    """
    # Sort by robustness
    candidates = robustness_df.sort_values('robustness_score', ascending=False)

    # Filter to features we have correlation data for
    candidates = candidates[candidates['feature'].isin(corr_matrix.columns)]

    selected = []
    selected_names = []

    for idx, row in candidates.iterrows():
        feature = row['feature']

        if len(selected) == 0:
            # First feature: pick most robust
            selected.append(row)
            selected_names.append(feature)
            print(f"\n1. {feature}")
            print(f"   Robustness: {row['robustness_score']:.6f}")
            print(f"   Avg LS-Alpha: {row['avg_ls_alpha']:.5f}")
            continue

        # Check correlation with already selected features
        correlations_with_selected = []
        for sel_feature in selected_names:
            if feature in corr_matrix.index and sel_feature in corr_matrix.columns:
                corr_val = abs(corr_matrix.loc[feature, sel_feature])
                correlations_with_selected.append(corr_val)

        if not correlations_with_selected:
            continue

        max_corr_with_selected = max(correlations_with_selected)
        avg_corr_with_selected = np.mean(correlations_with_selected)

        # Accept if correlation is below threshold
        if max_corr_with_selected < max_corr:
            selected.append(row)
            selected_names.append(feature)
            print(f"\n{len(selected)}. {feature}")
            print(f"   Robustness: {row['robustness_score']:.6f}")
            print(f"   Avg LS-Alpha: {row['avg_ls_alpha']:.5f}")
            print(f"   Max corr with ensemble: {max_corr_with_selected:.4f}")
            print(f"   Avg corr with ensemble: {avg_corr_with_selected:.4f}")

            if len(selected) >= max_features:
                break

    return pd.DataFrame(selected)

# Build ensemble
print(f"\nBuilding ensemble with max correlation {MAX_CORRELATION}...")
ensemble = greedy_ensemble_selection(
    strong_features,
    feature_corr_matrix,
    max_features=20,
    max_corr=MAX_CORRELATION
)

print("\n" + "=" * 80)
print(f"FINAL ENSEMBLE: {len(ensemble)} FEATURES")
print("=" * 80)

# Analyze ensemble diversity
print("\nEnsemble composition by signal type:")
ensemble['signal_type'] = ensemble['feature'].str.split('__').str[0]
signal_counts = ensemble['signal_type'].value_counts()
for signal_type, count in signal_counts.items():
    print(f"  {signal_type}: {count}")

# Ensemble performance summary
print("\nEnsemble performance summary:")
print(f"  Average robustness: {ensemble['robustness_score'].mean():.6f}")
print(f"  Average LS-alpha (21d): {ensemble['ls_alpha_21d'].mean():.5f}")
print(f"  Average LS-alpha (63d): {ensemble['ls_alpha_63d'].mean():.5f}")
print(f"  Average LS-alpha (126d): {ensemble['ls_alpha_126d'].mean():.5f}")
print(f"  Average correlation (21d): {ensemble['corr_21d'].mean():+.4f}")
print(f"  Average correlation (63d): {ensemble['corr_63d'].mean():+.4f}")
print(f"  Average correlation (126d): {ensemble['corr_126d'].mean():+.4f}")

# Calculate average pairwise correlation within ensemble
ensemble_features = ensemble['feature'].tolist()
ensemble_corrs = []
for i, feat1 in enumerate(ensemble_features):
    for feat2 in ensemble_features[i+1:]:
        if feat1 in feature_corr_matrix.index and feat2 in feature_corr_matrix.columns:
            ensemble_corrs.append(abs(feature_corr_matrix.loc[feat1, feat2]))

print(f"\nEnsemble diversity:")
print(f"  Avg pairwise |correlation|: {np.mean(ensemble_corrs):.4f}")
print(f"  Max pairwise |correlation|: {np.max(ensemble_corrs):.4f}")
print(f"  Min pairwise |correlation|: {np.min(ensemble_corrs):.4f}")

# Save ensemble
output_file = Path("data/feature_analysis/complementary_ensemble.parquet")
ensemble.to_parquet(output_file)

output_csv = Path("data/feature_analysis/complementary_ensemble.csv")
ensemble.to_csv(output_csv, index=False)

print(f"\nEnsemble saved to:")
print(f"  - {output_file}")
print(f"  - {output_csv}")

# Create alternative ensembles with different strategies
print("\n" + "=" * 80)
print("ALTERNATIVE ENSEMBLE STRATEGIES")
print("=" * 80)

# Strategy 2: Maximize diversity (very low correlation)
print("\nStrategy 2: Maximum Diversity (max_corr=0.30)...")
ensemble_diverse = greedy_ensemble_selection(
    strong_features,
    feature_corr_matrix,
    max_features=15,
    max_corr=0.30
)

diverse_corrs = []
diverse_features = ensemble_diverse['feature'].tolist()
for i, feat1 in enumerate(diverse_features):
    for feat2 in diverse_features[i+1:]:
        if feat1 in feature_corr_matrix.index and feat2 in feature_corr_matrix.columns:
            diverse_corrs.append(abs(feature_corr_matrix.loc[feat1, feat2]))

print(f"  Selected {len(ensemble_diverse)} features")
print(f"  Avg pairwise |correlation|: {np.mean(diverse_corrs):.4f}")
print(f"  Avg robustness: {ensemble_diverse['robustness_score'].mean():.6f}")

ensemble_diverse.to_parquet(Path("data/feature_analysis/diverse_ensemble.parquet"))
ensemble_diverse.to_csv(Path("data/feature_analysis/diverse_ensemble.csv"), index=False)

# Strategy 3: Signal-type balanced (ensure representation)
print("\nStrategy 3: Signal-Type Balanced...")

# Get top features by signal type
signal_types = strong_features['feature'].str.split('__').str[0].unique()
balanced_ensemble = []

features_per_signal = 3  # Take top 3 from each signal type

for signal_type in signal_types[:7]:  # Top 7 signal types
    signal_features = strong_features[
        strong_features['feature'].str.startswith(signal_type + '__')
    ].head(features_per_signal)

    balanced_ensemble.append(signal_features)

ensemble_balanced = pd.concat(balanced_ensemble)
print(f"  Selected {len(ensemble_balanced)} features from {len(signal_types[:7])} signal types")
print(f"  Avg robustness: {ensemble_balanced['robustness_score'].mean():.6f}")

ensemble_balanced.to_parquet(Path("data/feature_analysis/balanced_ensemble.parquet"))
ensemble_balanced.to_csv(Path("data/feature_analysis/balanced_ensemble.csv"), index=False)

# Summary comparison
print("\n" + "=" * 80)
print("ENSEMBLE COMPARISON")
print("=" * 80)

ensembles = {
    'Complementary (r<0.50)': ensemble,
    'Diverse (r<0.30)': ensemble_diverse,
    'Balanced (by signal)': ensemble_balanced
}

comparison = []
for name, ens in ensembles.items():
    comparison.append({
        'Ensemble': name,
        'N Features': len(ens),
        'Avg Robustness': ens['robustness_score'].mean(),
        'Avg LS-Alpha (63d)': ens['ls_alpha_63d'].mean(),
        'Avg Corr (63d)': ens['corr_63d'].mean(),
    })

df_comparison = pd.DataFrame(comparison)
print("\n" + df_comparison.to_string(index=False))

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("\nUse 'Complementary' ensemble for best balance of:")
print("  - Individual feature strength (high robustness)")
print("  - Diversity (low pairwise correlation)")
print("  - Ensemble size (20 features = manageable)")
print("\nThis ensemble will make different mistakes on different market regimes,")
print("leading to more robust combined predictions.")
print("=" * 80)
