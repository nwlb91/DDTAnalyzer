"""
Analyze which r/ssbm DDT commenters have the biggest effect on comment count.

Statistical methodology:
1. Build a baseline model predicting DDT comment count from confounders:
   - Day of week (categorical)
   - Long-term time trend (linear + quadratic)
   - Month-of-year (seasonal effects / tournament schedule)
   - Rolling activity level (captures community-wide surges from events)
2. Compute residuals (actual - predicted) to get "unexplained" comment variation.
3. For each user, compare mean residual when they're present vs absent.
   This gives us their confound-adjusted marginal effect.
4. Statistical significance via Welch's t-test + Bonferroni correction.
5. Also run a multi-user Ridge regression as a robustness check, which
   controls for user-user correlations (e.g. friend groups showing up together).
6. Final ranking blends both approaches and assigns tier labels.
"""

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_data():
    """Load scraped posts and comments from disk into a DataFrame."""
    posts_file = os.path.join(DATA_DIR, "ddt_posts.json")
    comments_dir = os.path.join(DATA_DIR, "comments")

    if not os.path.exists(posts_file):
        print("Error: No scraped data found. Run scrape.py first.")
        sys.exit(1)

    with open(posts_file) as f:
        posts = json.load(f)

    records = []
    for post in posts:
        pid = post["id"]
        comment_file = os.path.join(comments_dir, f"{pid}.json")
        if not os.path.exists(comment_file):
            continue

        with open(comment_file) as f:
            comments = json.load(f)

        dt = datetime.fromtimestamp(post["created_utc"], tz=timezone.utc)
        commenters = set()
        commenter_counts = Counter()
        for c in comments:
            author = c.get("author", "")
            if author:
                commenters.add(author)
                commenter_counts[author] += 1

        records.append({
            "post_id": pid,
            "title": post["title"],
            "date": dt.date(),
            "datetime": dt,
            "day_of_week": dt.strftime("%A"),
            "day_of_week_num": dt.weekday(),
            "month": dt.month,
            "year": dt.year,
            "total_comments": len(comments),
            "num_unique_commenters": len(commenters),
            "commenters": commenters,
            "commenter_counts": dict(commenter_counts),
        })

    df = pd.DataFrame(records)
    if df.empty:
        print("Error: No DDT data loaded.")
        sys.exit(1)

    df = df.sort_values("date").reset_index(drop=True)
    # Time index for trend (0 = first DDT, 1 = second, etc.)
    df["time_index"] = range(len(df))
    # Rolling mean of comment count (14-day window) to capture event-driven surges
    df["rolling_mean_14"] = (
        df["total_comments"]
        .rolling(window=14, min_periods=1, center=True)
        .mean()
    )

    # Compute how many days each DDT was active (gap to next DDT).
    # A DDT left up for multiple days accumulates extra comments, which
    # would otherwise be mistaken for user-driven engagement.
    dates = pd.to_datetime(df["date"])
    gaps = dates.diff(periods=-1).abs().dt.days
    df["days_active"] = gaps.fillna(1).astype(int).clip(lower=1)

    n_multiday = (df["days_active"] > 1).sum()
    if n_multiday > 0:
        print(f"  Note: {n_multiday} DDTs were left up for multiple days (controlled for in model)")

    print(f"Loaded {len(df)} DDTs spanning {df['date'].min()} to {df['date'].max()}")
    print(f"  Total comments: {df['total_comments'].sum():,}")
    print(f"  Mean comments/DDT: {df['total_comments'].mean():.1f}")
    print(f"  Median comments/DDT: {df['total_comments'].median():.1f}")

    return df


def build_confound_features(df):
    """
    Build confound feature matrix (day of week, time trend, month, rolling activity).

    Returns (X_confounds, feature_names).
    """
    features = {}

    # Day of week dummies (drop Monday as reference)
    for day in ["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        features[f"dow_{day}"] = (df["day_of_week"] == day).astype(float)

    # Time trend (linear + quadratic, normalized)
    t = df["time_index"].values.astype(float)
    t_norm = t / max(t.max(), 1)
    features["trend_linear"] = t_norm
    features["trend_quadratic"] = t_norm ** 2

    # Month dummies (drop January as reference) - captures tournament seasons
    for m in range(2, 13):
        features[f"month_{m}"] = (df["month"] == m).astype(float)

    # Rolling activity (leave-one-out: exclude current day to avoid leakage)
    # We use a lagged version: average of previous 7 days
    rolling_lag = df["total_comments"].rolling(window=7, min_periods=1).mean().shift(1)
    rolling_lag = rolling_lag.fillna(df["total_comments"].mean())
    features["rolling_activity_7d"] = rolling_lag.values

    # Days active: how many days a DDT was pinned/left up.
    # A DDT stuck up for 2-3 days naturally accumulates more comments.
    features["days_active"] = df["days_active"].values.astype(float)

    X = pd.DataFrame(features, index=df.index)
    return X


def fit_baseline_model(df, X_confounds):
    """
    Fit OLS baseline model: total_comments ~ confounders.

    Returns the residuals (actual - predicted), which represent
    "unexplained" variation in comment count.
    """
    y = df["total_comments"].values.astype(float)
    X = sm.add_constant(X_confounds.values.astype(float))

    model = sm.OLS(y, X).fit()

    print(f"\n--- Baseline Model (confounders only) ---")
    print(f"  R² = {model.rsquared:.3f}  (confounders explain {model.rsquared * 100:.1f}% of variance)")
    print(f"  Adjusted R² = {model.rsquared_adj:.3f}")

    # Show day-of-week effects
    print(f"\n  Day-of-week effects (vs Monday):")
    dow_features = [c for c in X_confounds.columns if c.startswith("dow_")]
    for i, feat in enumerate(dow_features):
        coef_idx = i + 1  # +1 for constant
        day_name = feat.replace("dow_", "")
        coef = model.params[coef_idx]
        pval = model.pvalues[coef_idx]
        sig = "*" if pval < 0.05 else ""
        print(f"    {day_name:>12s}: {coef:+.1f} comments  (p={pval:.3f}) {sig}")

    residuals = model.resid
    return residuals, model


def analyze_user_effects_residual(df, residuals, min_ddts=5, min_comments=10):
    """
    Approach 1: Residual-based analysis.

    For each user, compare mean residual when they're present vs absent.
    A positive effect means DDTs with this user tend to have MORE comments
    than expected (after controlling for day-of-week, time trends, etc.).

    Args:
        df: DataFrame of DDT records
        residuals: Confound-adjusted residuals from baseline model
        min_ddts: Minimum number of DDTs a user must appear in
        min_comments: Minimum total comments across all DDTs

    Returns:
        DataFrame with user effect estimates
    """
    # Collect all users and their DDT presence
    all_users = Counter()
    user_total_comments = Counter()
    for _, row in df.iterrows():
        for user, count in row["commenter_counts"].items():
            all_users[user] += 1
            user_total_comments[user] += count

    # Filter users by minimums
    eligible_users = {
        u for u, n_ddts in all_users.items()
        if n_ddts >= min_ddts and user_total_comments[u] >= min_comments
    }
    print(f"\n--- Residual-Based User Effect Analysis ---")
    print(f"  {len(all_users)} total unique users")
    print(f"  {len(eligible_users)} users meet thresholds (>={min_ddts} DDTs, >={min_comments} comments)")

    results = []
    n_ddts = len(df)

    for user in eligible_users:
        present_mask = np.array([user in row["commenters"] for _, row in df.iterrows()])
        absent_mask = ~present_mask

        n_present = present_mask.sum()
        n_absent = absent_mask.sum()

        if n_present < 2 or n_absent < 2:
            continue

        resid_present = residuals[present_mask]
        resid_absent = residuals[absent_mask]

        mean_present = resid_present.mean()
        mean_absent = resid_absent.mean()
        effect = mean_present - mean_absent

        # Welch's t-test
        t_stat, p_value = stats.ttest_ind(resid_present, resid_absent, equal_var=False)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((n_present - 1) * resid_present.std() ** 2 + (n_absent - 1) * resid_absent.std() ** 2)
            / (n_present + n_absent - 2)
        )
        cohens_d = effect / pooled_std if pooled_std > 0 else 0

        # User's own comment contribution (average comments per DDT they're in)
        own_comments = np.mean([
            row["commenter_counts"].get(user, 0)
            for _, row in df.iterrows()
            if user in row["commenters"]
        ])

        results.append({
            "user": user,
            "n_ddts_present": int(n_present),
            "n_ddts_absent": int(n_absent),
            "presence_rate": n_present / n_ddts,
            "effect_residual": effect,
            "mean_resid_present": mean_present,
            "mean_resid_absent": mean_absent,
            "t_stat": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "avg_own_comments": own_comments,
            "total_comments": user_total_comments[user],
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("  No eligible users found. Try lowering thresholds.")
        return results_df

    # Bonferroni correction
    n_tests = len(results_df)
    results_df["p_value_corrected"] = np.minimum(results_df["p_value"] * n_tests, 1.0)
    results_df["significant"] = results_df["p_value_corrected"] < 0.05

    results_df = results_df.sort_values("effect_residual", ascending=False).reset_index(drop=True)
    return results_df


def analyze_user_effects_ridge(df, X_confounds, min_ddts=5, min_comments=10, alpha=10.0):
    """
    Approach 2: Multi-user Ridge regression.

    Fits: total_comments ~ confounders + user1_present + user2_present + ...

    Ridge regression handles multicollinearity between users who tend to
    appear together. The coefficients represent each user's marginal effect
    on comment count, controlling for confounders AND other users.

    Args:
        df: DataFrame of DDT records
        X_confounds: Confound feature matrix
        min_ddts: Minimum DDTs for user inclusion
        min_comments: Minimum total comments for user inclusion
        alpha: Ridge regularization strength

    Returns:
        DataFrame with user ridge coefficients
    """
    # Identify eligible users
    all_users = Counter()
    user_total_comments = Counter()
    for _, row in df.iterrows():
        for user, count in row["commenter_counts"].items():
            all_users[user] += 1
            user_total_comments[user] += count

    eligible_users = sorted([
        u for u, n in all_users.items()
        if n >= min_ddts and user_total_comments[u] >= min_comments
    ])

    if not eligible_users:
        print("\n--- Ridge Regression ---")
        print("  No eligible users found.")
        return pd.DataFrame()

    # Build user presence matrix
    user_features = {}
    for user in eligible_users:
        user_features[f"user_{user}"] = np.array([
            1.0 if user in row["commenters"] else 0.0
            for _, row in df.iterrows()
        ])

    X_users = pd.DataFrame(user_features, index=df.index)

    # Combine confounds + user features
    X_all = pd.concat([X_confounds, X_users], axis=1)
    y = df["total_comments"].values.astype(float)

    # Standardize features for Ridge (important for fair penalization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all.values)

    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y)

    # Extract user coefficients (need to un-scale)
    n_confounds = X_confounds.shape[1]
    coefs = model.coef_

    # The Ridge coefficients are for standardized features.
    # To get interpretable coefficients: coef_original = coef_scaled / std(feature)
    stds = scaler.scale_
    coefs_original = coefs / stds

    user_coefs = {}
    for i, user in enumerate(eligible_users):
        coef_idx = n_confounds + i
        user_coefs[user] = coefs_original[coef_idx]

    # Score
    y_pred = model.predict(X_scaled)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n--- Ridge Regression (alpha={alpha}) ---")
    print(f"  {len(eligible_users)} users included")
    print(f"  R² = {r2:.3f}  (confounders + users explain {r2 * 100:.1f}% of variance)")

    results = []
    for user in eligible_users:
        results.append({
            "user": user,
            "ridge_coef": user_coefs[user],
            "n_ddts": all_users[user],
            "total_comments": user_total_comments[user],
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("ridge_coef", ascending=False).reset_index(drop=True)
    return results_df


def compute_combined_ranking(residual_df, ridge_df):
    """
    Combine residual-based and Ridge-based rankings into a single score.

    Uses rank-averaging: for each user, average their percentile rank
    from both methods. This is robust to outliers in either method.
    """
    if residual_df.empty or ridge_df.empty:
        # Fall back to whichever is available
        if not residual_df.empty:
            residual_df["combined_score"] = residual_df["effect_residual"]
            return residual_df
        elif not ridge_df.empty:
            ridge_df["combined_score"] = ridge_df["ridge_coef"]
            return ridge_df
        return pd.DataFrame()

    # Merge on user
    merged = residual_df.merge(ridge_df[["user", "ridge_coef"]], on="user", how="inner")

    if merged.empty:
        return residual_df

    # Percentile ranks (0 = lowest effect, 1 = highest effect)
    merged["rank_residual"] = merged["effect_residual"].rank(pct=True)
    merged["rank_ridge"] = merged["ridge_coef"].rank(pct=True)
    merged["combined_score"] = (merged["rank_residual"] + merged["rank_ridge"]) / 2

    merged = merged.sort_values("combined_score", ascending=False).reset_index(drop=True)
    return merged


def assign_tiers(combined_df, n_tiers=6):
    """
    Assign tier labels based on combined score percentiles.

    Tiers: S (top ~5%), A (next ~15%), B (next ~25%), C (next ~25%),
           D (next ~20%), F (bottom ~10%)
    """
    if combined_df.empty:
        return combined_df

    df = combined_df.copy()
    n = len(df)

    # Tier boundaries (percentile thresholds from the top)
    tier_specs = [
        ("S", 0.95),   # top 5%
        ("A", 0.80),   # next 15%
        ("B", 0.55),   # next 25%
        ("C", 0.30),   # next 25%
        ("D", 0.10),   # next 20%
        ("F", 0.00),   # bottom 10%
    ]

    score = df["combined_score"]
    tiers = []
    for _, row in df.iterrows():
        s = row["combined_score"]
        assigned = "F"
        for tier_name, threshold in tier_specs:
            if s >= threshold:
                assigned = tier_name
                break
        tiers.append(assigned)

    df["tier"] = tiers
    return df


def print_tier_list(tiered_df, top_n=None):
    """Print the tier list in a readable format."""
    if tiered_df.empty:
        print("\nNo users to display.")
        return

    print("\n" + "=" * 90)
    print("  DDT USER PRESENCE EFFECT TIER LIST")
    print("  (Effect = how much a user's presence changes comment count, controlling for confounders)")
    print("=" * 90)

    tier_order = ["S", "A", "B", "C", "D", "F"]
    tier_labels = {
        "S": "SUPERSTAR  - DDTs light up when they show up",
        "A": "MAJOR      - Noticeably boosts discussion",
        "B": "SOLID      - Positive contributor",
        "C": "NEUTRAL    - Average presence effect",
        "D": "MINOR      - Below-average effect",
        "F": "QUIET      - Minimal measured effect",
    }

    count = 0
    for tier in tier_order:
        tier_users = tiered_df[tiered_df["tier"] == tier]
        if tier_users.empty:
            continue

        print(f"\n{'─' * 90}")
        print(f"  TIER {tier}: {tier_labels.get(tier, '')}")
        print(f"{'─' * 90}")
        print(f"  {'User':<24s} {'Effect':>8s} {'Ridge':>8s} {'DDTs':>6s} "
              f"{'Pres%':>6s} {'AvgOwn':>7s} {'Sig':>5s}")
        print(f"  {'─' * 22}   {'─' * 6}   {'─' * 6}   {'─' * 4}   "
              f"{'─' * 4}   {'─' * 5}   {'─' * 3}")

        for _, row in tier_users.iterrows():
            if top_n and count >= top_n:
                remaining = len(tiered_df) - count
                print(f"\n  ... and {remaining} more users (use --top-n 0 to show all)")
                return

            user = row["user"]
            if len(user) > 22:
                user = user[:20] + ".."

            effect = row.get("effect_residual", 0)
            ridge = row.get("ridge_coef", float("nan"))
            n_ddts = row.get("n_ddts_present", row.get("n_ddts", 0))
            pres = row.get("presence_rate", 0) * 100
            own = row.get("avg_own_comments", 0)
            sig = "***" if row.get("p_value_corrected", 1) < 0.001 else \
                  "**" if row.get("p_value_corrected", 1) < 0.01 else \
                  "*" if row.get("p_value_corrected", 1) < 0.05 else ""

            ridge_str = f"{ridge:+.1f}" if not np.isnan(ridge) else "N/A"

            print(f"  {user:<24s} {effect:+8.1f} {ridge_str:>8s} {n_ddts:6.0f} "
                  f"{pres:5.1f}% {own:7.1f} {sig:>5s}")
            count += 1

    print(f"\n{'=' * 90}")
    print("  Significance: * p<0.05  ** p<0.01  *** p<0.001  (Bonferroni-corrected)")
    print("  Effect: adjusted difference in total comments when user is present vs absent")
    print("  Ridge: marginal effect controlling for all other users simultaneously")
    print("  AvgOwn: user's own average comments per DDT they appear in")
    print(f"{'=' * 90}")


def print_summary_stats(df):
    """Print descriptive statistics about the DDT dataset."""
    print(f"\n{'=' * 60}")
    print("  DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Number of DDTs: {len(df)}")
    print(f"  Total comments: {df['total_comments'].sum():,}")

    print(f"\n  Comments per DDT:")
    print(f"    Mean:   {df['total_comments'].mean():.1f}")
    print(f"    Median: {df['total_comments'].median():.1f}")
    print(f"    Std:    {df['total_comments'].std():.1f}")
    print(f"    Min:    {df['total_comments'].min()}")
    print(f"    Max:    {df['total_comments'].max()}")

    print(f"\n  Comments by day of week:")
    dow_stats = df.groupby("day_of_week")["total_comments"].agg(["mean", "median", "count"])
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in dow_order:
        if day in dow_stats.index:
            row = dow_stats.loc[day]
            print(f"    {day:>12s}: mean={row['mean']:.1f}  median={row['median']:.1f}  (n={row['count']:.0f})")

    # Multi-day DDTs
    multiday = df[df["days_active"] > 1]
    if not multiday.empty:
        print(f"\n  Multi-day DDTs: {len(multiday)}")
        for _, row in multiday.iterrows():
            print(f"    {row['date']} ({row['days_active']} days up, {row['total_comments']} comments)")
    else:
        print(f"\n  Multi-day DDTs: 0")

    # Unique users
    all_users = set()
    for _, row in df.iterrows():
        all_users.update(row["commenters"])
    print(f"\n  Unique commenters: {len(all_users):,}")
    print(f"{'=' * 60}")


def run_analysis(min_ddts=5, min_comments=10, ridge_alpha=10.0, top_n=50):
    """
    Run the full analysis pipeline.

    Args:
        min_ddts: Minimum number of DDTs a user must appear in to be ranked
        min_comments: Minimum total comments across all DDTs
        ridge_alpha: Ridge regularization strength
        top_n: Number of users to display (0 = all)
    """
    # Load data
    df = load_data()
    print_summary_stats(df)

    # Build confound features
    X_confounds = build_confound_features(df)

    # Fit baseline model and get residuals
    residuals, baseline_model = fit_baseline_model(df, X_confounds)

    # Approach 1: Residual-based per-user analysis
    residual_df = analyze_user_effects_residual(df, residuals, min_ddts=min_ddts, min_comments=min_comments)

    # Approach 2: Multi-user Ridge regression
    ridge_df = analyze_user_effects_ridge(df, X_confounds, min_ddts=min_ddts, min_comments=min_comments, alpha=ridge_alpha)

    # Combine rankings
    combined = compute_combined_ranking(residual_df, ridge_df)

    # Assign tiers
    tiered = assign_tiers(combined)

    # Display
    display_n = top_n if top_n > 0 else None
    print_tier_list(tiered, top_n=display_n)

    # Save results to CSV
    output_file = os.path.join(DATA_DIR, "tier_list.csv")
    if not tiered.empty:
        tiered.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

    return tiered


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze which DDT commenters most affect comment count"
    )
    parser.add_argument("--min-ddts", type=int, default=5,
                        help="Minimum DDTs a user must appear in (default: 5)")
    parser.add_argument("--min-comments", type=int, default=10,
                        help="Minimum total comments across DDTs (default: 10)")
    parser.add_argument("--ridge-alpha", type=float, default=10.0,
                        help="Ridge regularization strength (default: 10.0)")
    parser.add_argument("--top-n", type=int, default=50,
                        help="Number of users to display, 0=all (default: 50)")
    args = parser.parse_args()

    run_analysis(
        min_ddts=args.min_ddts,
        min_comments=args.min_comments,
        ridge_alpha=args.ridge_alpha,
        top_n=args.top_n,
    )
