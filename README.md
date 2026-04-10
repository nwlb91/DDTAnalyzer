# DDT Analyzer

Analyze which r/ssbm Daily Discussion Thread commenters have the biggest effect on comment count.

## What it does

Scrapes DDT posts and comments from r/ssbm, then uses statistical analysis to measure each user's marginal effect on thread activity, controlling for confounding factors.

## Quick start

```bash
pip install -r requirements.txt

# Full pipeline: scrape from Reddit + analyze
python main.py

# Analyze only (after initial scrape)
python main.py --skip-scrape

# Test with synthetic data
python generate_test_data.py
python main.py --skip-scrape
```

## Tunable parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--min-ddts` | 5 | Minimum DDTs a user must appear in to be ranked |
| `--min-comments` | 10 | Minimum total comments across all DDTs |
| `--ridge-alpha` | 10.0 | Ridge regularization strength |
| `--top-n` | 50 | Number of users to display (0 = all) |
| `--max-pages` | 20 | Reddit search result pages to fetch |
| `--refresh-posts` | off | Force re-fetch post list from Reddit |
| `--refresh-comments` | off | Force re-fetch all comments |
| `--skip-scrape` | off | Skip scraping, use cached data |

## Statistical methodology

### Confounders controlled for

1. **Day of week** - weekday vs weekend activity differences
2. **Time trend** - linear + quadratic trend capturing community growth/decline
3. **Month of year** - seasonal patterns (tournament seasons, school schedules)
4. **Rolling activity** - 7-day lagged moving average captures event-driven surges (majors, community drama) without leaking current-day info

### Two complementary approaches

**Approach 1: Residual analysis**
- Fit a baseline OLS model: `comments ~ confounders`
- Compute residuals (unexplained variation)
- For each user, compare mean residual when present vs absent
- Statistical significance via Welch's t-test with Bonferroni correction

**Approach 2: Multi-user Ridge regression**
- Fit: `comments ~ confounders + all_user_presence_indicators`
- Ridge (L2) regularization handles multicollinearity between users who tend to appear together
- Coefficients represent marginal effects controlling for other users

### Combined ranking

Final tier assignment averages percentile ranks from both approaches, making it robust to the weaknesses of either method alone.

## Output

Users are ranked into tiers (S/A/B/C/D/F) with:
- **Effect**: adjusted comment count difference (present vs absent)
- **Ridge**: marginal effect controlling for all other users
- **DDTs**: number of threads the user appeared in
- **Pres%**: percentage of DDTs the user appeared in
- **AvgOwn**: user's own average comments per DDT
- **Sig**: statistical significance (`*` p<0.05, `**` p<0.01, `***` p<0.001, Bonferroni-corrected)

Results are also saved to `data/tier_list.csv`.

## Files

- `main.py` - CLI entry point (scrape + analyze)
- `scrape.py` - Reddit data collection with caching
- `analyze.py` - Statistical analysis and tier list generation
- `generate_test_data.py` - Synthetic data generator for testing
