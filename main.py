#!/usr/bin/env python3
"""
DDT Analyzer - Measure which r/ssbm users most affect Daily Discussion Thread activity.

This tool scrapes DDT posts and comments from r/ssbm, then uses statistical
analysis to determine which user's presence has the biggest effect on comment
count, controlling for confounding factors (day of week, time trends, seasonal
patterns, community-wide activity surges).

Usage:
    python main.py                          # Full pipeline: scrape + analyze
    python main.py --skip-scrape            # Analyze only (use cached data)
    python main.py --min-ddts 10            # Require 10+ DDT appearances
    python main.py --min-comments 20        # Require 20+ total comments
    python main.py --top-n 0               # Show all users (no limit)
    python main.py --ridge-alpha 5.0        # Adjust Ridge regularization

Scraping options:
    python main.py --max-pages 30           # Fetch more search result pages
    python main.py --refresh-posts          # Re-fetch post list from Reddit
    python main.py --refresh-comments       # Re-fetch all comments from Reddit
"""

import argparse
import sys

from scrape import scrape_all
from analyze import run_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Analyze which r/ssbm DDT commenters most affect comment count.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        Full run (scrape + analyze)
  python main.py --skip-scrape          Analyze cached data only
  python main.py --min-ddts 10          Users must appear in 10+ DDTs
  python main.py --min-comments 20      Users must have 20+ total comments
  python main.py --top-n 100            Show top 100 users
  python main.py --top-n 0              Show all users

Statistical methodology:
  1. Baseline model predicts comment count from confounders:
     day-of-week, time trend, month, rolling activity level
  2. Residual analysis: for each user, compare mean residual
     when present vs absent (Welch's t-test, Bonferroni-corrected)
  3. Ridge regression: multi-user model controlling for
     user-user correlations (friend groups, etc.)
  4. Combined ranking averages percentile ranks from both methods
        """,
    )

    # Scraping options
    scrape_group = parser.add_argument_group("Scraping options")
    scrape_group.add_argument("--skip-scrape", action="store_true",
                              help="Skip scraping, use cached data")
    scrape_group.add_argument("--max-pages", type=int, default=20,
                              help="Max search result pages (default: 20)")
    scrape_group.add_argument("--refresh-posts", action="store_true",
                              help="Force re-fetch of post list")
    scrape_group.add_argument("--refresh-comments", action="store_true",
                              help="Force re-fetch of all comments")

    # Analysis options
    analysis_group = parser.add_argument_group("Analysis options")
    analysis_group.add_argument("--min-ddts", type=int, default=5,
                                help="Min DDTs a user must appear in (default: 5)")
    analysis_group.add_argument("--min-comments", type=int, default=10,
                                help="Min total comments across DDTs (default: 10)")
    analysis_group.add_argument("--ridge-alpha", type=float, default=10.0,
                                help="Ridge regularization strength (default: 10.0)")
    analysis_group.add_argument("--top-n", type=int, default=50,
                                help="Users to display, 0=all (default: 50)")

    args = parser.parse_args()

    # Step 1: Scrape (unless skipped)
    if not args.skip_scrape:
        print("=" * 60)
        print("  STEP 1: SCRAPING DDT DATA FROM r/ssbm")
        print("=" * 60)
        posts, comments = scrape_all(
            max_search_pages=args.max_pages,
            force_refresh_posts=args.refresh_posts,
            force_refresh_comments=args.refresh_comments,
        )
        total = sum(len(c) for c in comments.values())
        print(f"\nScraping complete: {len(posts)} DDTs, {total:,} comments\n")
    else:
        print("Skipping scrape, using cached data.\n")

    # Step 2: Analyze
    print("=" * 60)
    print("  STEP 2: STATISTICAL ANALYSIS")
    print("=" * 60)
    run_analysis(
        min_ddts=args.min_ddts,
        min_comments=args.min_comments,
        ridge_alpha=args.ridge_alpha,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
