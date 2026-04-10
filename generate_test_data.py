"""
Generate realistic synthetic DDT data for testing the analysis pipeline.

Simulates a subreddit with:
- Day-of-week effects (weekdays busier than weekends)
- A few "superstar" users whose presence correlates with high activity
- Some users who are just frequent but don't drive engagement
- Seasonal variation and a long-term trend
- Occasional community events that spike activity
"""

import json
import os
import random
from datetime import datetime, timedelta, timezone

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
POSTS_FILE = os.path.join(DATA_DIR, "ddt_posts.json")
COMMENTS_DIR = os.path.join(DATA_DIR, "comments")


def generate():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(COMMENTS_DIR, exist_ok=True)

    rng = np.random.default_rng(42)
    random.seed(42)

    # Simulate 365 days of DDTs
    start_date = datetime(2025, 4, 10, 16, 0, 0, tzinfo=timezone.utc)
    n_days = 365

    # Define user archetypes
    # Format: (name, base_show_rate, engagement_effect, avg_own_comments)
    users = [
        # S-tier: users whose presence strongly correlates with more comments
        ("MeleeGod42", 0.40, 25, 8),
        ("FoxMain2007", 0.35, 20, 12),

        # A-tier: solid positive effect
        ("WaveDashWizard", 0.55, 12, 6),
        ("ShineSpike_", 0.45, 10, 5),
        ("PM_ME_TECH", 0.50, 8, 4),

        # B-tier: modest positive effect
        ("Falco_Enjoyer", 0.60, 5, 3),
        ("TourneyTO_Steve", 0.40, 5, 4),
        ("SetCountBot", 0.70, 3, 2),
        ("MarfMain", 0.55, 4, 3),
        ("SlippiRanked", 0.50, 4, 3),

        # C-tier: neutral
        ("CasualMelee", 0.65, 0, 2),
        ("PuffIsLame", 0.50, 0, 3),
        ("UCFDebater", 0.45, -1, 2),
        ("NetplayAndy", 0.60, 0, 2),
        ("LocalSceneFan", 0.40, 1, 2),
        ("TopPlayerWatcher", 0.55, 0, 1),
        ("SmashClipGuy", 0.35, 1, 2),
        ("TechChaseKing", 0.45, 0, 2),

        # D-tier: slightly negative (their posts maybe aren't great?)
        ("TierListDebater", 0.50, -5, 4),
        ("RulesetComplainer", 0.35, -4, 3),
        ("SaltyRunback", 0.40, -3, 2),
        ("IciesApologist", 0.30, -3, 2),

        # F-tier: negative effect
        ("TrollPoster9000", 0.25, -10, 6),
        ("copypasta_bot_v2", 0.20, -8, 3),

        # Extra filler users (many occasional commenters)
    ]

    # Add 40 more filler users with minimal effect
    for i in range(40):
        name = f"ssbm_fan_{i:03d}"
        show_rate = rng.uniform(0.05, 0.30)
        effect = rng.normal(0, 1.5)
        own_comments = rng.integers(1, 3)
        users.append((name, show_rate, effect, int(own_comments)))

    # Day-of-week baseline effects (Mon=0, Sun=6)
    dow_effects = {
        0: 10,   # Monday - solid
        1: 8,    # Tuesday
        2: 5,    # Wednesday
        3: 12,   # Thursday - pre-weekend hype
        4: 15,   # Friday - big day
        5: -5,   # Saturday - people are at tournaments/outside
        6: -8,   # Sunday - same
    }

    posts = []
    all_comments = {}

    base_activity = 80  # baseline comments per DDT

    for day_offset in range(n_days):
        dt = start_date + timedelta(days=day_offset)
        dow = dt.weekday()
        month = dt.month

        # Skip ~5% of days randomly (DDT not posted or holiday).
        # This also simulates multi-day DDTs: when a day is skipped,
        # the previous DDT stays up longer and accumulates extra comments.
        if rng.random() < 0.05:
            # Add extra comments to the previous DDT (it's still pinned)
            if posts:
                prev_id = posts[-1]["id"]
                extra = int(rng.integers(15, 50))
                for j in range(extra):
                    all_comments[prev_id].append({
                        "author": f"lurker_{rng.integers(0, 500):04d}",
                        "created_utc": dt.timestamp() + rng.integers(0, 86400),
                        "score": int(rng.integers(-1, 10)),
                        "body_length": int(rng.integers(5, 200)),
                        "id": f"c_{prev_id}_extra_{j}",
                    })
                posts[-1]["num_comments"] = len(all_comments[prev_id])
                # Re-save updated comments
                comment_file = os.path.join(COMMENTS_DIR, f"{prev_id}.json")
                with open(comment_file, "w") as f:
                    json.dump(all_comments[prev_id], f, indent=2)
            continue

        post_id = f"sim_{day_offset:04d}"

        # Calculate expected comment count
        expected = base_activity

        # Day of week effect
        expected += dow_effects[dow]

        # Seasonal effect (summer/winter break = more activity)
        if month in (6, 7, 12, 1):
            expected += 10
        elif month in (8, 9):  # back to school
            expected -= 5

        # Long-term trend (slight decline over the year, realistic for subreddits)
        expected += -0.02 * day_offset

        # Occasional event spikes (major tournaments, drama, etc.)
        # ~5% of days have a community event
        event_spike = 0
        if rng.random() < 0.05:
            event_spike = rng.integers(20, 60)
            expected += event_spike

        # Determine which users show up today
        present_users = []
        for name, show_rate, effect, avg_own in users:
            # Users are more likely to show up on high-activity days
            adjusted_rate = show_rate
            if event_spike > 0:
                adjusted_rate = min(0.95, show_rate + 0.2)

            # Weekend warriors vs weekday regulars
            if dow >= 5:
                adjusted_rate *= 0.8

            if rng.random() < adjusted_rate:
                present_users.append((name, effect, avg_own))
                expected += effect

        # Add noise
        expected += rng.normal(0, 10)
        total_comments = max(5, int(expected))

        # Generate individual comments
        comments = []

        # First, add comments from present users
        for name, effect, avg_own in present_users:
            n_comments = max(1, int(rng.poisson(avg_own)))
            for j in range(n_comments):
                comments.append({
                    "author": name,
                    "created_utc": dt.timestamp() + rng.integers(0, 86400),
                    "score": int(rng.integers(-2, 20)),
                    "body_length": int(rng.integers(10, 500)),
                    "id": f"c_{post_id}_{name}_{j}",
                })

        # Fill remaining comments with anonymous/rare users
        while len(comments) < total_comments:
            anon_name = f"lurker_{rng.integers(0, 500):04d}"
            comments.append({
                "author": anon_name,
                "created_utc": dt.timestamp() + rng.integers(0, 86400),
                "score": int(rng.integers(-1, 10)),
                "body_length": int(rng.integers(5, 200)),
                "id": f"c_{post_id}_anon_{len(comments)}",
            })

        title_date = dt.strftime("%B %d, %Y")
        posts.append({
            "id": post_id,
            "title": f"Daily Discussion Thread {title_date}",
            "created_utc": dt.timestamp(),
            "num_comments": len(comments),
            "score": int(rng.integers(5, 30)),
            "permalink": f"/r/ssbm/comments/{post_id}/daily_discussion_thread/",
            "author": "AutoModerator",
        })

        # Save comments
        comment_file = os.path.join(COMMENTS_DIR, f"{post_id}.json")
        with open(comment_file, "w") as f:
            json.dump(comments, f, indent=2)

        all_comments[post_id] = comments

    # Save posts
    with open(POSTS_FILE, "w") as f:
        json.dump(posts, f, indent=2)

    total = sum(len(c) for c in all_comments.values())
    print(f"Generated {len(posts)} synthetic DDTs with {total:,} total comments")
    print(f"Data saved to {DATA_DIR}/")


if __name__ == "__main__":
    generate()
