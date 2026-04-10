"""
Scrape r/ssbm Daily Discussion Threads from Reddit.

Fetches DDT posts via Reddit's public JSON API, then fetches all comments
for each post. Results are cached to disk as JSON to avoid re-fetching.
"""

import json
import os
import re
import time
from datetime import datetime, timezone

import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
POSTS_FILE = os.path.join(DATA_DIR, "ddt_posts.json")
COMMENTS_DIR = os.path.join(DATA_DIR, "comments")

HEADERS = {
    "User-Agent": "DDTAnalyzer/1.0 (research script; analyzing DDT comment patterns)"
}

# Reddit rate limit: ~60 requests/min for unauthenticated. We'll be conservative.
REQUEST_DELAY = 1.5  # seconds between requests


def _get_json(url, params=None, max_retries=3):
    """Fetch JSON from Reddit with rate limiting and retries."""
    for attempt in range(max_retries):
        time.sleep(REQUEST_DELAY)
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Request error ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return None
    return None


def search_ddt_posts(max_pages=20):
    """
    Search r/ssbm for Daily Discussion Thread posts.

    Uses Reddit search API with pagination. Returns list of post metadata dicts.
    """
    posts = []
    after = None
    url = "https://www.reddit.com/r/ssbm/search.json"

    for page in range(max_pages):
        params = {
            "q": 'title:"Daily Discussion Thread"',
            "restrict_sr": "on",
            "sort": "new",
            "t": "all",
            "limit": 100,
        }
        if after:
            params["after"] = after

        print(f"  Fetching search page {page + 1}...")
        data = _get_json(url, params=params)
        if not data or "data" not in data:
            print("  No more search results or error.")
            break

        children = data["data"].get("children", [])
        if not children:
            break

        for child in children:
            post = child["data"]
            # Filter to actual DDTs by title pattern
            title = post.get("title", "")
            if not re.search(r"Daily Discussion Thread", title, re.IGNORECASE):
                continue

            posts.append({
                "id": post["id"],
                "title": title,
                "created_utc": post["created_utc"],
                "num_comments": post["num_comments"],
                "score": post.get("score", 0),
                "permalink": post.get("permalink", ""),
                "author": post.get("author", "[deleted]"),
            })

        after = data["data"].get("after")
        if not after:
            print(f"  Reached end of search results after {page + 1} pages.")
            break

    # Deduplicate by post ID
    seen = set()
    unique_posts = []
    for p in posts:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique_posts.append(p)

    # Sort by date (oldest first)
    unique_posts.sort(key=lambda p: p["created_utc"])
    return unique_posts


def _extract_comments_from_tree(tree, comments_list):
    """Recursively extract comments from Reddit's comment tree structure."""
    if isinstance(tree, list):
        for item in tree:
            _extract_comments_from_tree(item, comments_list)
        return

    if isinstance(tree, dict):
        kind = tree.get("kind")
        data = tree.get("data", {})

        if kind == "Listing":
            for child in data.get("children", []):
                _extract_comments_from_tree(child, comments_list)

        elif kind == "t1":  # Comment
            author = data.get("author", "[deleted]")
            if author not in ("[deleted]", "[removed]", "AutoModerator"):
                comments_list.append({
                    "author": author,
                    "created_utc": data.get("created_utc", 0),
                    "score": data.get("score", 0),
                    "body_length": len(data.get("body", "")),
                    "id": data.get("id", ""),
                })
            # Recurse into replies
            replies = data.get("replies")
            if replies and isinstance(replies, dict):
                _extract_comments_from_tree(replies, comments_list)

        elif kind == "more":
            # "Load more comments" stubs - we'll fetch these separately
            more_ids = data.get("children", [])
            if more_ids:
                comments_list.append({"_more_ids": more_ids, "_parent_id": data.get("parent_id", "")})


def _fetch_more_comments(post_id, comment_ids):
    """Fetch 'more comments' that weren't included in the initial response."""
    # Use the morechildren API endpoint
    url = "https://www.reddit.com/api/morechildren.json"
    all_comments = []

    # Process in batches of 100 (API limit)
    for i in range(0, len(comment_ids), 100):
        batch = comment_ids[i:i + 100]
        params = {
            "api_type": "json",
            "link_id": f"t3_{post_id}",
            "children": ",".join(batch),
            "limit_children": False,
            "sort": "top",
        }
        data = _get_json(url, params=params)
        if not data:
            continue

        # morechildren returns a flat list in json.data.things
        things = (data.get("json", {}).get("data", {}).get("things", []))
        for thing in things:
            if thing.get("kind") == "t1":
                tdata = thing["data"]
                author = tdata.get("author", "[deleted]")
                if author not in ("[deleted]", "[removed]", "AutoModerator"):
                    all_comments.append({
                        "author": author,
                        "created_utc": tdata.get("created_utc", 0),
                        "score": tdata.get("score", 0),
                        "body_length": len(tdata.get("body", "")),
                        "id": tdata.get("id", ""),
                    })

    return all_comments


def fetch_comments_for_post(post_id):
    """Fetch all comments for a given post, handling 'load more' stubs."""
    url = f"https://www.reddit.com/r/ssbm/comments/{post_id}.json"
    params = {"limit": 500, "depth": 10, "sort": "top"}

    data = _get_json(url, params=params)
    if not data or not isinstance(data, list) or len(data) < 2:
        return []

    comments = []
    _extract_comments_from_tree(data[1], comments)

    # Separate actual comments from "more" stubs
    real_comments = [c for c in comments if "author" in c]
    more_stubs = [c for c in comments if "_more_ids" in c]

    # Fetch "more" comments
    all_more_ids = []
    for stub in more_stubs:
        all_more_ids.extend(stub["_more_ids"])

    if all_more_ids:
        print(f"    Fetching {len(all_more_ids)} additional comments...")
        extra = _fetch_more_comments(post_id, all_more_ids)
        real_comments.extend(extra)

    return real_comments


def scrape_all(max_search_pages=20, force_refresh_posts=False, force_refresh_comments=False):
    """
    Main scraping function. Fetches posts and comments, caching results to disk.

    Args:
        max_search_pages: Max number of search result pages to fetch
        force_refresh_posts: If True, re-fetch the post list even if cached
        force_refresh_comments: If True, re-fetch comments even if cached

    Returns:
        Tuple of (posts_list, dict mapping post_id -> comments_list)
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(COMMENTS_DIR, exist_ok=True)

    # Step 1: Get DDT posts
    if not force_refresh_posts and os.path.exists(POSTS_FILE):
        print(f"Loading cached posts from {POSTS_FILE}")
        with open(POSTS_FILE, "r") as f:
            posts = json.load(f)
        print(f"  {len(posts)} posts loaded from cache.")
    else:
        print("Searching for DDT posts on r/ssbm...")
        posts = search_ddt_posts(max_pages=max_search_pages)
        with open(POSTS_FILE, "w") as f:
            json.dump(posts, f, indent=2)
        print(f"  Found and cached {len(posts)} DDT posts.")

    # Step 2: Fetch comments for each post
    all_comments = {}
    total = len(posts)
    for i, post in enumerate(posts):
        pid = post["id"]
        comment_file = os.path.join(COMMENTS_DIR, f"{pid}.json")

        if not force_refresh_comments and os.path.exists(comment_file):
            with open(comment_file, "r") as f:
                all_comments[pid] = json.load(f)
        else:
            print(f"  [{i + 1}/{total}] Fetching comments for: {post['title'][:60]}...")
            comments = fetch_comments_for_post(pid)
            with open(comment_file, "w") as f:
                json.dump(comments, f, indent=2)
            all_comments[pid] = comments
            print(f"    Got {len(comments)} comments.")

    return posts, all_comments


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape r/ssbm DDT posts and comments")
    parser.add_argument("--max-pages", type=int, default=20,
                        help="Max search result pages to fetch (default: 20)")
    parser.add_argument("--refresh-posts", action="store_true",
                        help="Force re-fetch of post list")
    parser.add_argument("--refresh-comments", action="store_true",
                        help="Force re-fetch of all comments")
    args = parser.parse_args()

    posts, comments = scrape_all(
        max_search_pages=args.max_pages,
        force_refresh_posts=args.refresh_posts,
        force_refresh_comments=args.refresh_comments,
    )

    total_comments = sum(len(c) for c in comments.values())
    print(f"\nDone! {len(posts)} DDTs, {total_comments} total comments scraped.")
