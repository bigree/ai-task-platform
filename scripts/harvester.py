"""
harvester.py
Reddit / Hacker News を巡回して「実装依頼」タスク候補を収集 → queue/tasks.json に追加
認証不要（公開APIのみ）
"""

import requests
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path

QUEUE_PATH = Path(__file__).parent.parent / "queue" / "tasks.json"

REDDIT_TARGETS = [
    {"subreddit": "MachineLearning", "query": "implement code paper"},
    {"subreddit": "MachineLearning", "query": "anyone implemented"},
    {"subreddit": "LocalLLaMA", "query": "implement reproduce"},
    {"subreddit": "learnmachinelearning", "query": "how to implement paper"},
    {"subreddit": "deeplearning", "query": "implementation code"},
    {"subreddit": "computervision", "query": "implementation needed"},
]

HN_QUERIES = [
    "implement paper code",
    "reproduce research",
    "ask hn code",
]

HEADERS = {"User-Agent": "ai-task-platform/1.0 (github.com/bigree)"}


# ── Reddit ──────────────────────────────────────────────

def search_reddit(subreddit: str, query: str, limit: int = 20) -> list[dict]:
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {"q": query, "sort": "new", "limit": limit, "t": "week", "restrict_sr": "1"}
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        posts = resp.json().get("data", {}).get("children", [])
        results = []
        for p in posts:
            d = p["data"]
            score = d.get("score", 0)
            if score < 2:
                continue
            results.append({
                "source": "reddit",
                "subreddit": subreddit,
                "title": d.get("title", ""),
                "body": (d.get("selftext", "") or "")[:300],
                "url": f"https://reddit.com{d.get('permalink', '')}",
                "score": score,
                "created_utc": d.get("created_utc", 0),
            })
        return results
    except Exception as e:
        print(f"  [reddit] Error {subreddit}/{query}: {e}")
        return []


# ── Hacker News ──────────────────────────────────────────

def search_hackernews(query: str, hits: int = 15) -> list[dict]:
    url = "https://hn.algolia.com/api/v1/search_by_date"
    params = {"query": query, "hitsPerPage": hits, "tags": "story", "numericFilters": "points>5"}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("hits", [])
        results = []
        for item in items:
            results.append({
                "source": "hackernews",
                "subreddit": None,
                "title": item.get("title", ""),
                "body": (item.get("story_text") or "")[:300],
                "url": item.get("url") or f"https://news.ycombinator.com/item?id={item.get('objectID')}",
                "score": item.get("points", 0),
                "created_utc": None,
            })
        return results
    except Exception as e:
        print(f"  [hn] Error {query}: {e}")
        return []


# ── Filtering ────────────────────────────────────────────

IMPLEMENTATION_KEYWORDS = [
    "implement", "reproduce", "replication", "code for", "pytorch",
    "tensorflow", "working code", "anyone coded", "open source",
    "github", "poc", "proof of concept", "from scratch",
]

def is_relevant(item: dict) -> bool:
    text = (item["title"] + " " + item["body"]).lower()
    return any(kw in text for kw in IMPLEMENTATION_KEYWORDS)


def make_task_id(item: dict) -> str:
    key = item["url"] or item["title"]
    return "harvest:" + hashlib.md5(key.encode()).hexdigest()[:12]


# ── Main ─────────────────────────────────────────────────

def load_queue() -> list[dict]:
    if QUEUE_PATH.exists():
        with open(QUEUE_PATH) as f:
            return json.load(f)
    return []


def save_queue(tasks: list[dict]) -> None:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_PATH, "w") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)


def run() -> None:
    existing = load_queue()
    existing_ids = {t["id"] for t in existing}
    new_tasks = []

    # Reddit
    for target in REDDIT_TARGETS:
        print(f"[harvester] Reddit r/{target['subreddit']}: {target['query']}")
        items = search_reddit(target["subreddit"], target["query"])
        for item in items:
            if not is_relevant(item):
                continue
            task_id = make_task_id(item)
            if task_id in existing_ids:
                continue
            task = {
                "id": task_id,
                "type": "community_request",
                "status": "pending",
                "title": item["title"],
                "description": item["body"],
                "meta": {
                    "source": item["source"],
                    "subreddit": item["subreddit"],
                    "url": item["url"],
                    "score": item["score"],
                },
                "discovered_at": datetime.utcnow().isoformat(),
                "github_repo": None,
            }
            new_tasks.append(task)
            existing_ids.add(task_id)
            print(f"  → {item['title'][:60]}")
        time.sleep(2)

    # Hacker News
    for query in HN_QUERIES:
        print(f"[harvester] HN: {query}")
        items = search_hackernews(query)
        for item in items:
            if not is_relevant(item):
                continue
            task_id = make_task_id(item)
            if task_id in existing_ids:
                continue
            task = {
                "id": task_id,
                "type": "community_request",
                "status": "pending",
                "title": item["title"],
                "description": item["body"],
                "meta": {
                    "source": "hackernews",
                    "url": item["url"],
                    "score": item["score"],
                },
                "discovered_at": datetime.utcnow().isoformat(),
                "github_repo": None,
            }
            new_tasks.append(task)
            existing_ids.add(task_id)
            print(f"  → {item['title'][:60]}")
        time.sleep(1)

    if new_tasks:
        save_queue(existing + new_tasks)
        print(f"\n[harvester] {len(new_tasks)} new tasks queued.")
    else:
        print("\n[harvester] No new tasks.")


if __name__ == "__main__":
    run()
