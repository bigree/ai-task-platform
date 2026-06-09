"""
paper_scout.py
arXiv の最新論文を取得 → Papers With Code でコード未実装を検出 → queue/tasks.json に追加
"""

import requests
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
import time

# 対象カテゴリ（外科AI・医療画像・CV・ML）
ARXIV_CATEGORIES = ["cs.CV", "cs.LG", "cs.AI", "eess.IV", "cs.RO"]
DAYS_BACK = 7
MAX_PER_CATEGORY = 30
QUEUE_PATH = Path(__file__).parent.parent / "queue" / "tasks.json"

ARXIV_NS = "http://www.w3.org/2005/Atom"


def fetch_arxiv_papers(category: str, max_results: int = MAX_PER_CATEGORY) -> list[dict]:
    """arXiv API から最新論文を取得する。"""
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    papers = []
    cutoff = datetime.utcnow() - timedelta(days=DAYS_BACK)

    for entry in root.findall(f"{{{ARXIV_NS}}}entry"):
        published_str = entry.findtext(f"{{{ARXIV_NS}}}published", "")
        try:
            published = datetime.fromisoformat(published_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            continue
        if published < cutoff:
            continue

        arxiv_id = entry.findtext(f"{{{ARXIV_NS}}}id", "").split("/abs/")[-1]
        title = entry.findtext(f"{{{ARXIV_NS}}}title", "").strip().replace("\n", " ")
        summary = entry.findtext(f"{{{ARXIV_NS}}}summary", "").strip().replace("\n", " ")[:300]
        authors = [
            a.findtext(f"{{{ARXIV_NS}}}name", "")
            for a in entry.findall(f"{{{ARXIV_NS}}}author")
        ]

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "summary": summary,
            "authors": authors[:5],
            "published": published_str,
            "category": category,
            "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
        })

    return papers


def has_code_on_pwc(arxiv_id: str) -> bool:
    """Papers With Code API でコード実装の有無を確認する。"""
    try:
        # まず論文が登録されているか確認
        resp = requests.get(
            "https://paperswithcode.com/api/v1/papers/",
            params={"arxiv_id": arxiv_id},
            timeout=15,
        )
        if resp.status_code != 200 or resp.json().get("count", 0) == 0:
            return False

        paper_id = resp.json()["results"][0]["id"]

        # リポジトリが紐付いているか確認
        repo_resp = requests.get(
            f"https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/",
            timeout=15,
        )
        if repo_resp.status_code == 200:
            return repo_resp.json().get("count", 0) > 0
    except Exception:
        pass
    return False


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

    for category in ARXIV_CATEGORIES:
        print(f"[paper_scout] Scanning {category} ...")
        papers = fetch_arxiv_papers(category)
        for paper in papers:
            task_id = f"paper:{paper['arxiv_id']}"
            if task_id in existing_ids:
                continue

            print(f"  Checking PwC: {paper['arxiv_id']} ...")
            if has_code_on_pwc(paper["arxiv_id"]):
                print(f"  → Code exists, skip.")
                continue

            task = {
                "id": task_id,
                "type": "paper_poc",
                "status": "pending",
                "title": paper["title"],
                "description": paper["summary"],
                "meta": {
                    "arxiv_id": paper["arxiv_id"],
                    "arxiv_url": paper["arxiv_url"],
                    "authors": paper["authors"],
                    "published": paper["published"],
                    "category": paper["category"],
                },
                "discovered_at": datetime.utcnow().isoformat(),
                "github_repo": None,
            }
            new_tasks.append(task)
            existing_ids.add(task_id)
            print(f"  → Queued: {paper['title'][:60]}")
            time.sleep(1)  # API rate limit

        time.sleep(2)

    if new_tasks:
        save_queue(existing + new_tasks)
        print(f"\n[paper_scout] {len(new_tasks)} new papers queued.")
    else:
        print("\n[paper_scout] No new papers.")


if __name__ == "__main__":
    run()
