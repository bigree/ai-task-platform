"""
github_publisher.py
PoC コードを生成後、GitHub に新規 repo を作成して push する。

使い方:
  python github_publisher.py \
    --task-id "paper:2401.12345" \
    --repo-name "poc-attention-mechanism-2401.12345" \
    --description "PoC implementation of ..." \
    --files src/model.py src/train.py README.md

環境変数:
  GITHUB_TOKEN  : Personal Access Token (repo scope)
  GITHUB_USER   : GitHub username (例: bigree)
"""

import argparse
import os
import json
import base64
from pathlib import Path
import requests

QUEUE_PATH = Path(__file__).parent.parent / "queue" / "tasks.json"


def get_headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise EnvironmentError("GITHUB_TOKEN が設定されていません")
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def create_repo(repo_name: str, description: str, user: str) -> str:
    """GitHub に新規 public リポジトリを作成し、URL を返す。"""
    headers = get_headers()
    payload = {
        "name": repo_name,
        "description": description,
        "private": False,
        "auto_init": False,
        "has_issues": True,
        "has_wiki": False,
    }
    resp = requests.post("https://api.github.com/user/repos", headers=headers, json=payload)
    if resp.status_code == 422:
        # すでに存在する場合はそのまま使う
        print(f"  [github] Repo {repo_name} already exists, reusing.")
        return f"https://github.com/{user}/{repo_name}"
    resp.raise_for_status()
    return resp.json()["html_url"]


def push_file(user: str, repo: str, filepath: str, content: str, commit_msg: str) -> None:
    """ファイルを GitHub リポジトリに追加/更新する。"""
    headers = get_headers()
    url = f"https://api.github.com/repos/{user}/{repo}/contents/{filepath}"

    # 既存ファイルの SHA を取得（更新時に必要）
    sha = None
    check = requests.get(url, headers=headers)
    if check.status_code == 200:
        sha = check.json().get("sha")

    payload = {
        "message": commit_msg,
        "content": base64.b64encode(content.encode()).decode(),
    }
    if sha:
        payload["sha"] = sha

    resp = requests.put(url, headers=headers, json=payload)
    resp.raise_for_status()


def publish(task_id: str, repo_name: str, description: str, files: dict[str, str]) -> str:
    """
    files: {相対パス: ファイル内容} の dict
    戻り値: 作成された GitHub リポジトリの URL
    """
    user = os.environ.get("GITHUB_USER", "bigree")

    print(f"[github_publisher] Creating repo: {repo_name}")
    repo_url = create_repo(repo_name, description, user)

    for filepath, content in files.items():
        print(f"  Pushing {filepath} ...")
        push_file(user, repo_name, filepath, content, f"Add {filepath}")

    # queue/tasks.json を更新
    update_queue(task_id, repo_url)

    print(f"[github_publisher] Done: {repo_url}")
    return repo_url


def update_queue(task_id: str, repo_url: str) -> None:
    if not QUEUE_PATH.exists():
        return
    with open(QUEUE_PATH) as f:
        tasks = json.load(f)
    for task in tasks:
        if task["id"] == task_id:
            task["status"] = "completed"
            task["github_repo"] = repo_url
            break
    with open(QUEUE_PATH, "w") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)


# ── CLI ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GitHub PoC Publisher")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--repo-name", required=True)
    parser.add_argument("--description", default="PoC implementation")
    parser.add_argument("--files", nargs="+", default=[], help="公開するファイルのパス")
    args = parser.parse_args()

    file_contents = {}
    for fp in args.files:
        p = Path(fp)
        if p.exists():
            file_contents[p.name] = p.read_text()
        else:
            print(f"  Warning: {fp} not found, skip.")

    publish(args.task_id, args.repo_name, args.description, file_contents)
