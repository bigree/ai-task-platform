# AI Task Platform

論文 PoC 実装 & 汎用 AI タスクのオープンプラットフォーム。

**コンセプト**: 毎日自動でネットを巡回し、コードが公開されていない論文やAIで処理できるタスクを拾って実装・公開する。

---

## 仕組み

```
[自動収集]                        [処理]                [公開]
arXiv（論文）   ─→               
Reddit / HN ─→  queue/tasks.json  →  Claude で処理  →  GitHub Repo
GitHub Issues ─→                 
```

| コンポーネント | 説明 |
|---|---|
| `scripts/paper_scout.py` | arXiv の最新論文 → Papers With Code 未掲載のものをキューへ |
| `scripts/harvester.py` | Reddit / HN から実装依頼を検出してキューへ |
| `scripts/github_publisher.py` | PoC コード完成後に GitHub Repo を自動作成・push |
| `docs/index.html` | GitHub Pages でホストするタスク投稿フォーム |
| `queue/tasks.json` | タスクキュー（GitHub Actions が毎日更新） |

---

## セットアップ

### 1. このリポジトリを fork / clone

```bash
git clone https://github.com/bigree/ai-task-platform
cd ai-task-platform
pip install requests
```

### 2. GitHub Actions の有効化

fork したリポジトリの **Actions** タブから有効化するだけ。  
毎日 JST 10:00 に `paper_scout` + `harvester` が自動実行され、`queue/tasks.json` が更新される。

### 3. GitHub Pages の有効化

Settings → Pages → Source: `main` branch `/docs` folder

### 4. PoC を公開する（手動）

```bash
export GITHUB_TOKEN=ghp_xxxxx
export GITHUB_USER=bigree

python scripts/github_publisher.py \
  --task-id "paper:2401.12345" \
  --repo-name "poc-method-name-2401.12345" \
  --description "PoC implementation of [Paper Title]" \
  --files model.py train.py README.md
```

---

## タスクキューの形式

```json
{
  "id": "paper:2401.12345",
  "type": "paper_poc",
  "status": "pending",
  "title": "論文タイトル",
  "description": "アブストラクト...",
  "meta": {
    "arxiv_id": "2401.12345",
    "arxiv_url": "https://arxiv.org/abs/2401.12345",
    "authors": ["Author A", "Author B"],
    "category": "cs.CV"
  },
  "discovered_at": "2026-06-09T00:00:00",
  "github_repo": null
}
```

`status`: `pending` → `in_progress` → `completed`

---

## タスクを依頼する

- **Webフォーム**: https://bigree.github.io/ai-task-platform/
- **GitHub Issues**: [New Issue](https://github.com/bigree/ai-task-platform/issues/new/choose)

---

## ライセンス

MIT — 生成した PoC コードはすべて MIT ライセンスで公開します。
