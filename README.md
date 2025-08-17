# マルチチャットボットRAGシステム v0.1.0

完全無料のローカル動作マルチチャットボット対応RAG（Retrieval-Augmented Generation）システムです。

## 🚀 クイックスタート

### 必要なツール

1. **Nix Package Manager** (https://nixos.org/nix/install)
   ```bash
   sh <(curl -L https://nixos.org/nix/install) --daemon
   ```

2. **Git**
   ```bash
   git clone https://github.com/rikumiura32543/multi_chatbot.git
   cd multi_chatbot
   ```

### インストール・起動

```bash
# 1. Nix開発環境に入る
nix-shell

# 2. 依存関係をインストール
uv sync

# 3. AIモデルをダウンロード（初回のみ）
ollama pull qwen2:7b

# 4. サーバー起動
uv run python main.py --host 0.0.0.0 --port 8000
```

ブラウザで http://localhost:8000 にアクセスしてください。

## ✨ 主な機能

- **🤖 マルチチャットボット管理**: 最大5つのチャットボットを作成・管理
- **📁 ローカルフォルダ対応RAG**: PDF/Word/Markdown文書をドラッグ&ドロップまたはフォルダ選択で簡単設定
- **🔍 ハイブリッド検索**: ベクトル検索(70%) + キーワード検索(30%)の組み合わせで高精度
- **💬 リアルタイムチャット**: 文書に基づいた正確な回答と引用元表示
- **🖼️ iframe埋め込み**: 外部Webサイトへの埋め込み対応
- **📱 レスポンシブデザイン**: PC・モバイル両対応のミニマルフラットUI

## 🏗️ システム要件

- **OS**: Linux, macOS, Windows (WSL2)
- **RAM**: 8GB以上推奨（qwen2:7bモデル用）
- **ディスク**: 10GB以上
- **CPU**: 2コア以上

## 📋 技術スタック

- **LLM**: Ollama + qwen2:7b（8GB RAM最適化）
- **ベクトルDB**: ChromaDB（軽量・高速）
- **フレームワーク**: FastAPI + LangChain
- **埋め込み**: all-MiniLM-L6-v2
- **UI**: Jinja2 + ミニマルフラットCSS
- **環境管理**: Nix + uv

## 🖥️ 使い方

### 1. チャットボット作成
1. ホーム画面で「新規チャットボット作成」
2. 名前と説明を入力
3. 「作成」ボタンをクリック

### 2. RAG設定
1. 作成したチャットボットの「RAG設定」ボタン
2. フォルダ選択または個別ファイルアップロード
3. 「RAG設定」ボタンで文書処理開始
4. 処理完了まで待機

### 3. チャット開始
1. 「チャット開始」ボタンでチャット画面へ
2. 質問を入力してEnterまたは送信ボタン
3. AI回答と参考文書の引用を確認

### 4. 外部サイト埋め込み
1. チャット画面の「埋め込みコード」ボタン
2. iframe または JavaScript コードをコピー
3. 外部サイトに貼り付け

## ⚙️ 設定

### 環境変数（オプション）
```bash
# .env ファイルを作成（オプション）
OLLAMA_HOST=127.0.0.1:11434
DATA_DIR=./data
LOG_LEVEL=INFO
```

### RAG設定のカスタマイズ
- **チャンクサイズ**: 文書分割の単位（デフォルト: 1000文字）
- **チャンクオーバーラップ**: 隣接チャンク間の重複（デフォルト: 200文字）
- **検索結果数**: 関連文書の取得数（デフォルト: 3件）

## 🚧 トラブルシューティング

### よくある問題

**Q: "qwen2:7b" モデルが見つからない**
```bash
ollama list  # モデル確認
ollama pull qwen2:7b  # モデルダウンロード
```

**Q: メモリ不足エラー**
```bash
# 軽量モデルに変更
ollama pull phi3:mini  # 3.8GB
ollama pull gemma2:2b  # 1.6GB
```

**Q: ポート8000にアクセスできない**
- ファイアウォール設定確認
- `--host 0.0.0.0` オプション確認

**Q: Nixインストールに失敗**
```bash
# 再試行
curl -L https://nixos.org/nix/install | sh -s -- --daemon
source ~/.bashrc
```

## 📊 パフォーマンス

実測データ（Google Cloud E2-standard-2環境）:
- **同時接続**: 50ユーザー対応
- **成功率**: 100%
- **平均応答時間**: 0.004秒
- **メモリ使用量**: 6-7GB（qwen2:7b使用時）

## 🔧 開発

### コード品質チェック
```bash
# フォーマット
uv run black src/ api/ --line-length 100

# リント
uv run ruff check src/ api/

# テスト
uv run pytest tests/ -v
```

### 負荷テスト
```bash
# 50人同時接続テスト
uv run python scripts/load_test.py

# システム監視
uv run python scripts/monitor_system.py
```

## 📜 ライセンス

MIT License - 商用利用・改変・再配布自由

## 🆘 サポート

- **Issues**: https://github.com/rikumiura32543/multi_chatbot/issues
- **Discussions**: GitHub Discussions
- **バグ報告**: Issue テンプレートを使用

## 🚀 ロードマップ

- [ ] OpenAI API対応
- [ ] より軽量なモデル対応
- [ ] Docker Compose対応
- [ ] 認証機能
- [ ] 多言語UI対応

---

**🔥 2024年8月 v0.1.0 - 初回リリース**

完全無料・ローカル動作・マルチチャットボット対応の汎用RAGシステムです。