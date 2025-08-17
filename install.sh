#!/bin/bash
# マルチチャットボットRAGシステム - 自動セットアップスクリプト

set -e

echo "🤖 マルチチャットボットRAGシステム v0.1.0 セットアップ開始"

# 1. Nixの確認
if ! command -v nix &> /dev/null; then
    echo "❌ Nixが見つかりません。以下のコマンドでインストールしてください:"
    echo "sh <(curl -L https://nixos.org/nix/install) --daemon"
    exit 1
fi

echo "✅ Nix確認完了"

# 2. Nix環境セットアップ
echo "🔧 開発環境セットアップ中..."
nix-shell --run "echo 'Nix環境確認完了'"

# 3. 依存関係インストール
echo "📦 依存関係インストール中..."
nix-shell --run "uv sync"

# 4. 必要なディレクトリ作成
echo "📁 ディレクトリ作成中..."
mkdir -p data/chatbots data/uploads logs

# 5. Ollamaサービス起動とモデルダウンロード
echo "🤖 AIモデル準備中..."
nix-shell --run "ollama serve &"
sleep 5

if ! nix-shell --run "ollama list | grep -q qwen2:7b"; then
    echo "📥 qwen2:7bモデルダウンロード中（約6.5GB、時間がかかります）..."
    nix-shell --run "ollama pull qwen2:7b"
else
    echo "✅ qwen2:7bモデル確認完了"
fi

# 6. 権限設定
chmod +x setup-nix.sh scripts/*.py

echo ""
echo "🎉 セットアップ完了！"
echo ""
echo "🚀 起動方法:"
echo "  nix-shell"
echo "  uv run python main.py --host 0.0.0.0 --port 8000"
echo ""
echo "🌐 アクセス: http://localhost:8000"
echo ""
echo "📖 詳細: README.md をご確認ください"