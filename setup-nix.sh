#!/bin/bash

# Nix環境セットアップスクリプト
# このプロジェクトでNixを確実に使用できるようにする

echo "🔧 Nix環境セットアップ開始..."

# Nixプロファイルの読み込み
if [ -e /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh ]; then
    echo "📦 Nixプロファイルを読み込み中..."
    . /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
    echo "✅ Nixプロファイル読み込み完了"
else
    echo "❌ Nixプロファイルが見つかりません: /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh"
    exit 1
fi

# Nixが利用可能か確認
if command -v nix &> /dev/null; then
    echo "✅ Nix利用可能: $(which nix)"
    echo "📊 Nixバージョン: $(nix --version)"
else
    echo "❌ Nixコマンドが見つかりません"
    exit 1
fi

# shell.nixファイルの存在確認
if [ -f "shell.nix" ]; then
    echo "✅ shell.nixファイル存在確認完了"
else
    echo "❌ shell.nixファイルが見つかりません"
    exit 1
fi

echo "🚀 Nix環境セットアップ完了"
echo ""
echo "次のコマンドでNix環境に入れます:"
echo "  source setup-nix.sh && nix-shell"
echo ""
echo "または環境変数を設定してから実行:"
echo "  export NIX_PATH=/nix/var/nix/profiles/per-user/root/channels"
echo "  nix-shell"