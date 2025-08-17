{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "labor-qa-local";
  
  buildInputs = with pkgs; [
    # Python環境
    python311
    python311Packages.pip
    
    # uv - 超高速Pythonパッケージマネージャー
    uv
    
    # Ollama - ローカルLLM
    ollama
    
    # Podman - コンテナランタイム
    podman
    buildah
    
    # 開発ツール
    git
    direnv
    
    # システムライブラリ（ChromaDBとSentenceTransformers用）
    gcc
    pkg-config
    openssl
    libffi
    sqlite
    
    # オプション: デバッグ・監視ツール
    htop
    curl
    jq
  ];

  shellHook = ''
    echo "🚀 労務相談AIエージェント開発環境へようこそ!"
    echo "📁 プロジェクト: labor-qa-local"
    echo "🤖 LLMモデル: gpt-oss:20b"
    echo "🔧 ツール: Nix + uv + Podman"
    echo ""
    
    # Python環境の確認
    echo "🐍 Python: $(python --version)"
    echo "📦 uv: $(uv --version)"
    
    # Ollamaサービスの起動確認
    if ! pgrep -x "ollama" > /dev/null; then
      echo "🔄 Ollamaサービスを起動中..."
      ollama serve &
      sleep 2
    fi
    
    # gpt-oss:20bモデルの確認
    if ! ollama list | grep -q "gpt-oss:20b"; then
      echo "⚠️  gpt-oss:20bモデルが見つかりません"
      echo "💡 以下のコマンドでダウンロードしてください:"
      echo "   ollama pull gpt-oss:20b"
    else
      echo "✅ gpt-oss:20bモデル準備完了"
    fi
    
    # 必要なディレクトリの作成
    mkdir -p data/documents
    mkdir -p data/vectorstore
    mkdir -p logs
    
    # 環境変数の設定
    export PYTHONPATH="$PWD/src:$PYTHONPATH"
    export OLLAMA_HOST="127.0.0.1:11434"
    export STREAMLIT_SERVER_PORT="8501"
    
    echo ""
    echo "🛠️  開発コマンド:"
    echo "   uv sync              # 依存関係同期"
    echo "   uv run streamlit run app.py  # アプリ起動"
    echo "   uv run pytest       # テスト実行"
    echo "   podman build -t labor-qa-local .  # コンテナビルド"
    echo ""
  '';

  # 環境変数
  NIX_ENFORCE_PURITY = false;
  
  # ビルド時の環境変数
  PYTHONPATH = "./src";
  OLLAMA_MODELS = "./data/models";
}