#!/usr/bin/env python3
"""
マルチチャットボット対応汎用RAGシステム - FastAPIメインアプリケーション

Google Cloud E2インスタンス（2core/8GB）対応
qwen2:7bモデルを使用した軽量構成
"""

import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "config"))

from config.settings import Settings

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/server.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 設定読み込み
settings = Settings()
# LangSmith設定を環境変数に適用
settings.setup_logging()  # これによりLangSmith環境変数も設定される

# 必要なディレクトリ作成
def ensure_directories():
    """必要なディレクトリを作成"""
    directories = [
        "logs",
        "data/chatbots", 
        "data/uploads",
        "ui/static",
        "ui/templates"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info("必要なディレクトリを作成しました")

ensure_directories()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションライフサイクル管理"""
    logger.info("🚀 マルチチャットボットRAGシステム起動中...")
    logger.info(f"⚙️  設定: {settings.environment}環境")
    logger.info(f"🤖 LLMモデル: {settings.llm_model_name}")
    logger.info(f"🔗 Ollama: {settings.ollama_host}")
    
    # Ollamaサービス確認
    try:
        import requests
        response = requests.get(f"http://{settings.ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Ollamaサービス接続確認完了")
        else:
            logger.warning("⚠️  Ollamaサービスに接続できません")
    except Exception as e:
        logger.error(f"❌ Ollama接続エラー: {e}")
    
    yield
    
    logger.info("🛑 アプリケーション終了")

# FastAPIアプリケーション作成
app = FastAPI(
    title="汎用RAGチャットボット",
    description="マルチチャットボット対応のRAGシステム",
    version="0.1.0",
    lifespan=lifespan
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイルとテンプレート設定
app.mount("/static", StaticFiles(directory="ui/static"), name="static")
templates = Jinja2Templates(directory="ui/templates")

# エラーハンドリング用ユーティリティ関数
def is_embedded_context(request: Request) -> bool:
    """埋め込みコンテキストかどうかを判定"""
    # URLパスで判定
    if '/embed/' in request.url.path:
        return True
    
    # Refererで判定
    referer = request.headers.get('referer', '')
    if '/embed/' in referer:
        return True
    
    # User-Agentで判定（iframe内の場合）
    user_agent = request.headers.get('user-agent', '')
    
    # X-Requested-Withヘッダーで判定（Ajax/iframe）
    requested_with = request.headers.get('x-requested-with', '')
    
    return False

def get_error_context(request: Request, status_code: int, detail: str) -> dict:
    """エラーページのコンテキストを生成"""
    is_embedded = is_embedded_context(request)
    
    # ステータスコード別のタイトルとメッセージ
    error_info = {
        400: {"title": "不正なリクエストです", "message": "リクエストの形式が正しくありません。"},
        401: {"title": "認証が必要です", "message": "この機能を利用するには認証が必要です。"},
        403: {"title": "アクセスが拒否されました", "message": "この操作を実行する権限がありません。"},
        404: {"title": "ページが見つかりません", "message": "お探しのページは存在しないか、移動された可能性があります。"},
        429: {"title": "リクエストが多すぎます", "message": "しばらく時間をおいてから再度お試しください。"},
        500: {"title": "サーバーエラーが発生しました", "message": "内部サーバーエラーが発生しました。管理者にお問い合わせください。"},
        502: {"title": "サービスが利用できません", "message": "一時的にサービスが利用できません。しばらくお待ちください。"},
        503: {"title": "メンテナンス中です", "message": "現在システムメンテナンス中です。しばらくお待ちください。"}
    }
    
    error_data = error_info.get(status_code, {
        "title": "エラーが発生しました", 
        "message": "予期しないエラーが発生しました。"
    })
    
    # チャットボットIDの抽出（埋め込みコンテキストの場合）
    chatbot_id = None
    if is_embedded:
        path_parts = request.url.path.split('/')
        if 'embed' in path_parts:
            try:
                idx = path_parts.index('embed')
                if idx + 1 < len(path_parts):
                    chatbot_id = path_parts[idx + 1]
            except (ValueError, IndexError):
                pass
    
    return {
        "request": request,
        "title": error_data["title"],
        "message": error_data["message"],
        "details": detail if settings.environment == "development" else None,
        "error_code": status_code,
        "is_embedded": is_embedded,
        "chatbot_id": chatbot_id
    }

# HTTPエラーハンドラー
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTP例外のハンドラー"""
    logger.error(f"HTTP {exc.status_code} エラー: {exc.detail} - Path: {request.url.path}")
    
    # APIエンドポイントの場合はJSONレスポンス
    if request.url.path.startswith('/api/'):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "status_code": exc.status_code,
                "detail": exc.detail,
                "path": request.url.path
            }
        )
    
    # 通常のページの場合はHTMLエラーページ
    context = get_error_context(request, exc.status_code, exc.detail)
    return templates.TemplateResponse(
        "error.html",
        context,
        status_code=exc.status_code
    )

# バリデーションエラーハンドラー
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """リクエストバリデーションエラーのハンドラー"""
    error_detail = f"バリデーションエラー: {exc.errors()}"
    logger.error(f"Validation error: {error_detail} - Path: {request.url.path}")
    
    # APIエンドポイントの場合はJSONレスポンス
    if request.url.path.startswith('/api/'):
        return JSONResponse(
            status_code=422,
            content={
                "error": True,
                "status_code": 422,
                "detail": "入力データの形式が正しくありません",
                "validation_errors": exc.errors(),
                "path": request.url.path
            }
        )
    
    # 通常のページの場合はHTMLエラーページ
    context = get_error_context(request, 422, "入力データの形式が正しくありません")
    return templates.TemplateResponse(
        "error.html",
        context,
        status_code=422
    )

# 一般的な例外ハンドラー
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """一般的な例外のハンドラー"""
    error_detail = f"予期しないエラー: {str(exc)}"
    logger.error(f"Unexpected error: {error_detail} - Path: {request.url.path}", exc_info=True)
    
    # APIエンドポイントの場合はJSONレスポンス
    if request.url.path.startswith('/api/'):
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "status_code": 500,
                "detail": "内部サーバーエラーが発生しました",
                "path": request.url.path
            }
        )
    
    # 通常のページの場合はHTMLエラーページ
    context = get_error_context(request, 500, "内部サーバーエラーが発生しました")
    return templates.TemplateResponse(
        "error.html",
        context,
        status_code=500
    )

# ルートエンドポイント
@app.get("/")
async def root(request: Request):
    """チャットボット選択画面"""
    return templates.TemplateResponse(
        "chat_selection.html",
        {"request": request, "title": "チャットボット選択"}
    )

@app.get("/settings/{chatbot_id}")
async def rag_settings(request: Request, chatbot_id: str):
    """RAG設定画面"""
    return templates.TemplateResponse(
        "rag_settings.html",
        {"request": request, "chatbot_id": chatbot_id, "title": "RAG設定"}
    )

@app.get("/chat/{chatbot_id}")
async def chat_interface(request: Request, chatbot_id: str):
    """チャット画面"""
    return templates.TemplateResponse(
        "chat_interface.html",
        {"request": request, "chatbot_id": chatbot_id, "title": "チャット"}
    )

@app.get("/embed/{chatbot_id}")
async def embed_chat(request: Request, chatbot_id: str):
    """埋め込み用チャット画面"""
    return templates.TemplateResponse(
        "embed.html",
        {"request": request, "chatbot_id": chatbot_id, "title": "チャット"}
    )

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "llm_model": settings.llm_model_name,
        "version": "0.1.0"
    }

# APIルーターを追加
from api.chatbots import router as chatbots_router
from api.chat import router as chat_router  
from api.rag import router as rag_router

app.include_router(chatbots_router, prefix="/api/chatbots", tags=["chatbots"])
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(rag_router, prefix="/api/rag", tags=["rag"])

if __name__ == "__main__":
    import uvicorn
    
    # コマンドライン引数処理
    import argparse
    parser = argparse.ArgumentParser(description="マルチチャットボットRAGシステム")
    parser.add_argument("--host", default="127.0.0.1", help="ホストアドレス")
    parser.add_argument("--port", type=int, default=8000, help="ポート番号")
    parser.add_argument("--workers", type=int, default=1, help="ワーカー数")
    parser.add_argument("--debug", action="store_true", help="デバッグモード")
    args = parser.parse_args()
    
    logger.info(f"🌐 サーバー起動: http://{args.host}:{args.port}")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.debug,
        log_level="debug" if args.debug else "info"
    )