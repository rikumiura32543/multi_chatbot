"""
マルチチャットボット対応汎用RAGシステム - 設定管理

アプリケーション全体の設定を管理するモジュール
環境変数の読み込み、バリデーション、デフォルト値の設定を行う
Google Cloud E2インスタンス対応、qwen2:7b最適化
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

class Settings(BaseSettings):
    """アプリケーション設定クラス"""
    
    # プロジェクト基本設定
    project_name: str = "マルチチャットボット対応汎用RAGシステム"
    version: str = "0.1.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Ollama LLM設定（qwen2:7b - 8GB RAM最適化）
    ollama_host: str = Field(default="127.0.0.1:11434", env="OLLAMA_HOST")
    llm_model_name: str = Field(default="qwen2:7b", env="LLM_MODEL_NAME")
    ollama_timeout: int = Field(default=300, env="OLLAMA_TIMEOUT")
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    
    # ベクトルストア設定（マルチチャットボット対応）
    data_dir: str = Field(default="./data", env="DATA_DIR")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    persist_directory: str = Field(default="./data/vectorstore", env="PERSIST_DIRECTORY")
    
    # FastAPI設定
    server_host: str = Field(default="127.0.0.1", env="SERVER_HOST")
    server_port: int = Field(default=8000, env="SERVER_PORT")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    max_chatbots: int = Field(default=5, env="MAX_CHATBOTS")
    
    # ログ設定
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_dir: str = Field(default="./logs", env="LOG_DIR")
    
    # RAG設定
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    similarity_search_k: int = Field(default=3, env="SIMILARITY_SEARCH_K")
    max_tokens: int = Field(default=4000, env="MAX_TOKENS")
    
    # UI設定（iframe埋め込み対応）
    enable_file_upload: bool = Field(default=True, env="ENABLE_FILE_UPLOAD")
    max_file_size_mb: int = Field(default=10, env="MAX_FILE_SIZE_MB")
    supported_file_types: List[str] = Field(default=["pdf", "md", "txt", "docx"])
    enable_iframe_embed: bool = Field(default=True, env="ENABLE_IFRAME_EMBED")
    allowed_embed_domains: List[str] = Field(default=["*"], env="ALLOWED_EMBED_DOMAINS")
    
    # デバッグ設定
    langchain_tracing_v2: bool = Field(default=False)
    ollama_debug: bool = Field(default=False)
    
    # LangSmith設定
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="labor-chatbot-rag", env="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", env="LANGSMITH_ENDPOINT")
    enable_langsmith: bool = Field(default=False, env="ENABLE_LANGSMITH")
    
    @field_validator("supported_file_types", mode="before")
    @classmethod
    def parse_file_types(cls, v):
        """サポートされるファイルタイプをパース"""
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """CORS設定をパース"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("allowed_embed_domains", mode="before")
    @classmethod
    def parse_embed_domains(cls, v):
        """埋め込み許可ドメインをパース"""
        if isinstance(v, str):
            return [domain.strip() for domain in v.split(",")]
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """ログレベルのバリデーション"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"ログレベルは {valid_levels} のいずれかを指定してください")
        return v.upper()
    
    def setup_logging(self) -> None:
        """ログ設定のセットアップ"""
        # ログディレクトリの作成
        log_path = Path(self.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # ログフォーマットの設定
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        log_file = log_path / "labor_qa.log"
        
        # ログ設定
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        # LangChainトレーシングの設定
        if self.langchain_tracing_v2:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            
        # LangSmithトレーシングの設定
        if self.enable_langsmith and self.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
            os.environ["LANGCHAIN_ENDPOINT"] = self.langsmith_endpoint
            logger = logging.getLogger(__name__)
            logger.info(f"LangSmith設定を環境変数に設定完了: プロジェクト={self.langsmith_project}")
        else:
            logger = logging.getLogger(__name__)
            logger.info(f"LangSmith無効: enable_langsmith={self.enable_langsmith}, has_api_key={bool(self.langsmith_api_key)}")
    
    def get_ollama_url(self) -> str:
        """OllamaのURL取得"""
        return f"http://{self.ollama_host}"
    
    def ensure_directories(self) -> None:
        """必要なディレクトリの作成"""
        directories = [
            self.data_dir,
            f"{self.data_dir}/chatbots",
            f"{self.data_dir}/uploads",
            self.log_dir,
            "ui/static",
            "ui/templates"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_max_file_size_bytes(self) -> int:
        """最大ファイルサイズをバイト単位で取得"""
        return self.max_file_size_mb * 1024 * 1024
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # 古いフィールドを無視

# グローバル設定インスタンス
settings = Settings()

def get_settings() -> Settings:
    """設定インスタンスの取得"""
    return settings

def setup_environment() -> Settings:
    """環境のセットアップ"""
    settings.ensure_directories()
    settings.setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 {settings.project_name} v{settings.version} 起動中...")
    logger.info(f"🌍 環境: {settings.labor_qa_env}")
    logger.info(f"🤖 LLMモデル: {settings.ollama_model}")
    logger.info(f"📁 ベクトルストア: {settings.vector_store_dir}")
    logger.info(f"📄 ドキュメント: {settings.documents_dir}")
    
    return settings

# 環境初期化時の自動セットアップ
if __name__ == "__main__":
    setup_environment()
    print("✅ 設定の初期化が完了しました")