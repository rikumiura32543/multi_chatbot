"""
チャットボット管理モジュール

マルチチャットボット（最大5個）の作成・削除・一覧表示機能
Google Cloud E2インスタンス対応の軽量実装
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging

from pydantic import BaseModel, Field
from config.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

class ChatbotInfo(BaseModel):
    """チャットボット情報"""
    id: str = Field(..., description="チャットボットID")
    name: str = Field(..., description="チャットボット名")
    description: str = Field(default="", description="説明")
    created_at: datetime = Field(default_factory=datetime.now, description="作成日時")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新日時")
    rag_configured: bool = Field(default=False, description="RAG設定完了フラグ")
    document_count: int = Field(default=0, description="処理済み文書数")
    folder_paths: List[str] = Field(default_factory=list, description="参照フォルダパス")

class ChatbotManager:
    """チャットボット管理クラス"""
    
    def __init__(self):
        self.chatbots_dir = Path(settings.data_dir) / "chatbots"
        self.config_file = self.chatbots_dir / "chatbots_config.json"
        self.max_chatbots = 5  # 最大チャットボット数
        
        # 必要なディレクトリ作成
        self.chatbots_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定ファイル初期化
        if not self.config_file.exists():
            self._save_config({})
    
    def _load_config(self) -> Dict[str, dict]:
        """設定ファイル読み込み"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー: {e}")
            return {}
    
    def _save_config(self, config: Dict[str, dict]):
        """設定ファイル保存"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"設定ファイル保存エラー: {e}")
            raise
    
    def list_chatbots(self) -> List[ChatbotInfo]:
        """チャットボット一覧取得"""
        config = self._load_config()
        chatbots = []
        
        for chatbot_id, data in config.items():
            try:
                chatbot = ChatbotInfo(**data)
                chatbots.append(chatbot)
            except Exception as e:
                logger.warning(f"チャットボット {chatbot_id} の読み込みエラー: {e}")
        
        # 作成日時順でソート
        chatbots.sort(key=lambda x: x.created_at, reverse=True)
        return chatbots
    
    def get_chatbot(self, chatbot_id: str) -> Optional[ChatbotInfo]:
        """特定のチャットボット情報取得"""
        config = self._load_config()
        
        if chatbot_id not in config:
            return None
        
        try:
            return ChatbotInfo(**config[chatbot_id])
        except Exception as e:
            logger.error(f"チャットボット {chatbot_id} の取得エラー: {e}")
            return None
    
    def create_chatbot(self, name: str, description: str = "") -> ChatbotInfo:
        """新規チャットボット作成"""
        config = self._load_config()
        
        # 最大数チェック
        if len(config) >= self.max_chatbots:
            raise ValueError(f"チャットボット数が上限（{self.max_chatbots}個）に達しています")
        
        # 同名チェック
        existing_names = [data.get("name", "") for data in config.values()]
        if name in existing_names:
            raise ValueError(f"チャットボット名 '{name}' は既に使用されています")
        
        # 新規チャットボット作成
        chatbot_id = f"bot_{uuid.uuid4().hex[:8]}"
        chatbot = ChatbotInfo(
            id=chatbot_id,
            name=name,
            description=description
        )
        
        # チャットボット専用ディレクトリ作成
        bot_dir = self.chatbots_dir / chatbot_id
        bot_dir.mkdir(exist_ok=True)
        (bot_dir / "documents").mkdir(exist_ok=True)
        (bot_dir / "vectorstore").mkdir(exist_ok=True)
        
        # 設定保存
        config[chatbot_id] = chatbot.model_dump()
        self._save_config(config)
        
        logger.info(f"新規チャットボット作成: {chatbot_id} - {name}")
        return chatbot
    
    def update_chatbot(self, chatbot_id: str, **updates) -> Optional[ChatbotInfo]:
        """チャットボット情報更新"""
        config = self._load_config()
        
        if chatbot_id not in config:
            return None
        
        # 更新データ適用
        config[chatbot_id].update(updates)
        config[chatbot_id]["updated_at"] = datetime.now().isoformat()
        
        # 名前重複チェック（名前更新時）
        if "name" in updates:
            other_names = [
                data.get("name", "") 
                for bid, data in config.items() 
                if bid != chatbot_id
            ]
            if updates["name"] in other_names:
                raise ValueError(f"チャットボット名 '{updates['name']}' は既に使用されています")
        
        self._save_config(config)
        
        try:
            return ChatbotInfo(**config[chatbot_id])
        except Exception as e:
            logger.error(f"チャットボット更新エラー: {e}")
            return None

    
    def update_chatbot_timestamp(self, chatbot_id: str) -> bool:
        """チャットボットのタイムスタンプを更新"""
        config = self._load_config()
        
        if chatbot_id not in config:
            return False
        
        config[chatbot_id]["updated_at"] = datetime.now().isoformat()
        self._save_config(config)
        
        logger.info(f"チャットボットタイムスタンプ更新: {chatbot_id}")
        return True
    
    def delete_chatbot(self, chatbot_id: str) -> bool:
        """チャットボット削除"""
        config = self._load_config()
        
        if chatbot_id not in config:
            return False
        
        # ファイル削除
        bot_dir = self.chatbots_dir / chatbot_id
        if bot_dir.exists():
            import shutil
            try:
                shutil.rmtree(bot_dir)
                logger.info(f"チャットボットディレクトリ削除: {bot_dir}")
            except Exception as e:
                logger.warning(f"ディレクトリ削除エラー: {e}")
        
        # 設定から削除
        chatbot_name = config[chatbot_id].get("name", "Unknown")
        del config[chatbot_id]
        self._save_config(config)
        
        logger.info(f"チャットボット削除: {chatbot_id} - {chatbot_name}")
        return True
    
    def get_chatbot_dir(self, chatbot_id: str) -> Path:
        """チャットボット専用ディレクトリパス取得"""
        return self.chatbots_dir / chatbot_id
    
    def get_vectorstore_dir(self, chatbot_id: str) -> Path:
        """ベクトルストアディレクトリパス取得"""
        return self.get_chatbot_dir(chatbot_id) / "vectorstore"
    
    def get_documents_dir(self, chatbot_id: str) -> Path:
        """文書ディレクトリパス取得"""
        return self.get_chatbot_dir(chatbot_id) / "documents"

# グローバルインスタンス
chatbot_manager = ChatbotManager()

def get_chatbot_manager() -> ChatbotManager:
    """チャットボット管理インスタンス取得"""
    return chatbot_manager