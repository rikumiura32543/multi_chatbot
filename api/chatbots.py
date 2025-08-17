"""
チャットボット管理API

マルチチャットボット（最大5個）のCRUD操作を提供
Google Cloud E2インスタンス対応の軽量実装
"""

import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

# プロジェクトモジュールのインポート
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from chatbot_manager import ChatbotManager, ChatbotInfo, get_chatbot_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# リクエスト/レスポンスモデル
class CreateChatbotRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50, description="チャットボット名")
    description: str = Field(default="", max_length=200, description="説明")

class UpdateChatbotRequest(BaseModel):
    name: str = Field(None, min_length=1, max_length=50, description="チャットボット名")
    description: str = Field(None, max_length=200, description="説明")
    rag_configured: bool = Field(None, description="RAG設定状態")
    document_count: int = Field(None, description="処理済み文書数")
    folder_paths: List[str] = Field(None, description="参照フォルダパス")

class ChatbotResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    rag_configured: bool
    document_count: int
    folder_paths: List[str]

@router.get("/list", response_model=List[ChatbotResponse])
async def list_chatbots():
    """チャットボット一覧取得"""
    try:
        manager = get_chatbot_manager()
        chatbots = manager.list_chatbots()
        
        # レスポンス形式に変換
        response = []
        for chatbot in chatbots:
            response.append(ChatbotResponse(
                id=chatbot.id,
                name=chatbot.name,
                description=chatbot.description,
                created_at=chatbot.created_at.isoformat(),
                updated_at=chatbot.updated_at.isoformat(),
                rag_configured=chatbot.rag_configured,
                document_count=chatbot.document_count,
                folder_paths=chatbot.folder_paths
            ))
        
        logger.info(f"チャットボット一覧取得: {len(response)}件")
        return response
        
    except Exception as e:
        logger.error(f"チャットボット一覧取得エラー: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="チャットボット一覧の取得に失敗しました"
        )

@router.get("/{chatbot_id}", response_model=ChatbotResponse)
async def get_chatbot(chatbot_id: str):
    """特定のチャットボット情報取得"""
    try:
        manager = get_chatbot_manager()
        chatbot = manager.get_chatbot(chatbot_id)
        
        if not chatbot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"チャットボット {chatbot_id} が見つかりません"
            )
        
        response = ChatbotResponse(
            id=chatbot.id,
            name=chatbot.name,
            description=chatbot.description,
            created_at=chatbot.created_at.isoformat(),
            updated_at=chatbot.updated_at.isoformat(),
            rag_configured=chatbot.rag_configured,
            document_count=chatbot.document_count,
            folder_paths=chatbot.folder_paths
        )
        
        logger.info(f"チャットボット取得: {chatbot_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"チャットボット取得エラー: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="チャットボット情報の取得に失敗しました"
        )

@router.post("/create", response_model=ChatbotResponse, status_code=status.HTTP_201_CREATED)
async def create_chatbot(request: CreateChatbotRequest):
    """新規チャットボット作成"""
    try:
        manager = get_chatbot_manager()
        
        # 最大数チェック（5個制限）
        existing_chatbots = manager.list_chatbots()
        if len(existing_chatbots) >= 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="チャットボット数が上限（5個）に達しています"
            )
        
        # 同名チェック
        existing_names = [bot.name for bot in existing_chatbots]
        if request.name in existing_names:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"チャットボット名 '{request.name}' は既に使用されています"
            )
        
        # チャットボット作成
        chatbot = manager.create_chatbot(
            name=request.name,
            description=request.description
        )
        
        response = ChatbotResponse(
            id=chatbot.id,
            name=chatbot.name,
            description=chatbot.description,
            created_at=chatbot.created_at.isoformat(),
            updated_at=chatbot.updated_at.isoformat(),
            rag_configured=chatbot.rag_configured,
            document_count=chatbot.document_count,
            folder_paths=chatbot.folder_paths
        )
        
        logger.info(f"チャットボット作成成功: {chatbot.id} - {chatbot.name}")
        return response
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"チャットボット作成エラー: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="チャットボットの作成に失敗しました"
        )

@router.put("/{chatbot_id}", response_model=ChatbotResponse)
async def update_chatbot(chatbot_id: str, request: UpdateChatbotRequest):
    """チャットボット情報更新"""
    try:
        manager = get_chatbot_manager()
        
        # 存在チェック
        existing_chatbot = manager.get_chatbot(chatbot_id)
        if not existing_chatbot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"チャットボット {chatbot_id} が見つかりません"
            )
        
        # 更新データ準備
        update_data = {}
        if request.name is not None:
            update_data["name"] = request.name
        if request.description is not None:
            update_data["description"] = request.description
        if request.rag_configured is not None:
            update_data["rag_configured"] = request.rag_configured
        if request.document_count is not None:
            update_data["document_count"] = request.document_count
        if request.folder_paths is not None:
            update_data["folder_paths"] = request.folder_paths
        
        # 更新実行
        updated_chatbot = manager.update_chatbot(chatbot_id, **update_data)
        
        if not updated_chatbot:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="チャットボットの更新に失敗しました"
            )
        
        response = ChatbotResponse(
            id=updated_chatbot.id,
            name=updated_chatbot.name,
            description=updated_chatbot.description,
            created_at=updated_chatbot.created_at.isoformat(),
            updated_at=updated_chatbot.updated_at.isoformat(),
            rag_configured=updated_chatbot.rag_configured,
            document_count=updated_chatbot.document_count,
            folder_paths=updated_chatbot.folder_paths
        )
        
        logger.info(f"チャットボット更新成功: {chatbot_id}")
        return response
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"チャットボット更新エラー: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="チャットボットの更新に失敗しました"
        )

@router.delete("/{chatbot_id}")
async def delete_chatbot(chatbot_id: str):
    """チャットボット削除"""
    try:
        manager = get_chatbot_manager()
        
        # 存在チェック
        existing_chatbot = manager.get_chatbot(chatbot_id)
        if not existing_chatbot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"チャットボット {chatbot_id} が見つかりません"
            )
        
        # 削除実行
        success = manager.delete_chatbot(chatbot_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="チャットボットの削除に失敗しました"
            )
        
        logger.info(f"チャットボット削除成功: {chatbot_id} - {existing_chatbot.name}")
        return {"message": f"チャットボット '{existing_chatbot.name}' を削除しました"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"チャットボット削除エラー: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="チャットボットの削除に失敗しました"
        )

@router.get("/{chatbot_id}/stats")
async def get_chatbot_stats(chatbot_id: str):
    """チャットボット統計情報取得"""
    try:
        manager = get_chatbot_manager()
        
        # 存在チェック
        chatbot = manager.get_chatbot(chatbot_id)
        if not chatbot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"チャットボット {chatbot_id} が見つかりません"
            )
        
        # ディレクトリサイズ計算
        bot_dir = manager.get_chatbot_dir(chatbot_id)
        vectorstore_dir = manager.get_vectorstore_dir(chatbot_id)
        documents_dir = manager.get_documents_dir(chatbot_id)
        
        def get_dir_size(path):
            if not path.exists():
                return 0
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        
        stats = {
            "chatbot_id": chatbot_id,
            "name": chatbot.name,
            "rag_configured": chatbot.rag_configured,
            "document_count": chatbot.document_count,
            "folder_count": len(chatbot.folder_paths),
            "total_size_bytes": get_dir_size(bot_dir),
            "vectorstore_size_bytes": get_dir_size(vectorstore_dir),
            "documents_size_bytes": get_dir_size(documents_dir),
            "created_at": chatbot.created_at.isoformat(),
            "updated_at": chatbot.updated_at.isoformat()
        }
        
        logger.info(f"チャットボット統計取得: {chatbot_id}")
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"チャットボット統計取得エラー: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="統計情報の取得に失敗しました"
        )