"""
チャット機能APIルーター

会話管理、メッセージ送受信、履歴管理のエンドポイント
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

# プロジェクト内モジュール
from src.chatbot_manager import get_chatbot_manager, ChatbotManager
from src.vector_store import VectorStoreManager
from src.llm_manager import LLMManager
from config.settings import get_settings, Settings

logger = logging.getLogger(__name__)
router = APIRouter()

# リクエスト/レスポンスモデル
class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    timestamp: Optional[str] = None  # 文字列として受け付ける
    rag_sources: Optional[List[str]] = None

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = []
    use_rag: bool = True
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

class ChatResponse(BaseModel):
    response: str
    rag_sources: Optional[List[str]] = None
    rag_context_used: bool = False
    model_used: str
    response_time: float
    timestamp: datetime

class ConversationHistory(BaseModel):
    chatbot_id: str
    messages: List[ChatMessage]
    total_messages: int
    last_updated: datetime

@router.post("/{chatbot_id}/message", response_model=ChatResponse)
async def send_message(
    chatbot_id: str,
    request: ChatRequest,
    chatbot_manager: ChatbotManager = Depends(get_chatbot_manager),
    settings: Settings = Depends(get_settings)
):
    """メッセージ送信とAI応答生成"""
    start_time = datetime.now()
    logger.info(f"チャットリクエスト受信 [{chatbot_id}]: {request.message[:50]}...")
    
    # デバッグ用：リクエスト詳細をログ出力
    logger.debug(f"リクエスト詳細 [{chatbot_id}]: message_length={len(request.message)}, history_count={len(request.conversation_history) if request.conversation_history else 0}, use_rag={request.use_rag}")
    
    try:
        # チャットボット存在確認
        chatbot = chatbot_manager.get_chatbot(chatbot_id)
        if not chatbot:
            logger.warning(f"チャットボットが見つかりません [{chatbot_id}]")
            raise HTTPException(status_code=404, detail="チャットボットが見つかりません")
        
        # LLMマネージャー初期化
        llm_manager = LLMManager()
        
        # RAG機能による関連文書検索
        rag_context = ""
        rag_sources = []
        rag_context_used = False
        
        if request.use_rag and chatbot.rag_configured:
            try:
                vector_store = VectorStoreManager(chatbot_id)
                # ハイブリッド検索（ベクトル検索 + キーワード検索）を使用
                search_results = vector_store.hybrid_search(
                    request.message, 
                    k=min(10, settings.similarity_search_k * 3),  # 最大10件まで検索
                    vector_weight=0.6,  # ベクトル検索の重み
                    keyword_weight=0.4   # キーワード検索の重み
                )
                
                if search_results:
                    # スコアをログ出力（詳細）
                    logger.info(f"RAG検索結果 [{chatbot_id}]: {len(search_results)}件")
                    for i, (doc, score) in enumerate(search_results[:5]):
                        logger.info(f"  結果{i+1}: スコア={score:.4f}, ファイル={doc.metadata.get('source', 'unknown').split('/')[-1]}, 内容={doc.page_content[:100]}...")
                    
                    # DEBUGレベルでログ設定を確実に有効化
                    import logging
                    vector_logger = logging.getLogger('src.vector_store')
                    vector_logger.setLevel(logging.DEBUG)
                    # コンソールハンドラーも追加してDEBUGログが確実に出力されるようにする
                    if not any(isinstance(h, logging.StreamHandler) for h in vector_logger.handlers):
                        handler = logging.StreamHandler()
                        handler.setLevel(logging.DEBUG)
                        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                        handler.setFormatter(formatter)
                        vector_logger.addHandler(handler)
                    
                    # 関連度の高い文書のみ使用（ハイブリッド検索では統合スコア）
                    # デバッグ用に閾値を一時的に低くして、まず検索結果が表示されるかを確認
                    relevant_docs = [(doc, score) for doc, score in search_results if score > 0.0]
                    logger.info(f"関連度フィルター後 [{chatbot_id}]: {len(relevant_docs)}件（閾値: >0.0）")
                    
                    if relevant_docs:
                        rag_context_used = True
                        rag_context = "\n\n".join([
                            f"【参考文書: {doc.metadata.get('source', 'unknown')}】\n{doc.page_content}"
                            for doc, _ in relevant_docs[:3]  # 上位3件まで
                        ])
                        # 重複を除いたファイル名リストを作成
                        unique_sources = set()
                        for doc, _ in relevant_docs:
                            filename = doc.metadata.get('source', 'unknown').split('/')[-1]
                            unique_sources.add(filename)
                        rag_sources = list(unique_sources)
                        
                        logger.info(f"RAG検索完了 [{chatbot_id}]: {len(relevant_docs)}件の関連文書を発見")
                    else:
                        logger.info(f"RAG検索結果 [{chatbot_id}]: 関連度の高い文書が見つかりませんでした")
            except Exception as e:
                logger.warning(f"RAG検索エラー [{chatbot_id}]: {e}")
                # RAGエラーでも会話は続行
        
        # 会話履歴の構築
        conversation_context = []
        if request.conversation_history:
            # 直近10回の会話のみ使用（メモリ最適化）
            recent_history = request.conversation_history[-10:]
            for msg in recent_history:
                conversation_context.append(f"{msg.role}: {msg.content}")
            logger.debug(f"会話履歴 [{chatbot_id}]: {len(conversation_context)}件の履歴を使用")
        
        # プロンプト生成
        if rag_context:
            system_prompt = f"""あなたは文書に基づいて回答する専門アシスタントです。
チャットボット名: {chatbot.name}
説明: {chatbot.description}

【重要な回答ルール】
1. 提供された参考文書の内容のみを使用して回答してください
2. 参考文書に記載されていない情報については、推測や一般知識で補完せず「参考文書に記載されていません」と明確に伝えてください
3. 参考文書の内容を正確に引用し、どの文書からの情報かを明示してください
4. 日本語で丁寧かつ簡潔に回答してください
5. 質問に関連する情報が参考文書にない場合は、「ご質問に関する情報は参考文書に含まれておりません」と回答してください

【参考文書】
{rag_context}"""
        else:
            system_prompt = f"""あなたは文書に基づいて回答する専門アシスタントです。
チャットボット名: {chatbot.name}
説明: {chatbot.description}

現在、ご質問に関連する参考文書が見つかりませんでした。
申し訳ございませんが、参考文書に基づいてのみ回答するため、この質問にはお答えできません。
より具体的な質問をしていただくか、関連する文書が登録されているかご確認ください。"""

        # LLM応答生成
        try:
            logger.info(f"LLM応答生成開始 [{chatbot_id}]: model={llm_manager.model_name}")
            response_text = llm_manager.generate_response(
                prompt=request.message,
                context=system_prompt,
                conversation_history=conversation_context,
                max_tokens=request.max_tokens or settings.max_tokens,
                temperature=request.temperature or 0.7
            )
            logger.info(f"LLM応答生成完了 [{chatbot_id}]: response_length={len(response_text)}")
        except Exception as e:
            logger.error(f"LLM応答生成エラー [{chatbot_id}]: {e}")
            raise HTTPException(status_code=500, detail="AI応答の生成に失敗しました")
        
        # 応答時間計算
        response_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"チャット応答完了 [{chatbot_id}]: {response_time:.2f}s, RAG使用: {rag_context_used}")
        
        return ChatResponse(
            response=response_text,
            rag_sources=rag_sources if rag_sources else None,
            rag_context_used=rag_context_used,
            model_used=llm_manager.model_name,
            response_time=response_time,
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"チャットメッセージ処理エラー [{chatbot_id}]: {e}")
        raise HTTPException(status_code=500, detail="メッセージの処理に失敗しました")

@router.get("/{chatbot_id}/history", response_model=ConversationHistory)
async def get_conversation_history(
    chatbot_id: str,
    limit: int = 50,
    chatbot_manager: ChatbotManager = Depends(get_chatbot_manager)
):
    """会話履歴取得"""
    try:
        # チャットボット存在確認
        chatbot = chatbot_manager.get_chatbot(chatbot_id)
        if not chatbot:
            raise HTTPException(status_code=404, detail="チャットボットが見つかりません")
        
        # 会話履歴ファイル読み込み
        history_file = chatbot_manager.get_chatbot_dir(chatbot_id) / "conversation_history.json"
        
        if history_file.exists():
            import json
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            messages = [ChatMessage(**msg) for msg in history_data.get('messages', [])]
            # 最新の指定件数のみ返す
            recent_messages = messages[-limit:] if len(messages) > limit else messages
            
            return ConversationHistory(
                chatbot_id=chatbot_id,
                messages=recent_messages,
                total_messages=len(messages),
                last_updated=datetime.fromisoformat(history_data.get('last_updated', datetime.now().isoformat()))
            )
        else:
            # 履歴がない場合は空を返す
            return ConversationHistory(
                chatbot_id=chatbot_id,
                messages=[],
                total_messages=0,
                last_updated=datetime.now()
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"会話履歴取得エラー [{chatbot_id}]: {e}")
        raise HTTPException(status_code=500, detail="会話履歴の取得に失敗しました")

@router.post("/{chatbot_id}/history/save")
async def save_conversation_history(
    chatbot_id: str,
    messages: List[ChatMessage],
    chatbot_manager: ChatbotManager = Depends(get_chatbot_manager)
):
    """会話履歴保存"""
    try:
        # チャットボット存在確認
        chatbot = chatbot_manager.get_chatbot(chatbot_id)
        if not chatbot:
            raise HTTPException(status_code=404, detail="チャットボットが見つかりません")
        
        # 会話履歴ファイル保存
        history_file = chatbot_manager.get_chatbot_dir(chatbot_id) / "conversation_history.json"
        
        history_data = {
            "chatbot_id": chatbot_id,
            "messages": [msg.model_dump() for msg in messages],
            "total_messages": len(messages),
            "last_updated": datetime.now().isoformat()
        }
        
        import json
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"会話履歴保存完了 [{chatbot_id}]: {len(messages)}件")
        return {"success": True, "message": f"会話履歴を保存しました（{len(messages)}件）"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"会話履歴保存エラー [{chatbot_id}]: {e}")
        raise HTTPException(status_code=500, detail="会話履歴の保存に失敗しました")

@router.delete("/{chatbot_id}/history")
async def clear_conversation_history(
    chatbot_id: str,
    chatbot_manager: ChatbotManager = Depends(get_chatbot_manager)
):
    """会話履歴クリア"""
    try:
        # チャットボット存在確認
        chatbot = chatbot_manager.get_chatbot(chatbot_id)
        if not chatbot:
            raise HTTPException(status_code=404, detail="チャットボットが見つかりません")
        
        # 会話履歴ファイル削除
        history_file = chatbot_manager.get_chatbot_dir(chatbot_id) / "conversation_history.json"
        
        if history_file.exists():
            history_file.unlink()
        
        logger.info(f"会話履歴クリア完了 [{chatbot_id}]")
        return {"success": True, "message": "会話履歴をクリアしました"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"会話履歴クリアエラー [{chatbot_id}]: {e}")
        raise HTTPException(status_code=500, detail="会話履歴のクリアに失敗しました")

@router.get("/{chatbot_id}/status")
async def get_chat_status(
    chatbot_id: str,
    chatbot_manager: ChatbotManager = Depends(get_chatbot_manager)
):
    """チャット状態取得"""
    try:
        # チャットボット存在確認
        chatbot = chatbot_manager.get_chatbot(chatbot_id)
        if not chatbot:
            raise HTTPException(status_code=404, detail="チャットボットが見つかりません")
        
        # LLMサービス状態確認
        llm_manager = LLMManager()
        llm_available = llm_manager.health_check()
        
        # RAG状態確認
        rag_available = False
        if chatbot.rag_configured:
            try:
                vector_store = VectorStoreManager(chatbot_id)
                rag_available = vector_store.health_check()
            except:
                rag_available = False
        
        # 会話履歴件数
        history_file = chatbot_manager.get_chatbot_dir(chatbot_id) / "conversation_history.json"
        conversation_count = 0
        if history_file.exists():
            import json
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
                conversation_count = len(history_data.get('messages', []))
        
        return {
            "chatbot_id": chatbot_id,
            "chatbot_name": chatbot.name,
            "llm_available": llm_available,
            "rag_available": rag_available,
            "rag_configured": chatbot.rag_configured,
            "conversation_count": conversation_count,
            "status": "ready" if llm_available else "error"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"チャット状態取得エラー [{chatbot_id}]: {e}")
        raise HTTPException(status_code=500, detail="チャット状態の取得に失敗しました")