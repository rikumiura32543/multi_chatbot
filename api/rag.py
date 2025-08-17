"""
RAG機能APIルーター

文書アップロード、ベクトルストア管理、検索機能のエンドポイント
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# プロジェクト内モジュール
from src.chatbot_manager import get_chatbot_manager, ChatbotManager
from src.vector_store import VectorStoreManager
from src.document_loader import DocumentLoader
from config.settings import get_settings, Settings

logger = logging.getLogger(__name__)
router = APIRouter()

# レスポンスモデル
class RagStatsResponse(BaseModel):
    chatbot_id: str
    total_chunks: int
    unique_files: int
    file_type_distribution: Dict[str, int]
    vector_store_size: int
    last_updated: Optional[str] = None
    is_rebuilding: bool = False

class SearchRequest(BaseModel):
    query: str
    k: int = 3

class SearchResult(BaseModel):
    content: str
    source: str
    score: float

class RagSettingsRequest(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    search_k: int = 3

class FolderSetupRequest(BaseModel):
    folder_paths: List[str]
    chunk_size: int = 1000
    chunk_overlap: int = 200

class UploadResponse(BaseModel):
    success: bool
    filename: str
    chunks_added: int
    message: str

@router.get("/{chatbot_id}/stats", response_model=RagStatsResponse)
async def get_rag_stats(
    chatbot_id: str,
    chatbot_manager: ChatbotManager = Depends(get_chatbot_manager),
    settings: Settings = Depends(get_settings)
):
    """RAG統計情報の取得"""
    try:
        # チャットボット存在確認
        chatbot = chatbot_manager.get_chatbot(chatbot_id)
        if not chatbot:
            raise HTTPException(status_code=404, detail="チャットボットが見つかりません")
        
        # ベクトルストア統計取得
        vector_store = VectorStoreManager(chatbot_id)
        stats = vector_store.get_collection_stats()
        
        # ベクトルストアサイズ計算
        vectorstore_dir = chatbot_manager.get_vectorstore_dir(chatbot_id)
        vector_store_size = 0
        if vectorstore_dir.exists():
            for file_path in vectorstore_dir.rglob("*"):
                if file_path.is_file():
                    vector_store_size += file_path.stat().st_size
        
        # 再構築状況確認（フラグファイルの存在確認）
        rebuild_flag_file = chatbot_manager.get_chatbot_dir(chatbot_id) / ".rebuilding"
        is_rebuilding = rebuild_flag_file.exists()
        
        return RagStatsResponse(
            chatbot_id=chatbot_id,
            total_chunks=stats.get('total_chunks', 0),
            unique_files=stats.get('unique_files', 0),
            file_type_distribution=stats.get('file_type_distribution', {}),
            vector_store_size=vector_store_size,
            last_updated=chatbot.updated_at.isoformat() if chatbot.updated_at else None,
            is_rebuilding=is_rebuilding
        )
        
    except Exception as e:
        logger.error(f"RAG統計取得エラー [{chatbot_id}]: {e}")
        raise HTTPException(status_code=500, detail="統計情報の取得に失敗しました")

@router.post("/{chatbot_id}/upload", response_model=UploadResponse)
async def upload_document(
    chatbot_id: str,
    file: UploadFile = File(...),
    chatbot_manager: ChatbotManager = Depends(get_chatbot_manager),
    settings: Settings = Depends(get_settings)
):
    """文書アップロードとベクトルストア追加"""
    try:
        # チャットボット存在確認
        chatbot = chatbot_manager.get_chatbot(chatbot_id)
        if not chatbot:
            raise HTTPException(status_code=404, detail="チャットボットが見つかりません")
        
        # ファイル形式チェック
        file_extension = Path(file.filename).suffix.lower().lstrip('.')
        if file_extension not in settings.supported_file_types:
            raise HTTPException(
                status_code=400, 
                detail=f"サポートされていないファイル形式です。対応形式: {', '.join(settings.supported_file_types)}"
            )
        
        # ファイルサイズチェック
        content = await file.read()
        if len(content) > settings.get_max_file_size_bytes():
            raise HTTPException(
                status_code=400,
                detail=f"ファイルサイズが上限（{settings.max_file_size_mb}MB）を超えています"
            )
        
        # ファイル保存
        documents_dir = chatbot_manager.get_documents_dir(chatbot_id)
        documents_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = documents_dir / file.filename
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # 文書読み込みとベクトル化
        document_loader = DocumentLoader()
        documents = document_loader.load_documents(str(documents_dir))
        
        # 新しいファイルのみ処理
        new_documents = [doc for doc in documents if doc.metadata.get('source', '').endswith(file.filename)]
        
        if not new_documents:
            raise HTTPException(status_code=400, detail="文書の読み込みに失敗しました")
        
        # ベクトルストアに追加
        vector_store = VectorStoreManager(chatbot_id)
        document_ids = vector_store.add_documents(new_documents)
        
        # チャットボット情報更新
        chatbot_manager.update_chatbot(
            chatbot_id,
            rag_configured=True,
            document_count=chatbot.document_count + 1
        )
        
        logger.info(f"文書アップロード成功 [{chatbot_id}]: {file.filename} - {len(document_ids)}チャンク")
        
        return UploadResponse(
            success=True,
            filename=file.filename,
            chunks_added=len(document_ids),
            message=f"'{file.filename}' を正常にアップロードし、{len(document_ids)}個のチャンクに分割しました"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文書アップロードエラー [{chatbot_id}]: {e}")
        raise HTTPException(status_code=500, detail="文書のアップロードに失敗しました")

@router.post("/{chatbot_id}/search")
async def search_documents(
    chatbot_id: str,
    request: SearchRequest,
    chatbot_manager: ChatbotManager = Depends(get_chatbot_manager)
):
    """文書検索"""
    try:
        # チャットボット存在確認
        chatbot = chatbot_manager.get_chatbot(chatbot_id)
        if not chatbot:
            raise HTTPException(status_code=404, detail="チャットボットが見つかりません")
        
        # ベクトルストア検索
        vector_store = VectorStoreManager(chatbot_id)
        results = vector_store.similarity_search_with_score(request.query, k=request.k)
        
        # 結果フォーマット
        search_results = []
        for doc, score in results:
            search_results.append(SearchResult(
                content=doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                source=doc.metadata.get('source', 'unknown'),
                score=float(score)
            ))
        
        logger.info(f"文書検索完了 [{chatbot_id}]: '{request.query}' - {len(search_results)}件")
        return search_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文書検索エラー [{chatbot_id}]: {e}")
        raise HTTPException(status_code=500, detail="文書検索に失敗しました")

@router.post("/{chatbot_id}/settings")
async def save_rag_settings(
    chatbot_id: str,
    settings_request: RagSettingsRequest,
    chatbot_manager: ChatbotManager = Depends(get_chatbot_manager)
):
    """RAG設定保存"""
    try:
        # チャットボット存在確認
        chatbot = chatbot_manager.get_chatbot(chatbot_id)
        if not chatbot:
            raise HTTPException(status_code=404, detail="チャットボットが見つかりません")
        
        # 設定ファイル保存（チャットボット専用ディレクトリ）
        settings_file = chatbot_manager.get_chatbot_dir(chatbot_id) / "rag_settings.json"
        
        import json
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings_request.model_dump(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"RAG設定保存 [{chatbot_id}]: {settings_request.model_dump()}")
        return {"success": True, "message": "RAG設定を保存しました"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG設定保存エラー [{chatbot_id}]: {e}")
        raise HTTPException(status_code=500, detail="RAG設定の保存に失敗しました")

@router.post("/{chatbot_id}/rebuild")
async def rebuild_vector_store(
    chatbot_id: str,
    chatbot_manager: ChatbotManager = Depends(get_chatbot_manager)
):
    """ベクトルストア再構築"""
    try:
        # チャットボット存在確認
        chatbot = chatbot_manager.get_chatbot(chatbot_id)
        if not chatbot:
            raise HTTPException(status_code=404, detail="チャットボットが見つかりません")
        
        # 再構築フラグファイル作成
        rebuild_flag_file = chatbot_manager.get_chatbot_dir(chatbot_id) / ".rebuilding"
        rebuild_flag_file.touch()
        
        logger.info(f"ベクトルストア再構築開始 [{chatbot_id}]")
        
        try:
            # 既存ベクトルストアクリア
            vector_store = VectorStoreManager(chatbot_id)
            vector_store.clear_collection()
            
            # 全文書再読み込み
            documents_dir = chatbot_manager.get_documents_dir(chatbot_id)
            documents_count = 0
            
            if documents_dir.exists():
                document_loader = DocumentLoader()
                documents = document_loader.load_documents(str(documents_dir))
                
                if documents:
                    # ベクトルストア再構築
                    document_ids = vector_store.add_documents(documents)
                    documents_count = len(document_ids)
                    
                    logger.info(f"ベクトルストア再構築完了 [{chatbot_id}]: {documents_count}チャンク")
            
            # 再構築完了後フラグファイル削除
            if rebuild_flag_file.exists():
                rebuild_flag_file.unlink()
                
            # チャットボット更新日時を更新
            chatbot_manager.update_chatbot_timestamp(chatbot_id)
            
            return {
                "success": True, 
                "message": f"ベクトルストアを再構築しました（{documents_count}チャンク）"
            }
            
        except Exception as rebuild_error:
            # エラー時もフラグファイルを削除
            if rebuild_flag_file.exists():
                rebuild_flag_file.unlink()
            raise rebuild_error
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ベクトルストア再構築エラー [{chatbot_id}]: {e}")
        # エラー時フラグファイル削除
        rebuild_flag_file = chatbot_manager.get_chatbot_dir(chatbot_id) / ".rebuilding"
        if rebuild_flag_file.exists():
            rebuild_flag_file.unlink()
        raise HTTPException(status_code=500, detail="ベクトルストアの再構築に失敗しました")

@router.delete("/{chatbot_id}/clear")
async def clear_vector_store(
    chatbot_id: str,
    chatbot_manager: ChatbotManager = Depends(get_chatbot_manager)
):
    """ベクトルストアクリア"""
    try:
        # チャットボット存在確認
        chatbot = chatbot_manager.get_chatbot(chatbot_id)
        if not chatbot:
            raise HTTPException(status_code=404, detail="チャットボットが見つかりません")
        
        # ベクトルストアクリア
        vector_store = VectorStoreManager(chatbot_id)
        vector_store.clear_collection()
        
        # チャットボット情報更新
        chatbot_manager.update_chatbot(
            chatbot_id,
            rag_configured=False,
            document_count=0
        )
        
        logger.info(f"ベクトルストアクリア完了 [{chatbot_id}]")
        return {"success": True, "message": "ベクトルストアをクリアしました"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ベクトルストアクリアエラー [{chatbot_id}]: {e}")
        raise HTTPException(status_code=500, detail="ベクトルストアのクリアに失敗しました")

@router.get("/{chatbot_id}/settings")
async def get_rag_settings(
    chatbot_id: str,
    chatbot_manager: ChatbotManager = Depends(get_chatbot_manager)
):
    """RAG設定取得"""
    try:
        # チャットボット存在確認
        chatbot = chatbot_manager.get_chatbot(chatbot_id)
        if not chatbot:
            raise HTTPException(status_code=404, detail="チャットボットが見つかりません")
        
        # 設定ファイル読み込み
        settings_file = chatbot_manager.get_chatbot_dir(chatbot_id) / "rag_settings.json"
        
        if settings_file.exists():
            import json
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings_data = json.load(f)
            return settings_data
        else:
            # デフォルト設定を返す
            return {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "search_k": 3
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG設定取得エラー [{chatbot_id}]: {e}")
        raise HTTPException(status_code=500, detail="RAG設定の取得に失敗しました")

@router.post("/{chatbot_id}/setup-from-folders")
async def setup_rag_from_folders(
    chatbot_id: str,
    request: FolderSetupRequest,
    chatbot_manager: ChatbotManager = Depends(get_chatbot_manager),
    settings: Settings = Depends(get_settings)
):
    """フォルダ指定でRAGセットアップ"""
    try:
        # チャットボット存在確認
        chatbot = chatbot_manager.get_chatbot(chatbot_id)
        if not chatbot:
            raise HTTPException(status_code=404, detail="チャットボットが見つかりません")
        
        # フォルダパス検証
        valid_folders = []
        for folder_path in request.folder_paths:
            folder = Path(folder_path)
            if not folder.exists():
                logger.warning(f"フォルダが存在しません: {folder_path}")
                continue
            if not folder.is_dir():
                logger.warning(f"ディレクトリではありません: {folder_path}")
                continue
            valid_folders.append(folder)
        
        if not valid_folders:
            raise HTTPException(status_code=400, detail="有効なフォルダが指定されていません")
        
        # 再構築フラグファイル作成
        rebuild_flag_file = chatbot_manager.get_chatbot_dir(chatbot_id) / ".rebuilding"
        rebuild_flag_file.touch()
        
        logger.info(f"フォルダ指定RAGセットアップ開始 [{chatbot_id}]: {len(valid_folders)}フォルダ")
        
        # ドキュメントローダー初期化
        document_loader = DocumentLoader()
        all_documents = []
        processed_files = []
        
        # 各フォルダから文書を読み込み
        for folder in valid_folders:
            logger.info(f"フォルダ処理中: {folder}")
            
            # フォルダ内のファイル一覧取得（サブフォルダは除外）
            supported_extensions = settings.supported_file_types  # ['.pdf', '.docx', '.md', '.txt']
            folder_files = []
            
            for file_path in folder.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    folder_files.append(file_path)
            
            if not folder_files:
                logger.warning(f"フォルダに対応ファイルがありません: {folder}")
                continue
            
            logger.info(f"フォルダ内ファイル数: {len(folder_files)}")
            
            # ファイルを文書として読み込み
            for file_path in folder_files:
                try:
                    # 個別ファイルとしてドキュメント読み込み
                    docs = document_loader.load_documents(str(file_path))
                    if docs:
                        all_documents.extend(docs)
                        processed_files.append({
                            'file': str(file_path),
                            'folder': str(folder),
                            'chunks': len(docs)
                        })
                        logger.debug(f"ファイル処理完了: {file_path.name} ({len(docs)}チャンク)")
                except Exception as e:
                    logger.error(f"ファイル読み込みエラー {file_path}: {e}")
                    continue
        
        if not all_documents:
            # フラグファイル削除
            if rebuild_flag_file.exists():
                rebuild_flag_file.unlink()
            raise HTTPException(status_code=400, detail="処理可能な文書が見つかりませんでした")
        
        # ベクトルストア管理
        vector_store = VectorStoreManager(chatbot_id)
        
        # 既存ベクトルストアをクリア
        vector_store.clear_collection()
        
        # 新しい文書をベクトルストアに追加
        document_ids = vector_store.add_documents(all_documents)
        
        # チャットボット情報更新
        chatbot_manager.update_chatbot(
            chatbot_id,
            rag_configured=True,
            document_count=len(processed_files)
        )
        
        # RAG設定保存
        settings_data = {
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
            "search_k": 3,
            "folder_paths": [str(f) for f in valid_folders],
            "processed_files": processed_files,
            "setup_method": "folders"
        }
        
        settings_file = chatbot_manager.get_chatbot_dir(chatbot_id) / "rag_settings.json"
        import json
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings_data, f, ensure_ascii=False, indent=2)
        
        # フラグファイル削除
        if rebuild_flag_file.exists():
            rebuild_flag_file.unlink()
        
        logger.info(f"フォルダ指定RAGセットアップ完了 [{chatbot_id}]: {len(processed_files)}ファイル, {len(document_ids)}チャンク")
        
        return {
            "success": True,
            "message": f"フォルダからRAGを設定しました",
            "processed_folders": len(valid_folders),
            "processed_files": len(processed_files),
            "total_chunks": len(document_ids),
            "files": processed_files
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"フォルダ指定RAGセットアップエラー [{chatbot_id}]: {e}")
        
        # エラー時フラグファイル削除
        rebuild_flag_file = chatbot_manager.get_chatbot_dir(chatbot_id) / ".rebuilding"
        if rebuild_flag_file.exists():
            rebuild_flag_file.unlink()
            
        raise HTTPException(status_code=500, detail=f"フォルダ指定RAGセットアップに失敗しました: {str(e)}")

@router.get("/{chatbot_id}/available-folders")
async def get_available_folders(
    chatbot_id: str,
    search_path: str = "/Users",
    chatbot_manager: ChatbotManager = Depends(get_chatbot_manager)
):
    """利用可能なフォルダ一覧取得（ブラウザ機能）"""
    try:
        # チャットボット存在確認
        chatbot = chatbot_manager.get_chatbot(chatbot_id)
        if not chatbot:
            raise HTTPException(status_code=404, detail="チャットボットが見つかりません")
        
        search_dir = Path(search_path)
        if not search_dir.exists() or not search_dir.is_dir():
            raise HTTPException(status_code=400, detail="無効なパスです")
        
        folders = []
        files_count = {}
        
        try:
            # ディレクトリ内のフォルダ一覧取得
            for item in search_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # フォルダ内の対応ファイル数をカウント
                    supported_extensions = ['.pdf', '.docx', '.md', '.txt']
                    file_count = 0
                    
                    try:
                        for file_item in item.iterdir():
                            if file_item.is_file() and file_item.suffix.lower() in supported_extensions:
                                file_count += 1
                    except PermissionError:
                        continue  # アクセス権限がない場合はスキップ
                    
                    folders.append({
                        'name': item.name,
                        'path': str(item),
                        'file_count': file_count
                    })
        
        except PermissionError:
            raise HTTPException(status_code=403, detail="フォルダアクセス権限がありません")
        
        # ファイル数でソート（多い順）
        folders.sort(key=lambda x: x['file_count'], reverse=True)
        
        return {
            "current_path": str(search_dir),
            "parent_path": str(search_dir.parent) if search_dir.parent != search_dir else None,
            "folders": folders[:50]  # 最大50フォルダまで
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"フォルダ一覧取得エラー: {e}")
        raise HTTPException(status_code=500, detail="フォルダ一覧の取得に失敗しました")

@router.post("/{chatbot_id}/upload-batch")
async def upload_batch_files(
    chatbot_id: str,
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    chatbot_manager: ChatbotManager = Depends(get_chatbot_manager)
):
    """バッチファイルアップロード（ローカルフォルダ選択用）"""
    try:
        # チャットボット存在確認
        chatbot = chatbot_manager.get_chatbot(chatbot_id)
        if not chatbot:
            raise HTTPException(status_code=404, detail="チャットボットが見つかりません")
        
        if not files:
            raise HTTPException(status_code=400, detail="ファイルが選択されていません")
        
        # 再構築フラグファイル作成
        rebuild_flag_file = chatbot_manager.get_chatbot_dir(chatbot_id) / ".rebuilding"
        rebuild_flag_file.touch()
        
        logger.info(f"バッチアップロード開始 [{chatbot_id}]: {len(files)}ファイル")
        
        # ドキュメントローダー初期化
        document_loader = DocumentLoader()
        all_documents = []
        processed_files = []
        
        # 一時保存ディレクトリ
        temp_dir = chatbot_manager.get_documents_dir(chatbot_id) / "temp_upload"
        temp_dir.mkdir(exist_ok=True)
        
        # 恒久的な文書ディレクトリも確保
        documents_dir = chatbot_manager.get_documents_dir(chatbot_id)
        documents_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 各ファイルを処理
            for file in files:
                # ファイル拡張子チェック
                file_extension = Path(file.filename).suffix.lower()
                if file_extension not in ['.pdf', '.docx', '.md', '.txt']:
                    logger.warning(f"非対応ファイル形式: {file.filename}")
                    continue
                
                # 一時ファイルとして保存
                temp_file_path = temp_dir / file.filename
                # 恒久的なファイルパス
                permanent_file_path = documents_dir / file.filename
                
                try:
                    # ファイル内容を読み取り
                    content = await file.read()
                    if not content:
                        logger.warning(f"空ファイル: {file.filename}")
                        continue
                    
                    # 一時ファイルに保存
                    with open(temp_file_path, 'wb') as temp_file:
                        temp_file.write(content)
                    
                    # 恒久的なファイルにも保存（再構築用）
                    with open(permanent_file_path, 'wb') as perm_file:
                        perm_file.write(content)
                    
                    # ドキュメントとして読み込み（単一ファイル用メソッドを使用）
                    docs = document_loader.load_single_document(str(temp_file_path))
                    
                    if docs:
                        all_documents.extend(docs)
                        processed_files.append({
                            'filename': file.filename,
                            'size': len(content),
                            'chunks': len(docs),
                            'type': file_extension
                        })
                        logger.debug(f"ファイル処理完了: {file.filename} ({len(docs)}チャンク)")
                    
                except Exception as e:
                    logger.error(f"ファイル処理エラー {file.filename}: {e}")
                    continue
                finally:
                    # 一時ファイル削除
                    if temp_file_path.exists():
                        temp_file_path.unlink()
            
            if not all_documents:
                raise HTTPException(status_code=400, detail="処理可能なファイルがありませんでした")
            
            # ベクトルストア管理
            vector_store = VectorStoreManager(chatbot_id)
            
            # 既存ベクトルストアをクリア
            vector_store.clear_collection()
            
            # 新しい文書をベクトルストアに追加
            document_ids = vector_store.add_documents(all_documents)
            
            # チャットボット情報更新
            chatbot_manager.update_chatbot(
                chatbot_id,
                rag_configured=True,
                document_count=len(processed_files)
            )
            
            # RAG設定保存
            settings_data = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "search_k": 3,
                "processed_files": processed_files,
                "setup_method": "batch_upload",
                "upload_source": "local_folders"
            }
            
            settings_file = chatbot_manager.get_chatbot_dir(chatbot_id) / "rag_settings.json"
            import json
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"バッチアップロード完了 [{chatbot_id}]: {len(processed_files)}ファイル, {len(document_ids)}チャンク")
            
            return {
                "success": True,
                "message": "ローカルフォルダからRAGを設定しました",
                "processed_files": len(processed_files),
                "total_chunks": len(document_ids),
                "files": processed_files
            }
            
        finally:
            # 一時ディレクトリクリーンアップ
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"バッチアップロードエラー [{chatbot_id}]: {e}")
        raise HTTPException(status_code=500, detail=f"バッチアップロードに失敗しました: {str(e)}")
    finally:
        # フラグファイル削除
        rebuild_flag_file = chatbot_manager.get_chatbot_dir(chatbot_id) / ".rebuilding"
        if rebuild_flag_file.exists():
            rebuild_flag_file.unlink()