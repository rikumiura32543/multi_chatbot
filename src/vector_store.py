"""
マルチチャットボット対応ベクトルストア管理モジュール

ChromaDBを使用したマルチチャットボット対応ベクトルストア
Google Cloud E2インスタンス（8GB RAM）最適化

機能:
- チャットボット別コレクション管理
- 文書の埋め込み化と保存
- セマンティック検索実行  
- ベクトルストアの永続化
- 文書の更新・削除
- 軽量化とメモリ最適化
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import hashlib

# ChromaDB
import chromadb
from chromadb.config import Settings as ChromaSettings

# LangChain
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

# プロジェクト設定
from config.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


class ChromaDBEmbeddings(Embeddings):
    """ChromaDBの内蔵埋め込み機能を使用するラッパークラス（軽量化）"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """文書の埋め込みを生成（ChromaDBが内部で実行）"""
        # ChromaDBが内部で処理するため、ダミーの埋め込みを返す
        return [[0.0] * 384 for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """クエリの埋め込みを生成（ChromaDBが内部で実行）"""
        # ChromaDBが内部で処理するため、ダミーの埋め込みを返す
        return [0.0] * 384


class VectorStoreManager:
    """マルチチャットボット対応ベクトルストア管理クラス"""
    
    def __init__(self, chatbot_id: str, persist_directory: Optional[str] = None, embedding_model: Optional[str] = None):
        """
        初期化
        
        Args:
            chatbot_id: チャットボットID（必須）
            persist_directory: ベクトルストアの永続化ディレクトリ
            embedding_model: 埋め込みモデル名
        """
        self.chatbot_id = chatbot_id
        self.persist_directory = persist_directory or settings.persist_directory
        self.embedding_model_name = embedding_model or settings.embedding_model
        self.collection_name = f"chatbot_{chatbot_id}"  # チャットボット別コレクション
        
        # チャットボット専用ディレクトリの作成
        self.chatbot_persist_dir = Path(self.persist_directory) / chatbot_id
        self.chatbot_persist_dir.mkdir(parents=True, exist_ok=True)
        
        # 埋め込みモデルの初期化
        self._initialize_embeddings()
        
        # ChromaDBクライアントの初期化
        self._initialize_chroma_client()
        
        # LangChainのChromaベクトルストアの初期化
        self._initialize_vectorstore()
        
        logger.info(f"VectorStoreManager初期化完了 - チャットボット: {self.chatbot_id}, ディレクトリ: {self.chatbot_persist_dir}")
    
    def _initialize_embeddings(self):
        """埋め込みモデルの初期化"""
        try:
            self.embeddings = ChromaDBEmbeddings(
                model_name=self.embedding_model_name
            )
            logger.info(f"埋め込みモデル初期化完了: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"埋め込みモデル初期化エラー: {e}")
            raise
    
    def _initialize_chroma_client(self):
        """ChromaDBクライアントの初期化"""
        try:
            # ChromaDB設定（チャットボット専用）
            self.client = chromadb.PersistentClient(
                path=str(self.chatbot_persist_dir),
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("ChromaDBクライアント初期化完了")
        except Exception as e:
            logger.error(f"ChromaDBクライアント初期化エラー: {e}")
            raise
    
    def _initialize_vectorstore(self):
        """ChromaDBコレクションの初期化"""
        try:
            # コレクションの取得または作成（チャットボット専用）
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": f"チャットボット {self.chatbot_id} 専用文書コレクション",
                    "chatbot_id": self.chatbot_id,
                    "created_by": "multi_chatbot_rag_system"
                }
            )
            logger.info(f"ベクトルストア初期化完了 - コレクション: {self.collection_name}")
        except Exception as e:
            logger.error(f"ベクトルストア初期化エラー: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        文書をベクトルストアに追加
        
        Args:
            documents: 追加する文書のリスト
            
        Returns:
            List[str]: 追加された文書のIDリスト
        """
        if not documents:
            logger.warning("追加する文書がありません")
            return []
        
        try:
            # 文書IDの生成（チャットボットID + ファイルパス + チャンクインデックスのハッシュ）
            document_ids = []
            document_texts = []
            metadatas = []
            
            for doc in documents:
                source = doc.metadata.get('source', 'unknown')
                chunk_index = doc.metadata.get('chunk_index', 0)
                doc_id = self._generate_document_id(source, chunk_index)
                
                # メタデータにチャットボットIDを追加
                doc_metadata = doc.metadata.copy()
                doc_metadata['chatbot_id'] = self.chatbot_id
                
                document_ids.append(doc_id)
                document_texts.append(doc.page_content)
                metadatas.append(doc_metadata)
            
            # ChromaDBに追加
            self.collection.add(
                documents=document_texts,
                metadatas=metadatas,
                ids=document_ids
            )
            
            logger.info(f"文書追加完了 [chatbot: {self.chatbot_id}]: {len(document_ids)}件")
            return document_ids
            
        except Exception as e:
            logger.error(f"文書追加エラー [chatbot: {self.chatbot_id}]: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = None, filter_dict: Optional[Dict] = None) -> List[Document]:
        """
        セマンティック検索の実行
        
        Args:
            query: 検索クエリ
            k: 取得する文書数
            filter_dict: メタデータフィルタ
            
        Returns:
            List[Document]: 関連文書のリスト
        """
        if k is None:
            k = settings.similarity_search_k
        
        try:
            # チャットボットIDフィルタを自動追加
            where_filter = {"chatbot_id": self.chatbot_id}
            if filter_dict:
                where_filter.update(filter_dict)
            
            # ChromaDBで検索実行
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where_filter
            )
            
            # 結果をLangChain Documentオブジェクトに変換
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, doc_text in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    doc = Document(
                        page_content=doc_text,
                        metadata=metadata
                    )
                    documents.append(doc)
            
            logger.info(f"検索完了 [chatbot: {self.chatbot_id}]: クエリ='{query[:50]}...', 結果数={len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"検索エラー [chatbot: {self.chatbot_id}]: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = None) -> List[tuple]:
        """
        スコア付きセマンティック検索
        
        Args:
            query: 検索クエリ
            k: 取得する文書数
            
        Returns:
            List[tuple]: (Document, score)のタプルリスト
        """
        if k is None:
            k = settings.similarity_search_k
        
        try:
            # ChromaDBで検索実行（距離付き）
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where={"chatbot_id": self.chatbot_id},
                include=['documents', 'metadatas', 'distances']
            )
            
            # 結果をタプルに変換
            documents_with_scores = []
            if results['documents'] and results['documents'][0]:
                for i, doc_text in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.0
                    
                    doc = Document(
                        page_content=doc_text,
                        metadata=metadata
                    )
                    # 距離をスコアに変換（距離が小さいほど類似度が高い）
                    # ChromaDBの距離の実際の値をログで確認
                    logger.debug(f"ベクトル検索 [{self.chatbot_id}]: 距離={distance:.6f}, 内容={doc_text[:50]}...")
                    
                    # ChromaDBの距離は通常0.0-2.0の範囲で、0に近いほど類似度が高い
                    # ただし、実際の距離範囲を確認してから適切に変換
                    if distance <= 1.0:
                        score = 1.0 - distance  # 0.0-1.0の範囲
                    else:
                        score = max(0.0, 2.0 - distance) / 2.0  # 1.0-2.0を0.5-0.0にマッピング
                    
                    documents_with_scores.append((doc, score))
            
            logger.info(f"スコア付き検索完了 [chatbot: {self.chatbot_id}]: クエリ='{query[:50]}...', 結果数={len(documents_with_scores)}")
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"スコア付き検索エラー [chatbot: {self.chatbot_id}]: {e}")
            return []
    
    def keyword_search(self, query: str, k: int = None) -> List[tuple]:
        """
        キーワード検索（全文検索）
        
        Args:
            query: 検索クエリ
            k: 取得する文書数
            
        Returns:
            List[tuple]: (Document, score)のタプルリスト
        """
        if k is None:
            k = settings.similarity_search_k
        
        try:
            # 全ての文書を取得
            results = self.collection.get(
                where={"chatbot_id": self.chatbot_id},
                include=['documents', 'metadatas']
            )
            
            documents_with_scores = []
            if results['documents']:
                # キーワードマッチングを実行
                query_lower = query.lower()
                query_terms = query_lower.split()
                logger.debug(f"キーワード検索 [{self.chatbot_id}]: クエリ単語={query_terms}")
                
                for i, doc_text in enumerate(results['documents']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    doc_text_lower = doc_text.lower()
                    
                    # スコア計算：マッチした単語数とその頻度
                    score = 0.0
                    matched_terms = []
                    for term in query_terms:
                        term_count = doc_text_lower.count(term)
                        if term_count > 0:
                            score += term_count * (len(term) / len(query))  # 長い単語により重みを付ける
                            matched_terms.append(f"{term}({term_count})")
                    
                    # スコアが0より大きい場合のみ追加
                    if score > 0:
                        doc = Document(
                            page_content=doc_text,
                            metadata=metadata
                        )
                        documents_with_scores.append((doc, score))
                        logger.debug(f"キーワードマッチ [{self.chatbot_id}]: スコア={score:.4f}, マッチ={matched_terms}, 内容={doc_text[:100]}...")
                    elif len(query_terms) <= 3:  # デバッグ：短いクエリの場合は詳細ログ
                        logger.debug(f"キーワードマッチなし [{self.chatbot_id}]: 内容={doc_text[:100]}...")
                
                # スコアでソート（降順）
                documents_with_scores.sort(key=lambda x: x[1], reverse=True)
                documents_with_scores = documents_with_scores[:k]
            
            logger.info(f"キーワード検索完了 [chatbot: {self.chatbot_id}]: クエリ='{query[:50]}...', 結果数={len(documents_with_scores)}")
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"キーワード検索エラー [chatbot: {self.chatbot_id}]: {e}")
            return []
    
    def hybrid_search(self, query: str, k: int = None, vector_weight: float = 0.7, keyword_weight: float = 0.3) -> List[tuple]:
        """
        ハイブリッド検索（ベクトル検索 + キーワード検索）
        
        Args:
            query: 検索クエリ
            k: 取得する文書数
            vector_weight: ベクトル検索の重み
            keyword_weight: キーワード検索の重み
            
        Returns:
            List[tuple]: (Document, combined_score)のタプルリスト
        """
        if k is None:
            k = settings.similarity_search_k
        
        # より多くの候補を取得してから結合
        extended_k = min(k * 3, 20)
        
        # ベクトル検索実行
        vector_results = self.similarity_search_with_score(query, extended_k)
        
        # キーワード検索実行
        keyword_results = self.keyword_search(query, extended_k)
        
        # 結果を結合してスコアを統合
        combined_results = {}
        
        # ベクトル検索結果を追加（距離ベースのスコア）
        for doc, vector_score in vector_results:
            doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
            combined_results[doc_id] = {
                'document': doc,
                'vector_score': vector_score,
                'keyword_score': 0.0
            }
        
        # キーワード検索結果を追加/更新
        for doc, keyword_score in keyword_results:
            doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
            if doc_id in combined_results:
                combined_results[doc_id]['keyword_score'] = keyword_score
            else:
                combined_results[doc_id] = {
                    'document': doc,
                    'vector_score': 0.0,
                    'keyword_score': keyword_score
                }
        
        # 統合スコア計算とソート
        final_results = []
        for doc_info in combined_results.values():
            # 正規化されたスコア（0-1の範囲）
            normalized_vector_score = min(doc_info['vector_score'], 1.0)
            max_keyword_score = max([r[1] for r in keyword_results]) if keyword_results else 1.0
            normalized_keyword_score = doc_info['keyword_score'] / max_keyword_score if max_keyword_score > 0 else 0.0
            
            # 統合スコア計算
            combined_score = (vector_weight * normalized_vector_score) + (keyword_weight * normalized_keyword_score)
            final_results.append((doc_info['document'], combined_score))
        
        # スコアでソート（降順）
        final_results.sort(key=lambda x: x[1], reverse=True)
        final_results = final_results[:k]
        
        logger.info(f"ハイブリッド検索完了 [chatbot: {self.chatbot_id}]: クエリ='{query[:50]}...', 結果数={len(final_results)}, ベクトル結果={len(vector_results)}, キーワード結果={len(keyword_results)}")
        return final_results
    
    def update_documents(self, documents: List[Document]) -> None:
        """
        文書の更新（既存文書を削除して再追加）
        
        Args:
            documents: 更新する文書のリスト
        """
        try:
            # 既存文書の削除
            sources = [doc.metadata.get('source') for doc in documents]
            unique_sources = list(set(sources))
            
            for source in unique_sources:
                if source:
                    self.delete_documents_by_source(source)
            
            # 新しい文書の追加
            self.add_documents(documents)
            
            logger.info(f"文書更新完了 [chatbot: {self.chatbot_id}]: {len(unique_sources)}ファイル, {len(documents)}チャンク")
            
        except Exception as e:
            logger.error(f"文書更新エラー [chatbot: {self.chatbot_id}]: {e}")
            raise
    
    def delete_documents_by_source(self, source_path: str) -> None:
        """
        指定ソースの文書を削除
        
        Args:
            source_path: 削除する文書のソースパス
        """
        try:
            # メタデータでフィルタして削除（チャットボットIDも考慮）
            results = self.collection.get(
                where={
                    "source": source_path,
                    "chatbot_id": self.chatbot_id
                }
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"文書削除完了 [chatbot: {self.chatbot_id}]: {source_path} ({len(results['ids'])}チャンク)")
            else:
                logger.info(f"削除対象文書が見つかりません [chatbot: {self.chatbot_id}]: {source_path}")
                
        except Exception as e:
            logger.error(f"文書削除エラー [chatbot: {self.chatbot_id}] {source_path}: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        コレクション統計の取得
        
        Returns:
            Dict[str, Any]: 統計情報
        """
        try:
            count = self.collection.count()
            
            # メタデータ統計
            if count > 0:
                results = self.collection.get(
                    where={"chatbot_id": self.chatbot_id},
                    include=['metadatas']
                )
                if results['metadatas']:
                    sources = [meta.get('source', 'unknown') for meta in results['metadatas']]
                    unique_sources = len(set(sources))
                    file_types = [meta.get('file_type', 'unknown') for meta in results['metadatas']]
                    type_counts = {}
                    for file_type in file_types:
                        type_counts[file_type] = type_counts.get(file_type, 0) + 1
                else:
                    unique_sources = 0
                    type_counts = {}
            else:
                unique_sources = 0
                type_counts = {}
            
            stats = {
                'chatbot_id': self.chatbot_id,
                'total_chunks': count,
                'unique_files': unique_sources,
                'file_type_distribution': type_counts,
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model_name,
                'persist_directory': str(self.chatbot_persist_dir)
            }
            
            logger.info(f"コレクション統計 [chatbot: {self.chatbot_id}]: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"統計取得エラー [chatbot: {self.chatbot_id}]: {e}")
            return {}
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], limit: int = 100) -> List[Document]:
        """
        メタデータによる検索
        
        Args:
            metadata_filter: メタデータフィルタ
            limit: 取得上限数
            
        Returns:
            List[Document]: 該当文書のリスト
        """
        try:
            # チャットボットIDフィルタを自動追加
            where_filter = {"chatbot_id": self.chatbot_id}
            where_filter.update(metadata_filter)
            
            results = self.collection.get(
                where=where_filter,
                limit=limit,
                include=['metadatas', 'documents']
            )
            
            documents = []
            if results['documents']:
                for i, doc_text in enumerate(results['documents']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    doc = Document(
                        page_content=doc_text,
                        metadata=metadata
                    )
                    documents.append(doc)
            
            logger.info(f"メタデータ検索完了 [chatbot: {self.chatbot_id}]: フィルタ={metadata_filter}, 結果数={len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"メタデータ検索エラー [chatbot: {self.chatbot_id}]: {e}")
            return []
    
    def clear_collection(self) -> None:
        """コレクション内の全文書を削除"""
        try:
            # コレクションを削除して再作成
            self.client.delete_collection(self.collection_name)
            self._initialize_vectorstore()
            
            logger.info(f"コレクションクリア完了 [chatbot: {self.chatbot_id}]")
            
        except Exception as e:
            logger.error(f"コレクションクリアエラー [chatbot: {self.chatbot_id}]: {e}")
    
    def _generate_document_id(self, source: str, chunk_index: int) -> str:
        """文書IDの生成（チャットボットID含む）"""
        # チャットボットID + ソースパス + チャンクインデックスのハッシュ値を使用
        content = f"{self.chatbot_id}#{source}#{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def health_check(self) -> bool:
        """ヘルスチェック"""
        try:
            # コレクションの存在確認
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                logger.warning(f"コレクションが存在しません [chatbot: {self.chatbot_id}]: {self.collection_name}")
                return False
            
            # 簡単なクエリテスト
            self.collection.query(
                query_texts=["test"], 
                n_results=1,
                where={"chatbot_id": self.chatbot_id}
            )
            
            logger.info(f"ヘルスチェック成功 [chatbot: {self.chatbot_id}]")
            return True
            
        except Exception as e:
            logger.error(f"ヘルスチェック失敗 [chatbot: {self.chatbot_id}]: {e}")
            return False


def create_vector_store(chatbot_id: str, persist_directory: Optional[str] = None) -> VectorStoreManager:
    """
    ベクトルストアの作成・初期化
    
    Args:
        chatbot_id: チャットボットID
        persist_directory: 永続化ディレクトリ
        
    Returns:
        VectorStoreManager: 初期化済みベクトルストア
    """
    return VectorStoreManager(chatbot_id=chatbot_id, persist_directory=persist_directory)


def search_documents(chatbot_id: str, query: str, vector_store: Optional[VectorStoreManager] = None, k: int = 3) -> List[Document]:
    """
    文書検索のヘルパー関数
    
    Args:
        chatbot_id: チャットボットID
        query: 検索クエリ
        vector_store: ベクトルストア（Noneの場合は新規作成）
        k: 取得文書数
        
    Returns:
        List[Document]: 検索結果
    """
    if vector_store is None:
        vector_store = create_vector_store(chatbot_id)
    
    return vector_store.similarity_search(query, k=k)


if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    
    # テスト用チャットボットIDでベクトルストアの作成
    test_chatbot_id = "test_bot_001"
    vs = create_vector_store(test_chatbot_id)
    
    # 統計情報の表示
    stats = vs.get_collection_stats()
    print(f"✅ ベクトルストア統計: {stats}")
    
    # ヘルスチェック
    if vs.health_check():
        print(f"✅ ベクトルストア [chatbot: {test_chatbot_id}] は正常に動作しています")
    else:
        print(f"⚠️ ベクトルストア [chatbot: {test_chatbot_id}] に問題があります")