"""
LLM管理モジュール - qwen2:7b対応Ollama連携

Google Cloud E2インスタンス（8GB RAM）最適化
マルチチャットボット対応の軽量LLM推論機能
"""

import json
import logging
import asyncio
from typing import Optional, Dict, Any, List
import requests
from datetime import datetime

from config.settings import Settings

# LangSmith用
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    # デコレータのダミー実装
    def traceable(name=None):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)
settings = Settings()

class LLMManager:
    """LLM管理クラス - qwen2:7b特化"""
    
    def __init__(self):
        self.model_name = settings.llm_model_name
        self.ollama_host = settings.ollama_host
        self.base_url = f"http://{self.ollama_host}"
        self.timeout = settings.ollama_timeout
        
        # qwen2:7b最適化設定
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 2048,
            "stop": ["Human:", "Assistant:", "\n\n---"],
            "stream": False
        }
        
        logger.info(f"LLM Manager初期化: {self.model_name} @ {self.base_url}")
    
    def check_model_availability(self) -> bool:
        """qwen2:7bモデルの利用可能性チェック"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                
                is_available = any(
                    self.model_name in model for model in available_models
                )
                
                if is_available:
                    logger.info(f"✅ {self.model_name} モデル利用可能")
                else:
                    logger.warning(f"⚠️  {self.model_name} モデルが見つかりません")
                    logger.info(f"利用可能モデル: {available_models}")
                
                return is_available
                
        except Exception as e:
            logger.error(f"❌ Ollama接続エラー: {e}")
            return False
    
    @traceable(name="llm_generate_response")
    def generate_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        conversation_history: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """qwen2:7bを使用した応答生成（LangSmithトレーシング対応）"""
        
        # LangSmithトレーシング用メタデータ
        trace_metadata = {
            "model": self.model_name,
            "has_context": bool(context),
            "has_history": bool(conversation_history),
            "prompt_length": len(prompt)
        }
        
        if LANGSMITH_AVAILABLE and settings.enable_langsmith:
            logger.info(f"LangSmithトレーシング有効 - プロジェクト: {settings.langsmith_project}")
        
        # コンテキスト付きプロンプト構築
        full_prompt = self._build_prompt(prompt, context, conversation_history)
        
        # 生成設定のマージ
        generation_params = {**self.generation_config, **kwargs}
        
        try:
            # Ollama API呼び出し
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "options": {
                        "temperature": generation_params["temperature"],
                        "top_p": generation_params["top_p"],
                        "top_k": generation_params["top_k"],
                        "num_predict": generation_params["max_tokens"],
                        "stop": generation_params["stop"]
                    },
                    "stream": generation_params["stream"]
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()
                
                # トレーシング用メタデータ更新
                trace_metadata.update({
                    "response_length": len(generated_text),
                    "eval_count": result.get("eval_count", 0),
                    "eval_duration_ms": result.get("eval_duration", 0) / 1e6
                })
                
                # 生成統計ログ
                if "eval_count" in result:
                    logger.info(
                        f"生成完了: {result.get('eval_count', 0)}トークン, "
                        f"{result.get('eval_duration', 0)/1e6:.1f}ms"
                    )
                
                return generated_text
            else:
                logger.error(f"LLM生成エラー: {response.status_code} - {response.text}")
                return "申し訳ございませんが、応答の生成中にエラーが発生しました。"
                
        except requests.Timeout:
            logger.error("LLM生成タイムアウト")
            return "処理時間が長すぎるため、タイムアウトしました。より簡潔な質問でお試しください。"
        except Exception as e:
            logger.error(f"LLM生成例外: {e}")
            return "システムエラーが発生しました。しばらく後でお試しください。"
    
    async def generate_response_async(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        **kwargs
    ) -> str:
        """非同期版応答生成"""
        import aiohttp
        
        full_prompt = self._build_prompt(prompt, context)
        generation_params = {**self.generation_config, **kwargs}
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(self.timeout)) as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": full_prompt,
                        "options": {
                            "temperature": generation_params["temperature"],
                            "top_p": generation_params["top_p"],
                            "top_k": generation_params["top_k"],
                            "num_predict": generation_params["max_tokens"],
                            "stop": generation_params["stop"]
                        },
                        "stream": False
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "").strip()
                    else:
                        logger.error(f"非同期LLM生成エラー: {response.status}")
                        return "応答生成中にエラーが発生しました。"
                        
        except asyncio.TimeoutError:
            logger.error("非同期LLM生成タイムアウト")
            return "処理時間が長すぎるため、タイムアウトしました。"
        except Exception as e:
            logger.error(f"非同期LLM生成例外: {e}")
            return "システムエラーが発生しました。"
    
    def health_check(self) -> bool:
        """Ollamaサービスのヘルスチェック"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                # モデルが利用可能かも確認
                tags = response.json()
                available_models = [model["name"] for model in tags.get("models", [])]
                is_model_available = any(self.model_name in model for model in available_models)
                
                if is_model_available:
                    logger.info(f"LLMヘルスチェック成功: {self.model_name}")
                    return True
                else:
                    logger.warning(f"LLMモデル未利用可能: {self.model_name}")
                    return False
            else:
                logger.warning(f"Ollamaサービス応答エラー: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"LLMヘルスチェック失敗: {e}")
            return False
    
    def _build_prompt(self, question: str, context: Optional[str] = None, conversation_history: Optional[List[str]] = None) -> str:
        """RAG対応プロンプト構築（会話履歴対応）"""
        
        # 会話履歴の構築
        history_text = ""
        if conversation_history:
            # 最新10回の会話のみ使用
            recent_history = conversation_history[-10:]
            history_text = "\n".join(recent_history)
            history_text = f"\n\n過去の会話:\n{history_text}\n"
        
        if context:
            # RAGコンテキスト付きプロンプト
            prompt = f"""あなたは親切で知識豊富なAIアシスタントです。提供された文書の内容に基づいて、ユーザーの質問に正確に答えてください。{history_text}

参考文書:
{context}

質問: {question}

回答は以下の点を心がけてください：
1. 提供された文書の内容に基づいて回答する
2. 文書に記載されていない内容については推測せず、「文書には記載されていません」と明示する
3. 回答は具体的で実用的なものにする
4. 日本語で自然な文章で回答する

回答:"""
        else:
            # 一般的な質問プロンプト
            prompt = f"""あなたは親切で知識豊富なAIアシスタントです。ユーザーの質問に適切に答えてください。{history_text}

質問: {question}

回答:"""
        
        return prompt
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報取得"""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model_name},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"モデル情報取得失敗: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"モデル情報取得例外: {e}"}
    
    def get_server_status(self) -> Dict[str, Any]:
        """Ollamaサーバー状態取得"""
        try:
            # サーバー生存確認
            health_response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            
            # 実行中モデル確認
            ps_response = requests.get(
                f"{self.base_url}/api/ps",
                timeout=5
            )
            
            status = {
                "server_alive": health_response.status_code == 200,
                "timestamp": datetime.now().isoformat(),
                "configured_model": self.model_name,
                "ollama_host": self.ollama_host
            }
            
            if ps_response.status_code == 200:
                running_models = ps_response.json().get("models", [])
                status["running_models"] = [
                    {
                        "name": model.get("name", ""),
                        "size": model.get("size", 0),
                        "digest": model.get("digest", "")[:12]
                    }
                    for model in running_models
                ]
            
            return status
            
        except Exception as e:
            return {
                "server_alive": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# グローバルインスタンス
llm_manager = LLMManager()

def get_llm_manager() -> LLMManager:
    """LLM管理インスタンス取得"""
    return llm_manager

# 便利関数
def generate_rag_response(question: str, context: str) -> str:
    """RAG応答生成"""
    return llm_manager.generate_response(question, context)

def check_llm_health() -> bool:
    """LLMヘルスチェック"""
    return llm_manager.check_model_availability()

if __name__ == "__main__":
    # テスト実行
    manager = LLMManager()
    
    print("🔍 モデル利用可能性チェック...")
    if manager.check_model_availability():
        print("✅ モデル利用可能")
        
        print("\n💬 テスト応答生成...")
        test_response = manager.generate_response("こんにちは！自己紹介をしてください。")
        print(f"応答: {test_response}")
        
        print("\n📊 サーバー状態...")
        status = manager.get_server_status()
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print("❌ モデル利用不可")