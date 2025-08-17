#!/usr/bin/env python3
"""
マルチチャットボットRAGシステム負荷テスト

同時接続50人を想定した負荷テスト
- 1000人の従業員中、1日最大50人が利用
- チャット機能、RAG機能、埋め込み機能をテスト
"""

import asyncio
import aiohttp
import time
import json
import random
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TestResult:
    """テスト結果"""
    endpoint: str
    status_code: int
    response_time: float
    success: bool
    error: str = None

class LoadTester:
    """負荷テスター"""
    
    def __init__(self, base_url: str = "http://localhost:8004"):
        self.base_url = base_url
        self.results: List[TestResult] = []
        self.chatbot_ids = ["bot_21f9f850"]  # テスト用チャットボットID
        
        # テスト用メッセージ
        self.test_messages = [
            "勤務時間について教えてください",
            "テレワークの業務範囲について",
            "労働基準法について教えて",
            "有給休暇の取得方法は？",
            "残業代の計算方法を教えてください",
            "育児休業について",
            "労災保険について教えて",
            "職場環境改善について",
            "ハラスメント対策について",
            "労働組合について"
        ]
    
    async def test_endpoint(self, session: aiohttp.ClientSession, endpoint: str, method: str = "GET", data: Dict = None) -> TestResult:
        """個別エンドポイントをテスト"""
        start_time = time.time()
        try:
            if method == "GET":
                async with session.get(f"{self.base_url}{endpoint}") as response:
                    await response.read()  # レスポンスを完全に読み込む
                    response_time = time.time() - start_time
                    return TestResult(
                        endpoint=endpoint,
                        status_code=response.status,
                        response_time=response_time,
                        success=response.status < 400
                    )
            elif method == "POST":
                headers = {"Content-Type": "application/json"}
                async with session.post(f"{self.base_url}{endpoint}", json=data, headers=headers) as response:
                    await response.read()
                    response_time = time.time() - start_time
                    return TestResult(
                        endpoint=endpoint,
                        status_code=response.status,
                        response_time=response_time,
                        success=response.status < 400
                    )
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                endpoint=endpoint,
                status_code=0,
                response_time=response_time,
                success=False,
                error=str(e)
            )
    
    async def simulate_user_session(self, session: aiohttp.ClientSession, user_id: int):
        """ユーザーセッションのシミュレーション"""
        chatbot_id = random.choice(self.chatbot_ids)
        
        # 1. ホームページアクセス
        result = await self.test_endpoint(session, "/")
        self.results.append(result)
        
        # 2. チャットボット一覧取得
        result = await self.test_endpoint(session, "/api/chatbots/list")
        self.results.append(result)
        
        # 3. チャット画面アクセス
        result = await self.test_endpoint(session, f"/chat/{chatbot_id}")
        self.results.append(result)
        
        # 4. チャットボット情報取得
        result = await self.test_endpoint(session, f"/api/chatbots/{chatbot_id}")
        self.results.append(result)
        
        # 5. 会話履歴取得
        result = await self.test_endpoint(session, f"/api/chat/{chatbot_id}/history")
        self.results.append(result)
        
        # 6. ランダムなメッセージ送信（RAGテスト）
        message = random.choice(self.test_messages)
        chat_data = {
            "message": message,
            "conversation_history": [],
            "use_rag": True
        }
        result = await self.test_endpoint(session, f"/api/chat/{chatbot_id}/message", "POST", chat_data)
        self.results.append(result)
        
        # ランダムな待機時間（実際のユーザー行動をシミュレート）
        await asyncio.sleep(random.uniform(1, 3))
        
        # 7. 埋め込みページアクセス（30%の確率）
        if random.random() < 0.3:
            result = await self.test_endpoint(session, f"/embed/{chatbot_id}")
            self.results.append(result)
    
    async def run_concurrent_test(self, concurrent_users: int = 50, duration_seconds: int = 300):
        """同時接続テストを実行"""
        print(f"負荷テスト開始: {concurrent_users}人同時接続、{duration_seconds}秒間実行")
        print(f"テスト開始時刻: {datetime.now()}")
        
        connector = aiohttp.TCPConnector(limit=200, limit_per_host=100)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            start_time = time.time()
            
            # 同時ユーザーセッションを開始
            for user_id in range(concurrent_users):
                task = asyncio.create_task(
                    self.simulate_continuous_user(session, user_id, duration_seconds)
                )
                tasks.append(task)
                
                # ユーザー追加間隔（段階的に負荷を増加）
                if user_id % 10 == 0:
                    await asyncio.sleep(0.5)
            
            # 全タスクの完了を待機
            await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
            print(f"\nテスト完了: 総実行時間 {total_time:.2f}秒")
            
        # 結果分析
        self.analyze_results()
    
    async def simulate_continuous_user(self, session: aiohttp.ClientSession, user_id: int, duration_seconds: int):
        """継続的なユーザーアクティビティをシミュレート"""
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time:
            try:
                await self.simulate_user_session(session, user_id)
                # ユーザー間のランダムな間隔
                await asyncio.sleep(random.uniform(5, 15))
            except Exception as e:
                print(f"User {user_id} error: {e}")
                await asyncio.sleep(2)
    
    def analyze_results(self):
        """テスト結果を分析"""
        if not self.results:
            print("テスト結果がありません")
            return
        
        print("\n" + "="*50)
        print("負荷テスト結果分析")
        print("="*50)
        
        total_requests = len(self.results)
        successful_requests = len([r for r in self.results if r.success])
        failed_requests = total_requests - successful_requests
        
        # 基本統計
        print(f"総リクエスト数: {total_requests}")
        print(f"成功: {successful_requests} ({successful_requests/total_requests*100:.1f}%)")
        print(f"失敗: {failed_requests} ({failed_requests/total_requests*100:.1f}%)")
        
        # レスポンス時間統計
        response_times = [r.response_time for r in self.results if r.success]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            
            print(f"\n応答時間統計:")
            print(f"平均: {avg_response_time:.3f}秒")
            print(f"最大: {max_response_time:.3f}秒")
            print(f"最小: {min_response_time:.3f}秒")
            
            # パーセンタイル計算
            response_times.sort()
            p95_index = int(len(response_times) * 0.95)
            p99_index = int(len(response_times) * 0.99)
            
            print(f"95パーセンタイル: {response_times[p95_index]:.3f}秒")
            print(f"99パーセンタイル: {response_times[p99_index]:.3f}秒")
        
        # エンドポイント別統計
        endpoint_stats = {}
        for result in self.results:
            if result.endpoint not in endpoint_stats:
                endpoint_stats[result.endpoint] = {
                    "total": 0,
                    "success": 0,
                    "response_times": []
                }
            
            endpoint_stats[result.endpoint]["total"] += 1
            if result.success:
                endpoint_stats[result.endpoint]["success"] += 1
                endpoint_stats[result.endpoint]["response_times"].append(result.response_time)
        
        print(f"\nエンドポイント別統計:")
        for endpoint, stats in endpoint_stats.items():
            success_rate = stats["success"] / stats["total"] * 100
            avg_time = sum(stats["response_times"]) / len(stats["response_times"]) if stats["response_times"] else 0
            print(f"  {endpoint}: {success_rate:.1f}% 成功, 平均応答時間: {avg_time:.3f}秒")
        
        # エラー分析
        errors = [r for r in self.results if not r.success]
        if errors:
            print(f"\nエラー分析:")
            error_counts = {}
            for error in errors:
                key = f"{error.status_code}: {error.error or 'HTTP Error'}"
                error_counts[key] = error_counts.get(key, 0) + 1
            
            for error, count in error_counts.items():
                print(f"  {error}: {count}回")
        
        # パフォーマンス判定
        print(f"\n" + "="*30)
        print("パフォーマンス評価")
        print("="*30)
        
        success_rate = successful_requests / total_requests * 100
        
        if success_rate >= 99.5:
            print("✅ 優秀: 99.5%以上の成功率")
        elif success_rate >= 99:
            print("✅ 良好: 99%以上の成功率")
        elif success_rate >= 95:
            print("⚠️  普通: 95%以上の成功率")
        else:
            print("❌ 問題: 95%未満の成功率")
        
        if response_times:
            if avg_response_time <= 1.0:
                print("✅ 高速: 平均応答時間 1秒以内")
            elif avg_response_time <= 3.0:
                print("✅ 良好: 平均応答時間 3秒以内")
            elif avg_response_time <= 5.0:
                print("⚠️  普通: 平均応答時間 5秒以内")
            else:
                print("❌ 遅い: 平均応答時間 5秒超過")

# メイン実行
async def main():
    """負荷テストメイン実行"""
    tester = LoadTester()
    
    print("=== マルチチャットボットRAGシステム負荷テスト ===\n")
    
    # テストシナリオ1: 軽い負荷テスト (10人同時接続, 60秒)
    print("シナリオ1: 軽い負荷テスト (10人同時接続, 60秒)")
    await tester.run_concurrent_test(concurrent_users=10, duration_seconds=60)
    
    # 結果をリセット
    tester.results.clear()
    print("\n" + "="*50 + "\n")
    
    # テストシナリオ2: 本格的負荷テスト (50人同時接続, 300秒)
    print("シナリオ2: 本格的負荷テスト (50人同時接続, 300秒)")
    await tester.run_concurrent_test(concurrent_users=50, duration_seconds=300)

if __name__ == "__main__":
    asyncio.run(main())