#!/usr/bin/env python3
"""
簡単な負荷テスト（軽量版）
"""

import asyncio
import aiohttp
import time
from datetime import datetime

async def test_endpoint(session, endpoint):
    """エンドポイントをテスト"""
    start_time = time.time()
    try:
        async with session.get(f"http://localhost:8004{endpoint}") as response:
            await response.read()
            response_time = time.time() - start_time
            return {
                "endpoint": endpoint,
                "status": response.status,
                "response_time": response_time,
                "success": response.status < 400
            }
    except Exception as e:
        return {
            "endpoint": endpoint,
            "status": 0,
            "response_time": time.time() - start_time,
            "success": False,
            "error": str(e)
        }

async def simple_user_session(session, user_id):
    """シンプルなユーザーセッション"""
    results = []
    
    # 基本的なエンドポイントテスト
    endpoints = [
        "/",
        "/api/chatbots/list",
        "/chat/bot_21f9f850",
        "/api/chatbots/bot_21f9f850"
    ]
    
    for endpoint in endpoints:
        result = await test_endpoint(session, endpoint)
        results.append(result)
        await asyncio.sleep(0.5)  # リクエスト間隔
    
    return results

async def run_simple_test(concurrent_users=5, duration_seconds=30):
    """簡単な負荷テスト実行"""
    print(f"簡単な負荷テスト開始: {concurrent_users}人同時接続、{duration_seconds}秒間")
    print(f"開始時刻: {datetime.now()}")
    
    all_results = []
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        while time.time() < end_time:
            # 同時ユーザーセッション実行
            tasks = []
            for user_id in range(concurrent_users):
                task = simple_user_session(session, user_id)
                tasks.append(task)
            
            # 結果収集
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for results in batch_results:
                if isinstance(results, list):
                    all_results.extend(results)
            
            await asyncio.sleep(2)  # バッチ間隔
    
    # 結果分析
    total_requests = len(all_results)
    successful = len([r for r in all_results if r.get('success')])
    failed = total_requests - successful
    
    response_times = [r['response_time'] for r in all_results if r.get('success')]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    print(f"\n=== テスト結果 ===")
    print(f"総リクエスト数: {total_requests}")
    print(f"成功: {successful} ({successful/total_requests*100:.1f}%)")
    print(f"失敗: {failed} ({failed/total_requests*100:.1f}%)")
    print(f"平均応答時間: {avg_response_time:.3f}秒")
    
    if response_times:
        print(f"最速応答: {min(response_times):.3f}秒")
        print(f"最遅応答: {max(response_times):.3f}秒")
    
    # エラー詳細
    errors = [r for r in all_results if not r.get('success')]
    if errors:
        print(f"\nエラー詳細:")
        error_counts = {}
        for error in errors:
            key = f"{error.get('status', 0)}: {error.get('error', 'Unknown')}"
            error_counts[key] = error_counts.get(key, 0) + 1
        
        for error, count in error_counts.items():
            print(f"  {error}: {count}回")
    
    print("\n簡単な負荷テスト完了")

if __name__ == "__main__":
    asyncio.run(run_simple_test())