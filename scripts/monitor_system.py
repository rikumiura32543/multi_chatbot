#!/usr/bin/env python3
"""
システムリソース監視スクリプト

負荷テスト中のCPU、メモリ、ネットワーク使用量を監視
"""

import time
import psutil
import json
import datetime
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class SystemStats:
    """システム統計"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int

class SystemMonitor:
    """システム監視クラス"""
    
    def __init__(self, log_file: str = "system_stats.json"):
        self.log_file = log_file
        self.stats_history: List[SystemStats] = []
        self.initial_network = psutil.net_io_counters()
    
    def get_current_stats(self) -> SystemStats:
        """現在のシステム統計を取得"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # メモリ使用量
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        
        # ディスク使用量
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent
        
        # ネットワーク統計
        network = psutil.net_io_counters()
        
        # アクティブなネットワーク接続数
        connections = len(psutil.net_connections(kind='tcp'))
        
        return SystemStats(
            timestamp=datetime.datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            disk_usage_percent=disk_usage_percent,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            active_connections=connections
        )
    
    def monitor_continuous(self, duration_seconds: int = 300, interval: int = 5):
        """継続的な監視"""
        print(f"システム監視開始: {duration_seconds}秒間、{interval}秒間隔")
        print(f"開始時刻: {datetime.datetime.now()}")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        while time.time() < end_time:
            stats = self.get_current_stats()
            self.stats_history.append(stats)
            
            # リアルタイム表示
            print(f"[{stats.timestamp}] "
                  f"CPU: {stats.cpu_percent:5.1f}%, "
                  f"メモリ: {stats.memory_percent:5.1f}% "
                  f"({stats.memory_available_gb:.1f}GB利用可能), "
                  f"接続数: {stats.active_connections}")
            
            # ログファイルに保存
            self.save_stats()
            
            time.sleep(interval)
        
        print(f"\n監視完了: {len(self.stats_history)}回のデータポイントを収集")
        self.analyze_stats()
    
    def save_stats(self):
        """統計をファイルに保存"""
        data = [
            {
                "timestamp": stat.timestamp,
                "cpu_percent": stat.cpu_percent,
                "memory_percent": stat.memory_percent,
                "memory_available_gb": stat.memory_available_gb,
                "disk_usage_percent": stat.disk_usage_percent,
                "network_bytes_sent": stat.network_bytes_sent,
                "network_bytes_recv": stat.network_bytes_recv,
                "active_connections": stat.active_connections
            }
            for stat in self.stats_history
        ]
        
        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def analyze_stats(self):
        """統計分析"""
        if not self.stats_history:
            print("統計データがありません")
            return
        
        print("\n" + "="*50)
        print("システムリソース分析")
        print("="*50)
        
        # CPU統計
        cpu_values = [s.cpu_percent for s in self.stats_history]
        avg_cpu = sum(cpu_values) / len(cpu_values)
        max_cpu = max(cpu_values)
        min_cpu = min(cpu_values)
        
        print(f"CPU使用率:")
        print(f"  平均: {avg_cpu:.1f}%")
        print(f"  最大: {max_cpu:.1f}%")
        print(f"  最小: {min_cpu:.1f}%")
        
        # メモリ統計
        memory_values = [s.memory_percent for s in self.stats_history]
        avg_memory = sum(memory_values) / len(memory_values)
        max_memory = max(memory_values)
        min_memory = min(memory_values)
        
        print(f"\nメモリ使用率:")
        print(f"  平均: {avg_memory:.1f}%")
        print(f"  最大: {max_memory:.1f}%")
        print(f"  最小: {min_memory:.1f}%")
        
        # 接続数統計
        connection_values = [s.active_connections for s in self.stats_history]
        avg_connections = sum(connection_values) / len(connection_values)
        max_connections = max(connection_values)
        min_connections = min(connection_values)
        
        print(f"\nネットワーク接続数:")
        print(f"  平均: {avg_connections:.0f}")
        print(f"  最大: {max_connections}")
        print(f"  最小: {min_connections}")
        
        # ネットワーク転送量
        if len(self.stats_history) > 1:
            first_stat = self.stats_history[0]
            last_stat = self.stats_history[-1]
            
            bytes_sent_diff = last_stat.network_bytes_sent - first_stat.network_bytes_sent
            bytes_recv_diff = last_stat.network_bytes_recv - first_stat.network_bytes_recv
            
            print(f"\nネットワーク転送量（期間中）:")
            print(f"  送信: {bytes_sent_diff / (1024**2):.1f} MB")
            print(f"  受信: {bytes_recv_diff / (1024**2):.1f} MB")
        
        # 警告チェック
        print(f"\n" + "="*30)
        print("リソース使用量評価")
        print("="*30)
        
        if max_cpu > 80:
            print("⚠️  警告: CPU使用率が80%を超えています")
        elif max_cpu > 60:
            print("📊 注意: CPU使用率が60%を超えています")
        else:
            print("✅ CPU使用率は正常範囲内です")
        
        if max_memory > 80:
            print("⚠️  警告: メモリ使用率が80%を超えています")
        elif max_memory > 60:
            print("📊 注意: メモリ使用率が60%を超えています")
        else:
            print("✅ メモリ使用率は正常範囲内です")
        
        if max_connections > 1000:
            print("⚠️  警告: 同時接続数が1000を超えています")
        elif max_connections > 500:
            print("📊 注意: 同時接続数が500を超えています")
        else:
            print("✅ 同時接続数は正常範囲内です")

def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='システムリソース監視')
    parser.add_argument('--duration', type=int, default=300, help='監視時間（秒）')
    parser.add_argument('--interval', type=int, default=5, help='監視間隔（秒）')
    parser.add_argument('--output', type=str, default='system_stats.json', help='出力ファイル名')
    
    args = parser.parse_args()
    
    monitor = SystemMonitor(log_file=args.output)
    
    try:
        monitor.monitor_continuous(
            duration_seconds=args.duration,
            interval=args.interval
        )
    except KeyboardInterrupt:
        print("\n\n監視を中断しました")
        if monitor.stats_history:
            monitor.analyze_stats()

if __name__ == "__main__":
    main()