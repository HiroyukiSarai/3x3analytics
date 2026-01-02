"""
レポート生成モジュール
戦術分析レポートの生成
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import csv


class ReportGenerator:
    """戦術分析レポート生成器"""
    
    def __init__(
        self,
        output_dir: str = "outputs/reports"
    ):
        """
        Args:
            output_dir: レポート出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_full_report(
        self,
        tracking_data: List[Dict],
        patterns: List,
        actions: List,
        game_info: Optional[Dict] = None
    ) -> str:
        """
        完全なレポートを生成
        
        Args:
            tracking_data: トラッキングデータ
            patterns: 戦術パターン
            actions: 検出されたアクション
            game_info: 試合情報
            
        Returns:
            レポートファイルパス
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON形式のレポート
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_frames': len(tracking_data),
                'game_info': game_info or {}
            },
            'summary': self._generate_summary(tracking_data, patterns, actions),
            'patterns': [self._pattern_to_dict(p) for p in patterns],
            'action_stats': self._action_statistics(actions),
            'player_stats': self._player_statistics(tracking_data)
        }
        
        # JSON保存
        json_path = self.output_dir / f"report_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        # HTML形式のレポート
        html_path = self.output_dir / f"report_{timestamp}.html"
        self._generate_html_report(report_data, html_path)
        
        # CSV形式の統計データ
        csv_path = self.output_dir / f"pattern_stats_{timestamp}.csv"
        self._generate_csv_stats(patterns, csv_path)
        
        print(f"Reports generated:")
        print(f"  JSON: {json_path}")
        print(f"  HTML: {html_path}")
        print(f"  CSV: {csv_path}")
        
        return str(json_path)
    
    def _generate_summary(
        self,
        tracking_data: List[Dict],
        patterns: List,
        actions: List
    ) -> Dict:
        """サマリー情報を生成"""
        total_frames = len(tracking_data)
        fps = 30.0  # デフォルト
        
        # 選手数をカウント
        all_player_ids = set()
        for frame in tracking_data:
            for player in frame.get('players', []):
                all_player_ids.add(player.get('track_id'))
        
        # パターン別統計
        pattern_summary = {}
        for pattern in patterns:
            p_dict = self._pattern_to_dict(pattern)
            pattern_summary[p_dict['pattern_name']] = {
                'count': p_dict['occurrences'],
                'success_rate': p_dict['success_rate']
            }
        
        return {
            'total_frames': total_frames,
            'duration_seconds': total_frames / fps,
            'total_players_tracked': len(all_player_ids),
            'total_patterns_detected': len(patterns),
            'total_actions_detected': len(actions),
            'pattern_breakdown': pattern_summary
        }
    
    def _pattern_to_dict(self, pattern) -> Dict:
        """パターンオブジェクトを辞書に変換"""
        if hasattr(pattern, 'to_dict'):
            return pattern.to_dict()
        elif isinstance(pattern, dict):
            return pattern
        else:
            return {
                'pattern_id': getattr(pattern, 'pattern_id', 'unknown'),
                'pattern_name': getattr(pattern, 'pattern_name', 'Unknown'),
                'occurrences': getattr(pattern, 'occurrences', 0),
                'success_rate': getattr(pattern, 'success_rate', 0.0),
                'description': getattr(pattern, 'description', '')
            }
    
    def _action_statistics(self, actions: List) -> Dict:
        """アクション統計を計算"""
        stats = {}
        
        for action in actions:
            action_type = action.action_type.value if hasattr(action.action_type, 'value') else str(action.action_type)
            
            if action_type not in stats:
                stats[action_type] = {
                    'count': 0,
                    'total_duration': 0,
                    'players_involved': set()
                }
            
            stats[action_type]['count'] += 1
            stats[action_type]['total_duration'] += action.duration_frames
            stats[action_type]['players_involved'].update(action.player_ids)
        
        # setをリストに変換
        for action_type in stats:
            stats[action_type]['players_involved'] = list(
                stats[action_type]['players_involved']
            )
            if stats[action_type]['count'] > 0:
                stats[action_type]['avg_duration'] = (
                    stats[action_type]['total_duration'] / stats[action_type]['count']
                )
        
        return stats
    
    def _player_statistics(self, tracking_data: List[Dict]) -> Dict:
        """選手別統計を計算"""
        player_stats = {}
        
        for frame in tracking_data:
            for player in frame.get('players', []):
                track_id = player.get('track_id')
                
                if track_id not in player_stats:
                    player_stats[track_id] = {
                        'team': player.get('team', 'A'),
                        'frames_visible': 0,
                        'ball_possession_frames': 0,
                        'positions': []
                    }
                
                player_stats[track_id]['frames_visible'] += 1
                
                if player.get('has_ball', False):
                    player_stats[track_id]['ball_possession_frames'] += 1
                
                # 位置を記録
                bbox = player.get('bbox', [0, 0, 0, 0])
                pos = ((bbox[0] + bbox[2]) / 2, bbox[3])
                player_stats[track_id]['positions'].append(pos)
        
        # 移動距離を計算
        for track_id in player_stats:
            positions = player_stats[track_id]['positions']
            
            if len(positions) > 1:
                total_distance = 0
                for i in range(1, len(positions)):
                    dx = positions[i][0] - positions[i-1][0]
                    dy = positions[i][1] - positions[i-1][1]
                    total_distance += np.sqrt(dx**2 + dy**2)
                
                player_stats[track_id]['total_distance'] = total_distance
            else:
                player_stats[track_id]['total_distance'] = 0
            
            # 位置リストは削除（サイズ削減）
            del player_stats[track_id]['positions']
        
        return player_stats
    
    def _generate_html_report(self, report_data: Dict, output_path: Path):
        """HTML形式のレポートを生成"""
        html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3x3バスケットボール戦術分析レポート</title>
    <style>
        :root {{
            --primary: #2563eb;
            --secondary: #7c3aed;
            --success: #10b981;
            --warning: #f59e0b;
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --text: #e2e8f0;
            --text-muted: #94a3b8;
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, var(--bg-dark) 0%, #1a1a2e 100%);
            color: var(--text);
            min-height: 100vh;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(37, 99, 235, 0.3);
        }}
        
        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        
        .subtitle {{
            color: rgba(255,255,255,0.8);
            font-size: 1.1rem;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .card {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .card h3 {{
            color: var(--primary);
            margin-bottom: 1rem;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .stat {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        .stat:last-child {{ border-bottom: none; }}
        
        .stat-value {{
            font-weight: 600;
            color: var(--success);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        th {{
            background: rgba(37, 99, 235, 0.2);
            color: var(--primary);
        }}
        
        tr:hover {{
            background: rgba(255,255,255,0.05);
        }}
        
        .progress-bar {{
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            border-radius: 4px;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 3rem;
            color: var(--text-muted);
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏀 3x3 戦術分析レポート</h1>
            <p class="subtitle">生成日時: {report_data['metadata']['generated_at']}</p>
        </header>
        
        <div class="grid">
            <div class="card">
                <h3>📊 基本統計</h3>
                <div class="stat">
                    <span>総フレーム数</span>
                    <span class="stat-value">{report_data['summary']['total_frames']:,}</span>
                </div>
                <div class="stat">
                    <span>分析時間</span>
                    <span class="stat-value">{report_data['summary']['duration_seconds']:.1f}秒</span>
                </div>
                <div class="stat">
                    <span>トラッキング選手数</span>
                    <span class="stat-value">{report_data['summary']['total_players_tracked']}</span>
                </div>
            </div>
            
            <div class="card">
                <h3>🎯 検出結果</h3>
                <div class="stat">
                    <span>戦術パターン数</span>
                    <span class="stat-value">{report_data['summary']['total_patterns_detected']}</span>
                </div>
                <div class="stat">
                    <span>アクション検出数</span>
                    <span class="stat-value">{report_data['summary']['total_actions_detected']}</span>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>📈 戦術パターン分析</h3>
            <table>
                <thead>
                    <tr>
                        <th>パターン名</th>
                        <th>出現回数</th>
                        <th>成功率</th>
                        <th>グラフ</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_pattern_rows(report_data['patterns'])}
                </tbody>
            </table>
        </div>
        
        <div class="card">
            <h3>⚡ アクション統計</h3>
            <table>
                <thead>
                    <tr>
                        <th>アクション種類</th>
                        <th>検出数</th>
                        <th>平均継続時間</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_action_rows(report_data['action_stats'])}
                </tbody>
            </table>
        </div>
        
        <footer class="footer">
            <p>3x3 Basketball Tactical Analysis System</p>
        </footer>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_pattern_rows(self, patterns: List[Dict]) -> str:
        """パターンテーブルの行を生成"""
        rows = []
        for p in patterns:
            success_rate = p.get('success_rate', 0) * 100
            rows.append(f"""
                <tr>
                    <td>{p.get('pattern_name', 'Unknown')}</td>
                    <td>{p.get('occurrences', 0)}</td>
                    <td>{success_rate:.1f}%</td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {success_rate}%"></div>
                        </div>
                    </td>
                </tr>
            """)
        return ''.join(rows) if rows else '<tr><td colspan="4">データなし</td></tr>'
    
    def _generate_action_rows(self, action_stats: Dict) -> str:
        """アクションテーブルの行を生成"""
        rows = []
        for action_type, stats in action_stats.items():
            avg_duration = stats.get('avg_duration', 0)
            rows.append(f"""
                <tr>
                    <td>{action_type}</td>
                    <td>{stats.get('count', 0)}</td>
                    <td>{avg_duration:.1f}フレーム</td>
                </tr>
            """)
        return ''.join(rows) if rows else '<tr><td colspan="3">データなし</td></tr>'
    
    def _generate_csv_stats(self, patterns: List, output_path: Path):
        """CSV形式の統計データを生成"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'パターンID', 'パターン名', '出現回数', '成功回数', '成功率', '説明'
            ])
            
            for pattern in patterns:
                p_dict = self._pattern_to_dict(pattern)
                writer.writerow([
                    p_dict.get('pattern_id', ''),
                    p_dict.get('pattern_name', ''),
                    p_dict.get('occurrences', 0),
                    p_dict.get('success_count', 0),
                    f"{p_dict.get('success_rate', 0) * 100:.1f}%",
                    p_dict.get('description', '')
                ])
    
    def generate_heatmap_data(
        self,
        tracking_data: List[Dict],
        resolution: Tuple[int, int] = (30, 22)
    ) -> np.ndarray:
        """
        ヒートマップデータを生成
        
        Args:
            tracking_data: トラッキングデータ
            resolution: ヒートマップの解像度
            
        Returns:
            ヒートマップ配列
        """
        heatmap = np.zeros(resolution)
        
        # コートサイズ
        court_width = 1500
        court_height = 1100
        
        for frame in tracking_data:
            for player in frame.get('players', []):
                court_pos = player.get('court_position')
                
                if court_pos is None:
                    bbox = player.get('bbox', [0, 0, 0, 0])
                    court_pos = ((bbox[0] + bbox[2]) / 2, bbox[3])
                
                # グリッドインデックスに変換
                grid_x = int(court_pos[0] / court_width * (resolution[0] - 1))
                grid_y = int(court_pos[1] / court_height * (resolution[1] - 1))
                
                grid_x = np.clip(grid_x, 0, resolution[0] - 1)
                grid_y = np.clip(grid_y, 0, resolution[1] - 1)
                
                heatmap[grid_x, grid_y] += 1
        
        # 正規化
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap


