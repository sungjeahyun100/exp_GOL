#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 statistics 폴더의 파일들을 실험 ID별 폴더로 재정리하는 스크립트
"""

import os
import shutil
from pathlib import Path
import re

def reorganize_statistics_folder():
    """statistics 폴더의 기존 파일들을 실험 ID별로 재정리합니다."""
    
    stats_dir = Path("/home/sjh100/바탕화면/exp_GOL/statistics")
    
    if not stats_dir.exists():
        print(f"❌ statistics 디렉터리가 존재하지 않습니다: {stats_dir}")
        return
    
    print("🔄 기존 파일들을 실험 ID별 폴더로 재정리 중...")
    
    # 모든 파일 목록 가져오기 (디렉터리 제외)
    files = [f for f in stats_dir.iterdir() if f.is_file()]
    
    if not files:
        print("✅ 재정리할 파일이 없습니다.")
        return
    
    # 파일별로 실험 ID 추출하고 이동
    moved_count = 0
    
    for file_path in files:
        filename = file_path.name
        
        # 파일명에서 실험 ID 추출
        # 패턴: filename_experimentID.extension
        patterns = [
            r'epoch_statistics_(.+)\.csv$',
            r'statistics_summary_(.+)\.txt$',
            r'statistics_trend_(.+)\.png$',
            r'statistics_distribution_(.+)\.png$',
            r'statistics_correlation_(.+)\.png$',
            r'selected_epochs_boxplot_(.+)\.png$'
        ]
        
        experiment_id = None
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                experiment_id = match.group(1)
                break
        
        if experiment_id:
            # 실험 ID별 폴더 생성
            exp_dir = stats_dir / experiment_id
            exp_dir.mkdir(exist_ok=True)
            
            # 파일 이동
            dest_path = exp_dir / filename
            try:
                shutil.move(str(file_path), str(dest_path))
                print(f"📁 {filename} → {experiment_id}/")
                moved_count += 1
            except Exception as e:
                print(f"❌ 파일 이동 실패: {filename} - {e}")
        else:
            print(f"⚠️ 실험 ID를 추출할 수 없는 파일: {filename}")
    
    print(f"\n✅ 재정리 완료: {moved_count}개 파일 이동")
    
    # 결과 구조 표시
    print("\n📁 현재 구조:")
    for item in sorted(stats_dir.iterdir()):
        if item.is_dir():
            print(f"📂 {item.name}/")
            for sub_item in sorted(item.iterdir()):
                print(f"   📄 {sub_item.name}")
        else:
            print(f"📄 {item.name}")

if __name__ == "__main__":
    reorganize_statistics_folder()
