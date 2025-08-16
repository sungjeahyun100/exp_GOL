#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
특정 에폭들의 상세 통계를 출력하는 스크립트
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_and_display_epoch_stats(csv_file, selected_epochs=None):
    """CSV 파일에서 에폭 통계를 로드하고 특정 에폭들의 상세 정보를 출력합니다."""
    
    if not Path(csv_file).exists():
        print(f"❌ CSV 파일을 찾을 수 없습니다: {csv_file}")
        return
    
    # 데이터 로드
    df = pd.read_csv(csv_file)
    
    print("="*80)
    print("📊 에폭별 손실 통계 상세 정보")
    print("="*80)
    
    # 기본 선택 에폭 (그래프에서 표시된 에폭들)
    if selected_epochs is None:
        selected_epochs = [1, 100, 200, 300, 500, 800, 1000]
    
    # 실제 존재하는 에폭들만 필터링
    available_epochs = [e for e in selected_epochs if e in df['epoch'].values]
    
    if not available_epochs:
        print("❌ 선택된 에폭들이 데이터에 없습니다.")
        return
    
    print(f"🎯 선택된 에폭들: {available_epochs}")
    print()
    
    # 테이블 헤더
    print(f"{'Epoch':>6} | {'Mean':>9} | {'Std':>9} | {'Var':>9} | {'CV':>8} | {'Min':>9} | {'Max':>9} | {'Range':>9}")
    print("-" * 80)
    
    # 각 에폭의 통계 출력
    for epoch in available_epochs:
        row = df[df['epoch'] == epoch].iloc[0]
        print(f"{epoch:6d} | {row['mean']:9.6f} | {row['std']:9.6f} | {row['var']:9.6f} | "
              f"{row['cv']:8.6f} | {row['min']:9.6f} | {row['max']:9.6f} | {row['range']:9.6f}")
    
    print()
    print("📈 전체 추세 분석:")
    first_epoch = df[df['epoch'] == available_epochs[0]].iloc[0]
    last_epoch = df[df['epoch'] == available_epochs[-1]].iloc[0]
    
    print(f"   평균 손실: {first_epoch['mean']:.6f} → {last_epoch['mean']:.6f} "
          f"(변화: {last_epoch['mean'] - first_epoch['mean']:+.6f})")
    print(f"   표준편차: {first_epoch['std']:.6f} → {last_epoch['std']:.6f} "
          f"(변화: {last_epoch['std'] - first_epoch['std']:+.6f})")
    print(f"   분산: {first_epoch['var']:.6f} → {last_epoch['var']:.6f} "
          f"(변화: {last_epoch['var'] - first_epoch['var']:+.6f})")
    print(f"   변동계수: {first_epoch['cv']:.6f} → {last_epoch['cv']:.6f} "
          f"(변화: {last_epoch['cv'] - first_epoch['cv']:+.6f})")
    
    print()
    print("🏆 극값 분석:")
    min_std_row = df.loc[df['std'].idxmin()]
    max_std_row = df.loc[df['std'].idxmax()]
    min_var_row = df.loc[df['var'].idxmin()]
    max_var_row = df.loc[df['var'].idxmax()]
    
    print(f"   최소 표준편차: Epoch {int(min_std_row['epoch'])} (std={min_std_row['std']:.6f})")
    print(f"   최대 표준편차: Epoch {int(max_std_row['epoch'])} (std={max_std_row['std']:.6f})")
    print(f"   최소 분산: Epoch {int(min_var_row['epoch'])} (var={min_var_row['var']:.6f})")
    print(f"   최대 분산: Epoch {int(max_var_row['epoch'])} (var={max_var_row['var']:.6f})")
    
    # 분위수 분석
    print()
    print("📊 분위수 분석:")
    std_percentiles = np.percentile(df['std'], [25, 50, 75, 90, 95])
    var_percentiles = np.percentile(df['var'], [25, 50, 75, 90, 95])
    
    print("   표준편차 분위수:")
    print(f"     25%: {std_percentiles[0]:.6f}")
    print(f"     50%: {std_percentiles[1]:.6f}")
    print(f"     75%: {std_percentiles[2]:.6f}")
    print(f"     90%: {std_percentiles[3]:.6f}")
    print(f"     95%: {std_percentiles[4]:.6f}")
    
    print("   분산 분위수:")
    print(f"     25%: {var_percentiles[0]:.6f}")
    print(f"     50%: {var_percentiles[1]:.6f}")
    print(f"     75%: {var_percentiles[2]:.6f}")
    print(f"     90%: {var_percentiles[3]:.6f}")
    print(f"     95%: {var_percentiles[4]:.6f}")

def main():
    # 최신 CSV 파일 찾기
    stats_dir = Path("/home/sjh100/바탕화면/exp_GOL/statistics")
    
    if not stats_dir.exists():
        print(f"❌ 통계 디렉터리를 찾을 수 없습니다: {stats_dir}")
        return
    
    # 실험 ID별 폴더에서 CSV 파일 찾기
    csv_files = []
    for exp_dir in stats_dir.iterdir():
        if exp_dir.is_dir():
            exp_csv_files = list(exp_dir.glob("epoch_statistics_*.csv"))
            csv_files.extend(exp_csv_files)
    
    if not csv_files:
        print(f"❌ 통계 CSV 파일을 찾을 수 없습니다: {stats_dir}")
        return
    
    # 가장 최근 파일 선택
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"📂 사용 파일: {latest_csv.parent.name}/{latest_csv.name}")
    
    # 그래프에 표시된 에폭들
    selected_epochs = [1, 100, 200, 300, 500, 800, 1000]
    
    load_and_display_epoch_stats(str(latest_csv), selected_epochs)

if __name__ == "__main__":
    main()
