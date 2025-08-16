#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
각 에폭별 배치 손실의 분산과 표준편차를 계산하는 스크립트
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def find_latest_experiment(graph_dir):
    """가장 최근 실험 데이터를 찾습니다."""
    graph_path = Path(graph_dir)
    if not graph_path.exists():
        print(f"❌ 그래프 디렉터리를 찾을 수 없습니다: {graph_dir}")
        return None, None
    
    # 모든 실험 디렉터리를 찾고 시간순으로 정렬
    exp_dirs = [d for d in graph_path.iterdir() if d.is_dir()]
    if not exp_dirs:
        print(f"❌ 실험 디렉터리가 없습니다: {graph_dir}")
        return None, None
    
    # 가장 최근 디렉터리 찾기
    latest_dir = max(exp_dirs, key=lambda x: x.stat().st_mtime)
    
    batch_file = latest_dir / "batch_loss.txt"
    epoch_file = latest_dir / "epoch_loss.txt"
    
    if not batch_file.exists() or not epoch_file.exists():
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {latest_dir}")
        return None, None
    
    print(f"✅ 실험 디렉터리: {latest_dir.name}")
    return str(batch_file), str(epoch_file)

def load_data(batch_file, epoch_file):
    """배치 손실과 에폭 손실 데이터를 로드합니다."""
    print("📊 데이터 로드 중...")
    
    # 배치 데이터 로드: epoch, batch_num, loss
    batch_df = pd.read_csv(batch_file, sep=' ', header=None, 
                          names=['epoch', 'batch_num', 'loss'])
    
    # 에폭 데이터 로드: epoch, avg_loss
    epoch_df = pd.read_csv(epoch_file, sep=' ', header=None, 
                          names=['epoch', 'avg_loss'])
    
    print(f"   - 총 배치 데이터: {len(batch_df)} 개")
    print(f"   - 총 에폭 데이터: {len(epoch_df)} 개")
    print(f"   - 에폭 범위: {batch_df['epoch'].min()} ~ {batch_df['epoch'].max()}")
    
    return batch_df, epoch_df

def calculate_epoch_statistics(batch_df):
    """각 에폭별 통계를 계산합니다."""
    print("🔢 에폭별 통계 계산 중...")
    
    # 각 에폭별로 그룹화하여 통계 계산
    epoch_stats = batch_df.groupby('epoch')['loss'].agg([
        'count',      # 배치 수
        'mean',       # 평균
        'std',        # 표준편차
        'var',        # 분산
        'min',        # 최솟값
        'max',        # 최댓값
        'median'      # 중앙값
    ]).reset_index()
    
    # 변동계수(CV) 계산: std/mean
    epoch_stats['cv'] = epoch_stats['std'] / epoch_stats['mean']
    
    # 범위(range) 계산
    epoch_stats['range'] = epoch_stats['max'] - epoch_stats['min']
    
    return epoch_stats

def save_statistics(epoch_stats, output_dir, experiment_id):
    """통계 결과를 파일로 저장합니다."""
    # 실험 ID별 개별 폴더 생성
    experiment_output_path = Path(output_dir) / experiment_id
    experiment_output_path.mkdir(parents=True, exist_ok=True)
    
    # CSV 파일로 저장
    csv_file = experiment_output_path / f"epoch_statistics_{experiment_id}.csv"
    epoch_stats.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"✅ CSV 저장: {csv_file}")
    
    # 요약 텍스트 파일 저장
    summary_file = experiment_output_path / f"statistics_summary_{experiment_id}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Game of Life CNN - Epoch Statistics Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"실험 ID: {experiment_id}\n")
        f.write(f"생성 일시: {pd.Timestamp.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}\n\n")
        
        f.write("📊 전체 통계 요약:\n")
        f.write(f"- 총 에폭 수: {len(epoch_stats)}\n")
        f.write(f"- 평균 표준편차: {epoch_stats['std'].mean():.6f}\n")
        f.write(f"- 평균 분산: {epoch_stats['var'].mean():.6f}\n")
        f.write(f"- 표준편차 범위: {epoch_stats['std'].min():.6f} ~ {epoch_stats['std'].max():.6f}\n")
        f.write(f"- 분산 범위: {epoch_stats['var'].min():.6f} ~ {epoch_stats['var'].max():.6f}\n\n")
        
        f.write("🔝 표준편차가 높은 상위 10 에폭:\n")
        top_std = epoch_stats.nlargest(10, 'std')[['epoch', 'std', 'var', 'cv']]
        for _, row in top_std.iterrows():
            f.write(f"   Epoch {int(row['epoch']):4d}: std={row['std']:.6f}, var={row['var']:.6f}, cv={row['cv']:.6f}\n")
        
        f.write("\n🔻 표준편차가 낮은 상위 10 에폭:\n")
        bottom_std = epoch_stats.nsmallest(10, 'std')[['epoch', 'std', 'var', 'cv']]
        for _, row in bottom_std.iterrows():
            f.write(f"   Epoch {int(row['epoch']):4d}: std={row['std']:.6f}, var={row['var']:.6f}, cv={row['cv']:.6f}\n")
            
    print(f"✅ 요약 저장: {summary_file}")
    
    return csv_file, summary_file

def create_visualizations(epoch_stats, batch_df, output_dir, experiment_id):
    """시각화 그래프를 생성합니다."""
    print("📈 시각화 그래프 생성 중...")
    
    # 실험 ID별 개별 폴더 사용
    experiment_output_path = Path(output_dir) / experiment_id
    
    # 스타일 설정
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. 에폭별 표준편차와 분산 트렌드
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 표준편차 트렌드
    ax1.plot(epoch_stats['epoch'], epoch_stats['std'], 'b-', linewidth=1.5, alpha=0.8)
    ax1.set_title('Standard Deviation of Batch Loss per Epoch', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Standard Deviation')
    ax1.grid(True, alpha=0.3)
    
    # 분산 트렌드
    ax2.plot(epoch_stats['epoch'], epoch_stats['var'], 'r-', linewidth=1.5, alpha=0.8)
    ax2.set_title('Variance of Batch Loss per Epoch', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Variance')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    trend_file = experiment_output_path / f"statistics_trend_{experiment_id}.png"
    plt.savefig(trend_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 트렌드 그래프: {trend_file}")
    
    # 2. 분포 히스토그램
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 표준편차 분포
    ax1.hist(epoch_stats['std'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title('Distribution of Standard Deviations')
    ax1.set_xlabel('Standard Deviation')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # 분산 분포
    ax2.hist(epoch_stats['var'], bins=50, alpha=0.7, color='red', edgecolor='black')
    ax2.set_title('Distribution of Variances')
    ax2.set_xlabel('Variance')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # 변동계수 분포
    ax3.hist(epoch_stats['cv'], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.set_title('Distribution of Coefficient of Variation')
    ax3.set_xlabel('CV (std/mean)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # 범위 분포
    ax4.hist(epoch_stats['range'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_title('Distribution of Loss Range per Epoch')
    ax4.set_xlabel('Range (max - min)')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    dist_file = experiment_output_path / f"statistics_distribution_{experiment_id}.png"
    plt.savefig(dist_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 분포 그래프: {dist_file}")
    
    # 3. 상관관계 히트맵
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_cols = ['mean', 'std', 'var', 'cv', 'range']
    correlation_matrix = epoch_stats[corr_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Matrix of Epoch Statistics', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    corr_file = experiment_output_path / f"statistics_correlation_{experiment_id}.png"
    plt.savefig(corr_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 상관관계 그래프: {corr_file}")
    
    # 4. 선택된 에폭들의 배치 손실 분포 (박스플롯)
    selected_epochs = [1, 100, 200, 300, 500, 800, 1000]
    available_epochs = [e for e in selected_epochs if e in batch_df['epoch'].values]
    
    if len(available_epochs) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        epoch_data = []
        epoch_labels = []
        
        for epoch in available_epochs:
            epoch_losses = batch_df[batch_df['epoch'] == epoch]['loss'].values
            epoch_data.append(epoch_losses)
            epoch_labels.append(f'Epoch {epoch}')
        
        box = ax.boxplot(epoch_data, tick_labels=epoch_labels, patch_artist=True)
        
        # 박스 색상 설정
        colors = plt.cm.Set3(np.linspace(0, 1, len(epoch_data)))
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('Batch Loss Distribution for Selected Epochs', fontsize=14, fontweight='bold')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        box_file = experiment_output_path / f"selected_epochs_boxplot_{experiment_id}.png"
        plt.savefig(box_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 박스플롯: {box_file}")

def print_summary_statistics(epoch_stats):
    """주요 통계를 콘솔에 출력합니다."""
    print("\n" + "="*60)
    print("📊 에폭별 손실 분산/표준편차 통계 요약")
    print("="*60)
    
    print(f"\n🔢 기본 정보:")
    print(f"   - 총 에폭 수: {len(epoch_stats)}")
    print(f"   - 에폭당 평균 배치 수: {epoch_stats['count'].mean():.1f}")
    
    print(f"\n📈 표준편차 통계:")
    print(f"   - 평균: {epoch_stats['std'].mean():.6f}")
    print(f"   - 최소: {epoch_stats['std'].min():.6f} (Epoch {int(epoch_stats.loc[epoch_stats['std'].idxmin(), 'epoch'])})")
    print(f"   - 최대: {epoch_stats['std'].max():.6f} (Epoch {int(epoch_stats.loc[epoch_stats['std'].idxmax(), 'epoch'])})")
    print(f"   - 중앙값: {epoch_stats['std'].median():.6f}")
    
    print(f"\n📊 분산 통계:")
    print(f"   - 평균: {epoch_stats['var'].mean():.6f}")
    print(f"   - 최소: {epoch_stats['var'].min():.6f} (Epoch {int(epoch_stats.loc[epoch_stats['var'].idxmin(), 'epoch'])})")
    print(f"   - 최대: {epoch_stats['var'].max():.6f} (Epoch {int(epoch_stats.loc[epoch_stats['var'].idxmax(), 'epoch'])})")
    print(f"   - 중앙값: {epoch_stats['var'].median():.6f}")
    
    print(f"\n🎯 변동계수 (CV) 통계:")
    print(f"   - 평균: {epoch_stats['cv'].mean():.6f}")
    print(f"   - 최소: {epoch_stats['cv'].min():.6f} (Epoch {int(epoch_stats.loc[epoch_stats['cv'].idxmin(), 'epoch'])})")
    print(f"   - 최대: {epoch_stats['cv'].max():.6f} (Epoch {int(epoch_stats.loc[epoch_stats['cv'].idxmax(), 'epoch'])})")

def main():
    parser = argparse.ArgumentParser(description='Game of Life CNN 에폭별 통계 계산')
    parser.add_argument('--graph-dir', default='/home/sjh100/바탕화면/exp_GOL/graph',
                       help='그래프 데이터 디렉터리 경로')
    parser.add_argument('--output-dir', default='/home/sjh100/바탕화면/exp_GOL/statistics',
                       help='결과 저장 디렉터리 경로')
    parser.add_argument('--experiment-id', help='특정 실험 ID 지정 (기본값: 최신 실험)')
    
    args = parser.parse_args()
    
    print("🚀 Game of Life CNN - 에폭별 통계 계산 시작")
    print(f"📂 그래프 디렉터리: {args.graph_dir}")
    print(f"💾 출력 디렉터리: {args.output_dir}")
    
    # 실험 데이터 찾기
    if args.experiment_id:
        graph_path = Path(args.graph_dir) / args.experiment_id
        batch_file = graph_path / "batch_loss.txt"
        epoch_file = graph_path / "epoch_loss.txt"
        if not batch_file.exists() or not epoch_file.exists():
            print(f"❌ 지정된 실험을 찾을 수 없습니다: {args.experiment_id}")
            return 1
        experiment_id = args.experiment_id
    else:
        batch_file, epoch_file = find_latest_experiment(args.graph_dir)
        if not batch_file:
            return 1
        experiment_id = Path(batch_file).parent.name
    
    # 데이터 로드
    batch_df, epoch_df = load_data(batch_file, epoch_file)
    
    # 통계 계산
    epoch_stats = calculate_epoch_statistics(batch_df)
    
    # 결과 저장
    csv_file, summary_file = save_statistics(epoch_stats, args.output_dir, experiment_id)
    
    # 시각화 생성
    create_visualizations(epoch_stats, batch_df, args.output_dir, experiment_id)
    
    # 요약 출력
    print_summary_statistics(epoch_stats)
    
    print(f"\n🎉 완료! 결과 저장 위치: {args.output_dir}")
    print(f"📄 CSV 파일: {csv_file.name}")
    print(f"📝 요약 파일: {summary_file.name}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
