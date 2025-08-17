#!/usr/bin/env python3
"""
기울기 소실 분석: Epoch Loss 데이터의 미분값 계산
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

def load_epoch_loss(file_path):
    """epoch_loss.txt 파일에서 데이터 로드"""
    epochs = []
    losses = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    epochs.append(int(parts[0]))
                    losses.append(float(parts[1]))
    
    return np.array(epochs), np.array(losses)

def calculate_derivatives(epochs, losses):
    """손실 함수의 1차, 2차 미분 계산"""
    # 1차 미분 (기울기)
    dloss_depoch = np.gradient(losses, epochs)
    
    # 2차 미분 (기울기의 변화율)
    d2loss_depoch2 = np.gradient(dloss_depoch, epochs)
    
    return dloss_depoch, d2loss_depoch2

def analyze_convergence(epochs, losses, dloss_depoch, d2loss_depoch2, experiment_name):
    """수렴 상태 분석"""
    print(f"\n{'='*60}")
    print(f"📊 {experiment_name} 분석")
    print(f"{'='*60}")
    
    # 기본 통계
    print(f"총 에폭: {len(epochs)}")
    print(f"초기 손실: {losses[0]:.6f}")
    print(f"최종 손실: {losses[-1]:.6f}")
    print(f"총 개선량: {losses[0] - losses[-1]:.6f}")
    
    # 미분값 분석
    print(f"\n🔍 미분값 분석:")
    print(f"평균 1차 미분 (기울기): {np.mean(dloss_depoch):.8f}")
    print(f"최종 100에폭 평균 기울기: {np.mean(dloss_depoch[-100:]):.8f}")
    print(f"최종 50에폭 평균 기울기: {np.mean(dloss_depoch[-50:]):.8f}")
    print(f"최종 10에폭 평균 기울기: {np.mean(dloss_depoch[-10:]):.8f}")
    
    # 기울기 크기 (절댓값)
    abs_gradient = np.abs(dloss_depoch)
    print(f"\n📉 기울기 크기 분석:")
    print(f"평균 기울기 크기: {np.mean(abs_gradient):.8f}")
    print(f"최종 100에폭 평균 기울기 크기: {np.mean(abs_gradient[-100:]):.8f}")
    print(f"최종 50에폭 평균 기울기 크기: {np.mean(abs_gradient[-50:]):.8f}")
    print(f"최종 10에폭 평균 기울기 크기: {np.mean(abs_gradient[-10:]):.8f}")
    
    # 기울기 소실 판정
    final_gradient_magnitude = np.mean(abs_gradient[-50:])
    initial_gradient_magnitude = np.mean(abs_gradient[:50])
    gradient_ratio = final_gradient_magnitude / initial_gradient_magnitude
    
    print(f"\n⚠️  기울기 소실 분석:")
    print(f"초기 50에폭 평균 기울기 크기: {initial_gradient_magnitude:.8f}")
    print(f"최종 50에폭 평균 기울기 크기: {final_gradient_magnitude:.8f}")
    print(f"기울기 비율 (최종/초기): {gradient_ratio:.4f}")
    
    if gradient_ratio < 0.01:
        print("🚨 심각한 기울기 소실 의심!")
    elif gradient_ratio < 0.1:
        print("⚠️  기울기 소실 가능성 있음")
    elif gradient_ratio < 0.5:
        print("✅ 정상적인 수렴")
    else:
        print("📈 활발한 학습 진행")
    
    # 수렴 안정성
    recent_variance = np.var(losses[-100:])
    print(f"\n📊 수렴 안정성:")
    print(f"최근 100에폭 손실 분산: {recent_variance:.8f}")
    
    return {
        'gradient_ratio': gradient_ratio,
        'final_gradient_magnitude': final_gradient_magnitude,
        'recent_variance': recent_variance,
        'total_improvement': losses[0] - losses[-1]
    }

def plot_derivatives(epochs, losses, dloss_depoch, d2loss_depoch2, experiment_name, output_dir):
    """미분값 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 원본 Loss 곡선
    axes[0, 0].plot(epochs, losses, 'b-', linewidth=2)
    axes[0, 0].set_title('Loss vs Epoch')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 1차 미분 (기울기)
    axes[0, 1].plot(epochs, dloss_depoch, 'r-', linewidth=2)
    axes[0, 1].set_title('Loss Gradient (dLoss/dEpoch)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Gradient')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 3. 기울기 크기 (절댓값)
    abs_gradient = np.abs(dloss_depoch)
    axes[1, 0].plot(epochs, abs_gradient, 'g-', linewidth=2)
    axes[1, 0].set_title('Absolute Gradient Magnitude')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('|Gradient|')
    axes[1, 0].set_yscale('log')  # 로그 스케일로 표시
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 2차 미분 (기울기 변화율)
    axes[1, 1].plot(epochs, d2loss_depoch2, 'm-', linewidth=2)
    axes[1, 1].set_title('Second Derivative (d²Loss/dEpoch²)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Second Derivative')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.suptitle(f'Gradient Analysis: {experiment_name}', fontsize=16)
    plt.tight_layout()
    
    # 저장
    output_path = os.path.join(output_dir, f'gradient_analysis_{experiment_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def main():
    # 그래프 디렉토리에서 모든 실험 찾기
    graph_dir = Path("graph")
    
    if not graph_dir.exists():
        print("graph 디렉토리가 없습니다.")
        return
    
    experiments = []
    
    # 모든 실험 디렉토리 탐색
    for exp_dir in graph_dir.iterdir():
        if exp_dir.is_dir():
            epoch_loss_file = exp_dir / "epoch_loss.txt"
            if epoch_loss_file.exists():
                experiments.append({
                    'name': exp_dir.name,
                    'path': exp_dir,
                    'epoch_loss_file': epoch_loss_file
                })
    
    if not experiments:
        print("epoch_loss.txt 파일을 찾을 수 없습니다.")
        return
    
    print(f"발견된 실험: {len(experiments)}개")
    
    # 분석 결과 저장용
    analysis_results = []
    
    # 출력 디렉토리 생성
    output_dir = Path("gradient_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # 각 실험 분석
    for exp in experiments:
        try:
            # 데이터 로드
            epochs, losses = load_epoch_loss(exp['epoch_loss_file'])
            
            if len(epochs) < 10:
                print(f"⚠️  {exp['name']}: 데이터가 너무 적음 (에폭 {len(epochs)}개)")
                continue
            
            # 미분 계산
            dloss_depoch, d2loss_depoch2 = calculate_derivatives(epochs, losses)
            
            # 분석
            result = analyze_convergence(epochs, losses, dloss_depoch, d2loss_depoch2, exp['name'])
            result['experiment_name'] = exp['name']
            analysis_results.append(result)
            
            # 시각화
            plot_path = plot_derivatives(epochs, losses, dloss_depoch, d2loss_depoch2, 
                                       exp['name'], output_dir)
            print(f"📊 그래프 저장: {plot_path}")
            
        except Exception as e:
            print(f"❌ {exp['name']} 분석 실패: {e}")
    
    # 전체 비교 결과
    if analysis_results:
        print(f"\n{'='*80}")
        print("🏆 실험 비교 요약")
        print(f"{'='*80}")
        
        # 기울기 소실 정도로 정렬
        sorted_results = sorted(analysis_results, key=lambda x: x['gradient_ratio'])
        
        print(f"{'실험명':<50} | {'기울기비율':<10} | {'최종기울기':<12} | {'총개선량':<10}")
        print("-" * 80)
        
        for result in sorted_results:
            name = result['experiment_name']
            if len(name) > 45:
                name = name[:42] + "..."
            
            print(f"{name:<50} | {result['gradient_ratio']:<10.4f} | "
                  f"{result['final_gradient_magnitude']:<12.8f} | "
                  f"{result['total_improvement']:<10.6f}")
        
        # 기울기 소실 가장 심한 실험
        worst_experiment = sorted_results[0]
        best_experiment = sorted_results[-1]
        
        print(f"\n🚨 기울기 소실이 가장 심한 실험:")
        print(f"   {worst_experiment['experiment_name']}")
        print(f"   기울기 비율: {worst_experiment['gradient_ratio']:.4f}")
        
        print(f"\n✅ 가장 건강한 학습:")
        print(f"   {best_experiment['experiment_name']}")
        print(f"   기울기 비율: {best_experiment['gradient_ratio']:.4f}")

if __name__ == "__main__":
    main()
