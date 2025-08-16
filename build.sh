#!/bin/bash

# exp_GOL 프로젝트 빌드 스크립트
# 헤더 전용 라이브러리 모드로 빌드합니다.

set -e  # 에러 발생 시 스크립트 중단

echo "=== exp_GOL 프로젝트 빌드 시작 ==="

# 빌드 디렉토리 생성
if [ ! -d "build" ]; then
    echo "빌드 디렉토리 생성 중..."
    mkdir build
else
    echo "기존 빌드 디렉토리 사용"
fi

cd build

# CMake 설정
echo "CMake 설정 중..."
cmake ..

# 빌드 실행 (헤더 전용 라이브러리는 컴파일 없음)
echo "빌드 실행 중..."
make -j$(nproc)

echo ""
echo "=== 빌드 완료 ==="
echo "프로젝트가 헤더 전용 라이브러리로 성공적으로 빌드되었습니다."
echo ""
echo "📁 빌드 결과:"
echo "   - 빌드 디렉토리: $(pwd)"
echo "   - 라이브러리 타입: 헤더 전용 (INTERFACE)"
echo "   - 포함 경로: ../src"
echo ""
echo "💡 사용 방법:"
echo "   - 헤더 파일들을 직접 포함하여 사용하세요"
echo "   - CUDA 소스(.cu) 파일은 대상 프로젝트에서 직접 컴파일하세요"
echo ""
echo "⚠️  CUDA 관련 주의사항:"
echo "   - 현재 환경(Ubuntu 24.04 + GCC 13.3.0)에서는 CUDA 12.8 호환성 문제가 있습니다"
echo "   - CUDA 기능을 사용하려면 GCC 11/12 또는 CUDA 11.x를 권장합니다"
