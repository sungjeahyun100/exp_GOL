#!/bin/bash
# GOL 실험 결과 그래프 생성 스크립트 (gnuplot 사용)
# - 새 형식: graph/<실험ID>/epoch_loss.txt, batch_loss.txt
# - 구 형식: graph/epoch_loss-<ID>.txt, batch_loss-<ID>.txt (폴백)

GRAPH_DIR="/home/sjh100/바탕화면/exp_GOL/graph"
OUTPUT_DIR="/home/sjh100/바탕화면/exp_GOL/plots"

command -v gnuplot >/dev/null 2>&1 || { echo "❌ gnuplot이 없습니다. sudo apt-get install gnuplot"; exit 1; }

mkdir -p "$OUTPUT_DIR"
echo "🔍 그래프 생성을 위한 데이터 파일 검색..."

LATEST_EPOCH=""
LATEST_BATCH=""
LATEST_ID=""
MODE=""  # "dir" or "flat"

# 1) 새 형식(디렉터리) 우선 탐색
mapfile -t EXP_DIRS < <(ls -td "$GRAPH_DIR"/*/ 2>/dev/null)
for d in "${EXP_DIRS[@]}"; do
  [ -f "${d}epoch_loss.txt" ] && [ -f "${d}batch_loss.txt" ] || continue
  LATEST_EPOCH="${d}epoch_loss.txt"
  LATEST_BATCH="${d}batch_loss.txt"
  LATEST_ID="$(basename "${d%/}")"
  MODE="dir"
  break
done

# 2) 구 형식(평면 파일) 폴백
if [ -z "$LATEST_ID" ]; then
  mapfile -t EPOCH_FILES < <(ls -t "$GRAPH_DIR"/epoch_loss-*.txt 2>/dev/null)
  for f in "${EPOCH_FILES[@]}"; do
    id="$(basename "$f")"
    id="${id#epoch_loss-}"; id="${id%.txt}"
    if [ -f "$GRAPH_DIR/batch_loss-$id.txt" ]; then
      LATEST_EPOCH="$f"
      LATEST_BATCH="$GRAPH_DIR/batch_loss-$id.txt"
      LATEST_ID="$id"
      MODE="flat"
      break
    fi
  done
fi

if [ -z "$LATEST_ID" ]; then
  echo "❌ epoch/batch 페어를 찾지 못했습니다. 경로: $GRAPH_DIR"
  ls -la "$GRAPH_DIR" || true
  exit 1
fi

echo "✅ 데이터 파일:"
echo "   Epoch: $(basename "$LATEST_EPOCH")"
echo "   Batch: $(basename "$LATEST_BATCH")"
echo "📊 실험 ID: $LATEST_ID (mode: $MODE)"

OUT_DIR="$OUTPUT_DIR/$LATEST_ID"
mkdir -p "$OUT_DIR"

# 통계 계산
TOTAL_EPOCHS=$(wc -l < "$LATEST_EPOCH" | tr -d ' ')
if [ -z "$TOTAL_EPOCHS" ] || [ "$TOTAL_EPOCHS" -eq 0 ]; then
  echo "❌ Epoch 파일이 비어있습니다: $LATEST_EPOCH"; exit 1
fi

INITIAL_LOSS=$(head -n 1 "$LATEST_EPOCH" | awk '{print $2}')
FINAL_LINE=$(tail -n 1 "$LATEST_EPOCH")
FINAL_EPOCH=$(echo "$FINAL_LINE" | awk '{print $1}')
FINAL_LOSS=$(echo "$FINAL_LINE" | awk '{print $2}')
MIN_LOSS=$(awk '{print $2}' "$LATEST_EPOCH" | sort -n | head -n 1)

# 마지막 에포크 배치 수 추정
MAX_BATCH=$(awk -v e="$FINAL_EPOCH" '($1==e){ if($2>m) m=$2 } END { if(m>0) print m; else print 40 }' "$LATEST_BATCH")

# 동적 에포크 선택
choose_epochs() {
  local T=$1
  if (( T <= 6 )); then
    seq 1 "$T"
    return
  fi
  local -a cand=()
  cand+=(1)
  local e2=$(( T/10 ));   (( e2 < 1 )) && e2=1;     cand+=("$e2")
  local e3=$(( T/5 ));    (( e3 < 1 )) && e3=1;     cand+=("$e3")
  local e4=$(( T*3/10 )); (( e4 < 1 )) && e4=1;     cand+=("$e4")
  local e5=$(( T/2 ));    (( e5 < 1 )) && e5=1;     cand+=("$e5")
  local e6=$(( T*8/10 )); (( e6 < 1 )) && e6=1;     cand+=("$e6")
  cand+=("$T")
  mapfile -t cand < <(printf "%s\n" "${cand[@]}" | awk '!seen[$0]++' | sort -n)
  printf "%s " "${cand[@]}"
}
read -r -a SELECTED_EPOCHS <<< "$(choose_epochs "$TOTAL_EPOCHS")"

echo "📈 Epoch Average Loss 그래프 생성..."
gnuplot << EOF
set terminal png enhanced size 1200,800 font 'Arial,12'
set output '${OUT_DIR}/epoch_avg_loss_${LATEST_ID}.png'
set title 'Epoch Average Loss' font 'Arial,16'
set xlabel 'Epoch' font 'Arial,14'
set ylabel 'Average Loss' font 'Arial,14'
set grid
set key top right
set autoscale
set style line 1 lc rgb '#1f77b4' lt 1 lw 2 pt 7 ps 0.5
plot '${LATEST_EPOCH}' using 1:2 with linespoints ls 1 title 'Training Loss'
EOF
if [ $? -ne 0 ]; then echo "❌ Epoch 그래프 실패"; else echo "✅ 저장: ${OUT_DIR}/epoch_avg_loss_${LATEST_ID}.png"; fi

echo "📈 에포크별 배치 손실 그래프 생성..."
epochs="${SELECTED_EPOCHS[*]}"

gnuplot << EOF
set terminal png enhanced size 1600,1000 font 'Arial,12'
set output '${OUT_DIR}/epoch_batch_comparison_${LATEST_ID}.png'
set title 'Batch Loss per Epoch (Selected Epochs)' font 'Arial,16'
set xlabel 'Batch Number (1-${MAX_BATCH})' font 'Arial,14'
set ylabel 'Loss' font 'Arial,14'
set grid
set key top left
set xrange [1:${MAX_BATCH}]

# gnuplot에서 직접 1열(에포크) 값을 필터링
epochs = "${epochs}"
plot for [i=1:words(epochs)] \
     '${LATEST_BATCH}' using ( (int(column(1))==int(word(epochs,i))) ? column(2) : 1/0 ):3 \
     with linespoints lw 1.5 title sprintf("Epoch %d", int(word(epochs,i)))
EOF
if [ $? -ne 0 ]; then echo "❌ 배치 그래프 실패"; else echo "✅ 저장: ${OUT_DIR}/epoch_batch_comparison_${LATEST_ID}.png"; fi

echo "📊 Loss 통계 요약 생성..."
LAST_EPOCH="$FINAL_EPOCH"
LAST_EPOCH_AVG=$(awk -v epoch="$LAST_EPOCH" 'BEGIN{sum=0;cnt=0} ($1==epoch){sum+=$3;cnt++} END{if(cnt>0) printf("%.6f", sum/cnt); else print "N/A"}' "$LATEST_BATCH")

cat > "${OUT_DIR}/training_summary_${LATEST_ID}.txt" << EOF2
Game of Life CNN Training Summary
=====================================
실험 ID: ${LATEST_ID}
생성 일시: $(date '+%Y년 %m월 %d일 %H:%M:%S')

📊 Epoch Loss 통계:
- 총 에포크 수: ${TOTAL_EPOCHS}
- 초기 손실: ${INITIAL_LOSS}
- 최종 손실: ${FINAL_LOSS}
- 최소 손실: ${MIN_LOSS}
- 전체 개선: $(echo "${INITIAL_LOSS} - ${FINAL_LOSS}" | bc -l)

📈 마지막 에포크 Batch 통계:
- 에포크: ${LAST_EPOCH}
- 평균 배치 손실: ${LAST_EPOCH_AVG}

🗂️ 생성된 그래프:
- Epoch Average Loss: epoch_avg_loss_${LATEST_ID}.png
- Epoch Batch Comparison: epoch_batch_comparison_${LATEST_ID}.png

📁 파일 위치: ${OUT_DIR}/
EOF2

echo "✅ 통계 요약: ${OUT_DIR}/training_summary_${LATEST_ID}.txt"
echo "🎉 완료. 출력: ${OUT_DIR}"