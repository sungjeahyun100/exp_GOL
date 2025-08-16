#exp_gol

사용환경:13th Gen Intel(R) Core(TM) i7-13650HX, Geforce RTX 4060 for laptop, ubuntu 24.04 LTS 

python 가상환경 활성화 명령어: source $(pwd)/.venv/bin/activate
빌드 명렁어: ./build.sh



#실험노트
based topic:풀지 못함이 수학적으로 증명된 문제를 인공지능이 풀이를 시도한다면 인공지능은 어떤 결론을 내릴 것인가? 

attampt 1: 분류형 신경망 모델(mlp)에게 초기조건만 쥐어준 후 미래를 예측하도록 훈련시켜본다.
우선 가장 잘 훈련되는 크기 및 설정을 찾은 후, 데이터를 점차적으로 늘려가면서 loss율의 추이를 관찰한다. 

attampt 1 detail:
GOL의 예측 불가능성을 활용하여 데이터셋을 생성하였다.
사용된 데이터 입력: 10*10 무작위 GOL패턴.
목표값:8비트 이진수 형태의 입력패턴의 살아남은 셀 개수(이때 이 개수는 100*100 보드에서 2500세데 또는 패턴 안정화가 될 때까지 세데를 진행시킨 후 살아있는 셀의 개수를 셈으로서 계산하였다.)

최적 설정 실험에서 쓰인 데이터의 양:1000개, epoch설정:100
추이 관찰 실험은 최적화의 한계로 시간이 오래걸려 진행이 불가함.

founded optimized model details:
simple MLP(Adam opt, not mini-batched) 
input layer 100 -> 128, LReLU act
hidden1 layer 128 -> 128, tanh act
hidden2 layer 128 -> 64, tanh act
output layer 64 -> 8, Softsign act
loss calc: cross entropy

reaches minimum loss val: 0.62

당면한 과제: 
목표값 설정의 타당성이 의심된다. 현제 살아있는 셀의 개수가 min:0, max:450이상인데, 분포 그래프를 보면 지나치게 50 미만의 경우로 쏠려있다.
따라서 세데를 진행시킨 100*100 패턴 자체를 타겟으로 잡을 계획이다.
여기서 총 세 가지 문제가 생긴다.
(1) 아웃풋이 지나치게 커 모델이 컴퓨팅 파워를 너무 많이 잡아먹음.
(2) 안정화된 패턴이 애초에 너무 작아 제데로 된 훈련이 안될 수도 있다는 점.
(3) 데이터셋 생성 자체가 너무 무거워져 추이 관찰 실험이 불가능할 수도 있다는 점(현제 계획 상으론 1000 -> 2000 -> 4000 -> 8000 ... 이렇게 비선형적으로 파일 개수를 늘려나가는 것, epoch수도 1000으로 늘릴 생각이였다.)    

attampt 2: CNN레이어를 추가하여 모델을 재구성하였다. 활성화 함수는 이전 모델을 참고하여 설정하였다.

attampt 2 datails:
epoch설정:1000
sample 개수:2000
bs = 50 //batch size
// Conv layers: 10x10 -> feature extraction
conv1(bs, 1, 10, 10,   8, 5, 5,  1, 1,  1, 1, &conv1_opt, d2::InitType::He, hs.model_str), // 10x10x1 -> 6x6x8, active:LReLU
conv2(bs, 8, 6, 6,     16, 3, 3, 1, 1,  1, 1, &conv2_opt, d2::InitType::He, hs.model_str), // 6x6x8 -> 4x4x16 , active:LReLU 
conv3(bs, 16, 4, 4,    32, 3, 3, 1, 1,  1, 1, &conv3_opt, d2::InitType::He, hs.model_str), // 4x4x16 -> 2x2x32, active:LReLU

// FC layers: extracted features -> prediction
fc1(bs, 2*2*32, 128, &fc1_opt, d2::InitType::He, hs.model_str),     // 128 -> 128, active:tanh
fc2(bs, 128, 64, &fc2_opt, d2::InitType::He, hs.model_str),         // 128 -> 64, active:tanh
fc3(bs, 64, 32, &fc3_opt, d2::InitType::He, hs.model_str),          // 64 -> 32, active:tanh
fc_out(bs, 32, 8, &fc_out_opt, d2::InitType::He, hs.model_str)     // 32 -> 8 (output), active:SoftSign

실험 결과:epoch 1000의 로스율 2.09079 //보다 자세한 실험 결과는 /graph와 /plots경로를 참조하라.

주석1:인공지능을 공부하면서, 내 코드에 문제가 있다는 것을 알았다. 난 지금까지 Softmax(Softsign(z)) 값을 받고 있었다. 난 이것이 잘못된 적용인 줄 모르고 계속 써 왔었다.
이제는 코드를 약간 수정하여 문제를 해결했다. 앞으론 Softmax(z)결과를 받을 것이다. 
주석2:아무래도 과거에 작성했던 데이터 생성 코드가 최적화가 덜 되어있어서 생성에 너무 오래 걸린다. 
그래서 실험의 방향을 틀어 이번에 새로 알게 된 스케일링 법칙을 실험에 적용해보기로 결정, openai측에서 발표한 8*8image의 수식을 참고하여 이론적 최소 로스율을 계산한 후, 실험 평균과 얼마나 일치하는지 알아보려 한다.



