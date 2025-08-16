#include<GOLdatabase_2.hpp>
#include<utility.hpp>

int main(){
    int sample;
    int seed;
    double ratio = 0.3;
    dataset_id info;

    std::cout << "생성할 샘플 개수를 입력하시오: ";
    std::cin >> sample;
    info.sample_quantity = sample;

    std::cout << "시드 값을 입력하시오 (같은 시드 -> 동일 데이터셋): ";
    std::cin >> seed;
    info.seed = seed;

    std::cout << "활성 비율(alive ratio)을 입력하시오 (기본 0.3): ";
    if(!(std::cin >> ratio)){
        ratio = 0.3;
    }
    info.alive_ratio = ratio;

    GOL_2::generateGameOfLifeData(sample, ratio, seed, info);
}
