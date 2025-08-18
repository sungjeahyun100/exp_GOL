#include <perceptron_2.hpp>
#include <utility.hpp>
#include <GOLdatabase_2.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;
namespace p2 = perceptron_2;

class GOLsolver_1{
    private:
        model_id model_info;
        dataset_id using_dataset;
        dataset_id test_data_info;

        p2::handleStream hs;
        p2::ActivateLayer act;
        p2::LossLayer loss;

        p2::convLayer conv1;
        p2::convLayer conv2;
        p2::convLayer conv3;

        p2::PerceptronLayer fc1;
        p2::PerceptronLayer fc2;
        p2::PerceptronLayer fc3;
        p2::PerceptronLayer fc4;
        p2::PerceptronLayer fc_out;

        p2::ActType fcAct;
        p2::ActType convAct;
        p2::ActType outAct;
        p2::LossType l;

        // 헬퍼 함수들 - 문자열을 enum으로 변환
        p2::ActType getActType(const std::string& actStr) {
            if (actStr == "LReLU") return p2::ActType::LReLU;
            else if (actStr == "Tanh") return p2::ActType::Tanh;
            else if (actStr == "ReLU") return p2::ActType::ReLU;
            else if (actStr == "Sigmoid") return p2::ActType::Sigmoid;
            else if (actStr == "Softmax") return p2::ActType::Softmax;
            else if (actStr == "Softplus") return p2::ActType::Softplus;
            else if (actStr == "Softsign") return p2::ActType::Softsign;
            else if (actStr == "ELU") return p2::ActType::ELU;
            else if (actStr == "SELU") return p2::ActType::SELU;
            else if (actStr == "Swish") return p2::ActType::Swish;
            else if (actStr == "Identity") return p2::ActType::Identity;
            else return p2::ActType::LReLU; // 기본값
        }

        p2::LossType getLossType(const std::string& lossStr) {
            if (lossStr == "BCEWithLogits") return p2::LossType::BCEWithLogits;
            else if (lossStr == "CrossEntropy") return p2::LossType::CrossEntropy;
            else if (lossStr == "MSE") return p2::LossType::MSE;
            else return p2::LossType::BCEWithLogits; // 기본값
        }

        d2::InitType getInitType(const std::string& initStr) {
            if (initStr == "He") return d2::InitType::He;
            else if (initStr == "Xavier") return d2::InitType::Xavier;
            else if (initStr == "Uniform") return d2::InitType::Uniform;
            else return d2::InitType::Xavier; // 기본값
        }

        p2::optType getOptType(const std::string& optStr) {
            if (optStr == "Adam") return p2::optType::Adam;
            else if (optStr == "SGD") return p2::optType::SGD;
            else return p2::optType::Adam; // 기본값
        }

    public:
        std::string id;

        // 현재 설정 조회 함수들
        const model_id& getModelInfo() const { return model_info; }
        
        void printCurrentConfig() const {
            std::cout << "=== 현재 모델 설정 ===" << std::endl;
            std::cout << "모델명: " << model_info.model_name << std::endl;
            std::cout << "Conv 활성화: " << model_info.conv_active << std::endl;
            std::cout << "Conv 초기화: " << model_info.conv_init << std::endl;
            std::cout << "FC 활성화: " << model_info.fc_active << std::endl;
            std::cout << "FC 초기화: " << model_info.fc_init << std::endl;
            std::cout << "옵티마이저: " << model_info.optimizer << std::endl;
            std::cout << "손실함수: " << model_info.loss << std::endl;
            std::cout << "에폭: " << model_info.epoch << std::endl;
            std::cout << "배치크기: " << model_info.batch_size << std::endl;
            std::cout << "학습률: " << model_info.learning_rate << std::endl;
            std::cout << "=====================" << std::endl;
        }

        // CUDA 환경 확인
        bool checkCudaEnvironment() {
            int deviceCount = 0;
            cudaError_t err = cudaGetDeviceCount(&deviceCount);
            if (err != cudaSuccess || deviceCount == 0) {
                std::cerr << "[FATAL] No CUDA device: " << cudaGetErrorString(err) << std::endl;
                return false;
            }
            std::cout << "CUDA devices found: " << deviceCount << std::endl;
            return true;
        }

        //save & load
        bool saveModel(const std::string& filepath) const {
            std::ofstream out(filepath, std::ios::binary);
            if (!out) return false;
            const char magic[4] = {'G','O','L','1'};
            out.write(magic, sizeof(magic));
            uint32_t layerCount = 8;
            out.write(reinterpret_cast<const char*>(&layerCount), sizeof(layerCount));
            conv1.saveBinary(out);
            conv2.saveBinary(out);
            conv3.saveBinary(out);
            fc1.saveBinary(out);
            fc2.saveBinary(out);
            fc3.saveBinary(out);
            fc4.saveBinary(out);
            fc_out.saveBinary(out);
            return static_cast<bool>(out);
        }

        bool loadModel(const std::string& filepath){
            std::ifstream in(filepath, std::ios::binary);
            if (!in) return false;
            char magic[4];
            in.read(magic, sizeof(magic));
            if (!in || magic[0] != 'G' || magic[1] != 'O' || magic[2] != 'L' || magic[3] != '1') return false;
            uint32_t layerCount = 0;
            in.read(reinterpret_cast<char*>(&layerCount), sizeof(layerCount));
            if (!in || layerCount != 8) return false;
            conv1.loadBinary(in, hs.model_str);
            conv2.loadBinary(in, hs.model_str);
            conv3.loadBinary(in, hs.model_str);
            fc1.loadBinary(in, hs.model_str);
            fc2.loadBinary(in, hs.model_str);
            fc3.loadBinary(in, hs.model_str);
            fc4.loadBinary(in, hs.model_str);
            fc_out.loadBinary(in, hs.model_str);
            return static_cast<bool>(in);
        }

        // 기본 생성자
        GOLsolver_1() : GOLsolver_1(50, 1e-6, 1000) {} // 기본값으로 위임

        //간편히 실행해볼 수 있도록 설정을 단순화한 생성자
        GOLsolver_1(int bs, double lr, int ep) : 
            // handleStream과 다른 멤버들을 먼저 초기화
            hs(),
            act(),
            loss(),
            // 레이어 초기화 (기본 생성자 사용)
            conv1(bs, 1, 10, 10, 8, 3, 3, 0, 0, 1, 1, p2::optType::Adam, d2::InitType::He, lr, hs.model_str),
            conv2(bs, 8, 8, 8, 16, 3, 3, 0, 0, 1, 1, p2::optType::Adam, d2::InitType::He, lr, hs.model_str),
            conv3(bs, 16, 6, 6, 32, 3, 3, 0, 0, 1, 1, p2::optType::Adam, d2::InitType::He, lr, hs.model_str),
            fc1(bs, 4*4*32, 256, p2::optType::Adam, d2::InitType::Xavier, lr, hs.model_str),
            fc2(bs, 256, 128, p2::optType::Adam, d2::InitType::Xavier, lr, hs.model_str),
            fc3(bs, 128, 64, p2::optType::Adam, d2::InitType::Xavier, lr, hs.model_str),
            fc4(bs, 64, 32, p2::optType::Adam, d2::InitType::Xavier, lr, hs.model_str),
            fc_out(bs, 32, 8, p2::optType::Adam, d2::InitType::Xavier, lr, hs.model_str),
            // 활성화 함수와 손실 함수 타입 설정
            convAct(p2::ActType::LReLU),
            fcAct(p2::ActType::Tanh),
            outAct(p2::ActType::Identity),
            l(p2::LossType::BCEWithLogits)
        {
            // model_info 설정
            model_info.model_name = "mercury";
            model_info.conv_active = "LReLU";
            model_info.conv_init = "He";
            model_info.fc_active = "Tanh";
            model_info.fc_init = "Xavier";
            model_info.conv_layer_count = 3;
            model_info.fc_layer_count = 5;
            model_info.optimizer = "Adam";
            model_info.loss = "BCEWithLogits";
            model_info.epoch = ep;
            model_info.batch_size = bs;
            model_info.learning_rate = lr;

            // dataset 설정
            using_dataset.seed = 54321;
            using_dataset.sample_quantity = 8000;
            using_dataset.alive_ratio = 0.3;

            test_data_info.seed = 98624;
            test_data_info.sample_quantity = 100;
            test_data_info.alive_ratio = 0.3;

            id = getModelId(model_info);
        }

        // model_info, dataset_info를 받는 생성자
        GOLsolver_1(const model_id& config, const dataset_id& dataset, const dataset_id& test_dataset, bool load_timestamp=true) :
            // handleStream과 다른 멤버들을 먼저 초기화
            hs(),
            act(), 
            loss(),
            // 레이어 초기화
            conv1(config.batch_size, 1, 10, 10, 8, 3, 3, 0, 0, 1, 1, getOptType(config.optimizer), getInitType(config.conv_init), config.learning_rate, hs.model_str),
            conv2(config.batch_size, 8, 8, 8, 16, 3, 3, 0, 0, 1, 1, getOptType(config.optimizer), getInitType(config.conv_init), config.learning_rate, hs.model_str),
            conv3(config.batch_size, 16, 6, 6, 32, 3, 3, 0, 0, 1, 1, getOptType(config.optimizer), getInitType(config.conv_init), config.learning_rate, hs.model_str),
            fc1(config.batch_size, 4*4*32, 256, getOptType(config.optimizer), getInitType(config.fc_init), config.learning_rate, hs.model_str),
            fc2(config.batch_size, 256, 128, getOptType(config.optimizer), getInitType(config.fc_init), config.learning_rate, hs.model_str),
            fc3(config.batch_size, 128, 64, getOptType(config.optimizer), getInitType(config.fc_init), config.learning_rate, hs.model_str),
            fc4(config.batch_size, 64, 32, getOptType(config.optimizer), getInitType(config.fc_init), config.learning_rate, hs.model_str),
            fc_out(config.batch_size, 32, 8, getOptType(config.optimizer), getInitType(config.fc_init), config.learning_rate, hs.model_str),
            // 활성화 함수와 손실 함수 타입 설정
            convAct(getActType(config.conv_active)),
            fcAct(getActType(config.fc_active)),
            l(getLossType(config.loss)),
            outAct(getLossType(config.loss) == p2::LossType::BCEWithLogits ? p2::ActType::Identity : 
                   (getLossType(config.loss) == p2::LossType::CrossEntropy ? p2::ActType::Softmax : p2::ActType::LReLU))
        {
            model_info = config;
            model_info.conv_layer_count = 3;
            model_info.fc_layer_count = 5;
            using_dataset = dataset;
            test_data_info = test_dataset;

            if(load_timestamp == true) id = getModelId(model_info);
            else if(load_timestamp == false) id = getModelIdWithoutTimestamp(model_info);

        }

        void genDataset(){
            GOL_2::generateGameOfLifeData(using_dataset.sample_quantity, using_dataset.alive_ratio, using_dataset.seed, using_dataset);
        }

        void genTestDataset(){
            GOL_2::generateGameOfLifeData(test_data_info.sample_quantity, test_data_info.alive_ratio, test_data_info.seed, test_data_info);
        }

        std::pair<d2::d_matrix_2<double>, double> forward(d2::d_matrix_2<double> X, d2::d_matrix_2<double> target, cudaStream_t str = 0){
            // Conv layers with activation
            conv1.forward(X, str);
            
            conv2.forward(act.Active(conv1.getOutput(), convAct, str), str);
            
            conv3.forward(act.Active(conv2.getOutput(), convAct, str), str);
            
            // FC layers with activation
            fc1.feedforward(act.Active(conv3.getOutput(), convAct, str), str);
            
            fc2.feedforward(act.Active(fc1.getOutput(), fcAct, str), str);
            
            fc3.feedforward(act.Active(fc2.getOutput(), fcAct, str), str);

            fc4.feedforward(act.Active(fc3.getOutput(), fcAct, str), str);

            fc_out.feedforward(act.Active(fc4.getOutput(), fcAct, str), str);
            // BCEWithLogits uses raw logits (no softmax)
            auto final_output = act.Active(fc_out.getOutput(), outAct, str);

            // Loss calculation  
            double loss_val = loss.getLoss(final_output, target, l, str);
            
            return {final_output, loss_val};
        }
        
        void backward(d2::d_matrix_2<double> final_output, d2::d_matrix_2<double> target, cudaStream_t str = 0){
            // Get loss gradient
            auto loss_grad = loss.getGrad(final_output, target, l, str);
            
            // Backward through FC layers (logits head uses Identity derivative)
            auto fc_out_act_deriv = act.d_Active(fc_out.getOutput(), outAct, str);
            auto delta_fc_out = fc_out.backprop(loss_grad, fc_out_act_deriv, str);

            auto fc4_act_deriv = act.d_Active(fc4.getOutput(), fcAct, str);
            auto delta_fc4 = fc4.backprop(delta_fc_out, fc4_act_deriv, str);

            auto fc3_act_deriv = act.d_Active(fc3.getOutput(), fcAct, str);
            auto delta_fc3 = fc3.backprop(delta_fc4, fc3_act_deriv, str);

            auto fc2_act_deriv = act.d_Active(fc2.getOutput(), fcAct, str);
            auto delta_fc2 = fc2.backprop(delta_fc3, fc2_act_deriv, str);
            
            auto fc1_act_deriv = act.d_Active(fc1.getOutput(), fcAct, str);
            auto delta_fc1 = fc1.backprop(delta_fc2, fc1_act_deriv, str);
            
            // Backward through conv layers
            auto conv3_act_deriv = act.d_Active(conv3.getOutput(), convAct, str);
            auto delta_conv3 = conv3.backward(delta_fc1, conv3_act_deriv, str);
            
            auto conv2_act_deriv = act.d_Active(conv2.getOutput(), convAct, str);
            auto delta_conv2 = conv2.backward(delta_conv3, conv2_act_deriv, str);
            
            auto conv1_act_deriv = act.d_Active(conv1.getOutput(), convAct, str);
            conv1.backward(delta_conv2, conv1_act_deriv, str);
        }

        void train(){
            auto start = std::chrono::steady_clock::now();
            
            // GOL 데이터 로드 (배치 형태로 직접 로드)
            auto [X, Y] = GOL_2::LoadingDataBatch(using_dataset, hs.model_str);

            int N = X.getRow();      // 전체 데이터 개수
            int input_size = X.getCol();   // 입력 크기 (100)
            int output_size = Y.getCol();  // 출력 크기 (8)
            
            std::cout << "[데이터 로드 완료] " << N << "개 샘플, 입력크기: " << input_size << ", 출력크기: " << output_size << std::endl;

            int B = model_info.batch_size;           // 배치 크기
            int num_batches = (N + B - 1) / B;  // 총 배치 수
            
            // 배치별로 데이터 미리 분할
            std::vector<d2::d_matrix_2<double>> batch_data(num_batches), batch_labels(num_batches);
            for(int i = 0; i < num_batches; ++i){
                batch_data[i] = X.getBatch(B, i*B);
                batch_labels[i] = Y.getBatch(B, i*B);
                printProgressBar(i+1, num_batches, start, "batch loading... (batch " + std::to_string(i+1) + "/" + std::to_string(num_batches) + ")");
            }
            std::cout << std::endl;
            std::cout << "[배치 로드 완료] 총 " << N << "개 데이터, " << num_batches << "개 배치" << std::endl;
            
            // Loss 데이터 저장을 위한 파일 생성
            std::string graphPath = "../graph/" + id;
            fs::create_directories(graphPath);
            std::ofstream epoch_loss_file(graphPath + "/epoch_loss.txt");
            std::ofstream batch_loss_file(graphPath + "/batch_loss.txt");

            // 훈련 루프
            std::string progress_avgloss;
            for(int e = 1; e <= model_info.epoch; e++) {
                double avgloss = 0;
                
                for(int j = 0; j < num_batches; j++){
                    // 순전파
                    auto [output, loss_val] = forward(batch_data[j], batch_labels[j], hs.model_str);
                    
                    avgloss += loss_val;
                    
                    // NaN 체크
                    if(std::isnan(loss_val)){
                        std::cerr << "Loss is NaN at batch " << j+1 << ", epoch " << e << std::endl;
                        std::cerr << "Output (first 10 elements): ";
                        output.cpyToHost();
                        for(int k=0; k<std::min(10, (int)output.size()); ++k) 
                            std::cerr << output.getHostPointer()[k] << " ";
                        std::cerr << std::endl;
                        std::cerr << "Labels (first 10 elements): ";
                        batch_labels[j].cpyToHost();
                        for(int k=0; k<std::min(10, (int)batch_labels[j].size()); ++k) 
                            std::cerr << batch_labels[j].getHostPointer()[k] << " ";
                        std::cerr << std::endl;
                        throw std::runtime_error("Invalid error in loss calc.");
                    }
                    
                    // 역전파
                    backward(output, batch_labels[j], hs.model_str);
                    
                    // 배치별 loss 저장
                    batch_loss_file << e << " " << j+1 << " " << loss_val << std::endl;
                    
                    // 진행 상황 표시
                    std::string progress_batch = "batch" + std::to_string(j+1);
                    std::string progress_loss = "loss:" + std::to_string(loss_val);
                    printProgressBar(e, model_info.epoch, start, progress_avgloss + " | " + progress_batch + " 의 " + progress_loss);
                }
                
                avgloss = avgloss / static_cast<double>(num_batches);
                progress_avgloss = "[epoch" + std::to_string(e+1) + "/" + std::to_string(model_info.epoch) + "의 avgloss]:" + std::to_string(avgloss);
                
                // Epoch별 평균 loss 저장
                epoch_loss_file << e << " " << avgloss << std::endl;
                
            }
            
            // 파일 닫기
            epoch_loss_file.close();
            batch_loss_file.close();
            
            std::cout << std::endl;
            std::cout << "총 학습 시간: "
                      << std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now() - start
                         ).count() << "초" << std::endl;
        }

        void test(){
            auto start = std::chrono::steady_clock::now();
            
            // GOL 데이터 로드 (배치 형태로 직접 로드)
            auto [X, Y] = GOL_2::LoadingDataBatch(test_data_info, hs.model_str);

            int N = X.getRow();      // 전체 데이터 개수
            int input_size = X.getCol();   // 입력 크기 (100)
            int output_size = Y.getCol();  // 출력 크기 (8)
            
            std::cout << "[데이터 로드 완료] " << N << "개 샘플, 입력크기: " << input_size << ", 출력크기: " << output_size << std::endl;

            int B = model_info.batch_size;           // 배치 크기
            int num_batches = (N + B - 1) / B;  // 총 배치 수
            
            // 배치별로 데이터 미리 분할
            std::vector<d2::d_matrix_2<double>> batch_data(num_batches), batch_labels(num_batches);
            for(int i = 0; i < num_batches; ++i){
                batch_data[i] = X.getBatch(B, i*B);
                batch_labels[i] = Y.getBatch(B, i*B);
                printProgressBar(i+1, num_batches, start, "batch loading... (batch " + std::to_string(i+1) + "/" + std::to_string(num_batches) + ")");
            }
            std::cout << std::endl;
            std::cout << "[배치 로드 완료] 총 " << N << "개 데이터, " << num_batches << "개 배치" << std::endl;

            // 결과 저장을 위한 파일 생성
            fs::create_directories("../result/");
            std::ofstream result_file("../result/test_results" + id + "-to_trained_by-" + getDatasetId(using_dataset) + ".txt");
            if (!result_file) {
                throw std::runtime_error("결과 파일을 열 수 없습니다.");
            }

            // 테스트 수행
            for (int i = 0; i < num_batches; ++i) {
                // 배치별로 모델에 입력하고 결과를 얻음
                auto [output, loss] = forward(batch_data[i], batch_labels[i], hs.model_str);
                output.cpyToHost(); // GPU에서 CPU로 데이터 복사
                result_file << "Batch " << i + 1 << ":\n";
                for (int j = 0; j < output.getRow(); ++j) {
                    result_file << "Sample " << j + 1 << ": ";
                    for(int k = 0; k < output.getCol(); ++k) {
                        result_file << output.getHostPointer()[j*output.getCol() + k] << ", ";
                    }
                    result_file << "\n";
                }
                result_file << "Loss: " << loss << "\n";
            }

            result_file.close();
        }

        // 전체 실행 프로세스
        void run() {
            try {
                std::cout << "\n=== GOL CNN Solver 시작 ===" << std::endl;
                
                // CUDA 환경 확인
                if (!checkCudaEnvironment()) {
                    throw std::runtime_error("CUDA 환경을 사용할 수 없습니다.");
                }

                // 현재 설정 출력
                printCurrentConfig();

                std::cout << "\n=== 데이터셋 생성 ===" << std::endl;
                genDataset();

                std::cout << "\n=== 모델 훈련 시작 ===" << std::endl;
                train();
                
                std::cout << "\n=== 훈련 완료! ===" << std::endl;
                
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl;
                throw;
            }
        }

};


int main(){
    try {
        model_id config;
        config.model_name = "mercury_custom";
        config.conv_active = "LReLU";
        config.conv_init = "He";
        config.fc_active = "Tanh";
        config.fc_init = "Xavier";
        config.optimizer = "Adam";
        config.loss = "BCEWithLogits";
        config.epoch = 1000;
        config.batch_size = 50;
        config.learning_rate = 1e-6;

        dataset_id dataset_info;
        dataset_info.seed = 54321;
        dataset_info.sample_quantity = 32000;
        dataset_info.alive_ratio = 0.3;

        dataset_id test_dataset_info;
        test_dataset_info.seed = 98624;
        test_dataset_info.sample_quantity = 500;
        test_dataset_info.alive_ratio = 0.3;

        std::string loaded_model_timestamp = "-2025-08-17_193828"; // 32000개의 샘플로 훈련된 모델의 타임스탬프

        GOLsolver_1 mercury(config, dataset_info, test_dataset_info, false);
        mercury.id = mercury.id + loaded_model_timestamp;

        // 모델 로드
        mercury.loadModel("../model_save/" + mercury.id);

        mercury.test();

    } catch (const std::exception& e) {
        std::cerr << "프로그램 실행 중 오류: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}


