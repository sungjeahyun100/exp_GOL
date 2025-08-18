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
        GOLsolver_1() : GOLsolver_1(1, 1e-6, 1000) {} // 기본값으로 위임

        //간편히 실행해볼 수 있도록 설정을 단순화한 생성자
        GOLsolver_1(int bs, double lr, int ep, bool use_non_default_stream = true) : 
            // handleStream과 다른 멤버들을 먼저 초기화
            hs(),
            act(),
            loss(),
            // 레이어 초기화 (기본 생성자 사용)
            conv1(bs, 1, 10, 10, 8, 3, 3, 0, 0, 1, 1, p2::optType::Adam, d2::InitType::He, lr, use_non_default_stream ? 0 : hs.model_str),
            conv2(bs, 8, 8, 8, 16, 3, 3, 0, 0, 1, 1, p2::optType::Adam, d2::InitType::He, lr, use_non_default_stream ? 0 : hs.model_str),
            conv3(bs, 16, 6, 6, 32, 3, 3, 0, 0, 1, 1, p2::optType::Adam, d2::InitType::He, lr, use_non_default_stream ? 0 : hs.model_str),
            fc1(bs, 4*4*32, 256, p2::optType::Adam, d2::InitType::Xavier, lr, use_non_default_stream ? 0 : hs.model_str),
            fc2(bs, 256, 128, p2::optType::Adam, d2::InitType::Xavier, lr, use_non_default_stream ? 0 : hs.model_str),
            fc3(bs, 128, 64, p2::optType::Adam, d2::InitType::Xavier, lr, use_non_default_stream ? 0 : hs.model_str),
            fc4(bs, 64, 32, p2::optType::Adam, d2::InitType::Xavier, lr, use_non_default_stream ? 0 : hs.model_str),
            fc_out(bs, 32, 8, p2::optType::Adam, d2::InitType::Xavier, lr, use_non_default_stream ? 0 : hs.model_str),
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
        
};

extern "C"{
    // 모델 인스턴스 생성 및 로드
    void* initModel(const char* model_path) {
        GOLsolver_1* model = new GOLsolver_1();
        model->loadModel(model_path);
        return static_cast<void*>(model);
    }
    
    // 예측 함수
    void predict(void* model_ptr, float* input_grid, float* output) {
        if (!model_ptr) return;
        
        GOLsolver_1* model = static_cast<GOLsolver_1*>(model_ptr);
        
        // 입력 데이터 준비
        d2::d_matrix_2<double> input(10, 10);
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                input(i, j) = input_grid[i*10 + j];
            }
        }
        input.cpyToDev(); // CPU에서 GPU로 데이터 복사
        
        // 예측 실행
        auto result = model->forward(input, d2::d_matrix_2<double>());
        result.first.cpyToHost(); // GPU에서 CPU로 데이터 복사
        // 출력 복사
        for (int i = 0; i < 8; i++) {
            output[i] = static_cast<float>(result.first(0, i));
        }
    }
    
    // 메모리 해제
    void freeModel(void* model_ptr) {
        if (model_ptr) {
            GOLsolver_1* model = static_cast<GOLsolver_1*>(model_ptr);
            delete model;
        }
    }
}

