#pragma once
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <cstdint>
#include <d_matrix_2.hpp>
#include <stdexcept>
#include <string>

struct model_id{
    std::string model_name;
    std::string conv_active;
    std::string conv_init;
    std::string fc_active;
    std::string fc_init;
    std::string loss;
    std::string optimizer;
    int conv_layer_count;
    int fc_layer_count;
    int epoch;
    int batch_size;
    double learning_rate;
};

struct dataset_id{
    uint32_t seed;
    uint32_t sample_quantity;
    double alive_ratio;
};

inline std::string getCurrentTimestamp()
{
    auto now = std::chrono::system_clock::now();
    std::time_t t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now = *std::localtime(&t_now);

    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y-%m-%d_%H%M%S");
    return oss.str();
}

inline std::string getModelId(model_id info){
    return info.model_name + "_" +
           "conv" + std::to_string(info.conv_layer_count) + "-" + info.conv_active + "-" + info.conv_init + "_" +
           "fc" + std::to_string(info.fc_layer_count) + "-" + info.fc_active + "-" + info.fc_init + "_" +
           info.loss + "_" +
           info.optimizer + "-" +
           std::to_string(info.epoch) + "-" +
           std::to_string(info.batch_size) + "-" +
           std::to_string(info.learning_rate) + "-" +
           getCurrentTimestamp();
}

inline std::string getModelIdWithoutTimestamp(model_id info){
    return info.model_name + "_" +
           "conv" + std::to_string(info.conv_layer_count) + "-" + info.conv_active + "-" + info.conv_init + "_" +
           "fc" + std::to_string(info.fc_layer_count) + "-" + info.fc_active + "-" + info.fc_init + "_" +
           info.loss + "_" +
           info.optimizer + "-" +
           std::to_string(info.epoch) + "-" +
           std::to_string(info.batch_size) + "-" +
           std::to_string(info.learning_rate);
}

inline std::string getDatasetId(dataset_id info){
    return "database-" + std::to_string(info.seed) + "_" +
           std::to_string(info.sample_quantity) + "_" +
           std::to_string(info.alive_ratio);
}

inline void printProgressBar(int current, int total, std::chrono::steady_clock::time_point startTime, std::string processname) {
    int width = 50;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(width * progress);
    
    auto elapsed = std::chrono::steady_clock::now() - startTime;
    int elapsedSec = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    std::cout << "[";
    for (int i = 0; i < width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% ";
    std::cout << '[' << processname << ']';
    std::cout << "(경과 시간: " << elapsedSec << " ms)                      \r";
    std::cout.flush();
}



// 읽을 idx 파일의 32비트 정수를 빅엔디안으로 파싱
inline uint32_t read_be32(std::ifstream& in) {
    uint8_t b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) throw std::runtime_error("Unexpected EOF while reading header");
    return (uint32_t(b[0]) << 24) |
           (uint32_t(b[1]) << 16) |
           (uint32_t(b[2]) <<  8) |
           (uint32_t(b[3]) <<  0);
}

// MNIST 이미지 로더: ubyte 파일에서 2D float 벡터로 변환
// 픽셀 값은 [0,255] -> [0.0,1.0] 구간 float로 정규화
inline std::vector<std::vector<float>> load_mnist_images(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open image file: " + path);
    uint32_t magic = read_be32(in);
    if (magic != 0x00000803) throw std::runtime_error("Invalid magic for images: " + std::to_string(magic));
    uint32_t N = read_be32(in);
    uint32_t rows = read_be32(in);
    uint32_t cols = read_be32(in);
    std::vector<std::vector<float>> images(N, std::vector<float>(rows * cols));
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < rows * cols; ++j) {
            uint8_t pixel;
            in.read(reinterpret_cast<char*>(&pixel), 1);
            if (!in) throw std::runtime_error("Unexpected EOF while reading pixels");
            images[i][j] = float(pixel) / 255.0f;
        }
    }
    return images;
}



// MNIST 레이블 로더: ubyte 파일에서 1D uint8_t 벡터로 변환
inline std::vector<uint8_t> load_mnist_labels(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open label file: " + path);
    uint32_t magic = read_be32(in);
    if (magic != 0x00000801) throw std::runtime_error("Invalid magic for labels: " + std::to_string(magic));
    uint32_t N = read_be32(in);
    std::vector<uint8_t> labels(N);
    in.read(reinterpret_cast<char*>(labels.data()), N);
    if (!in) throw std::runtime_error("Unexpected EOF while reading labels");
    return labels;
}

// MNIST 이미지 파일(.idx3-ubyte) 로드 -> d_matrix_2<float> (각 컬럼이 하나의 샘플)
inline d_matrix_ver2::d_matrix_2<double> load_images_matrix(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open MNIST image file: " + path);
    uint32_t magic = read_be32(in);
    if (magic != 0x00000803) throw std::runtime_error("Invalid MNIST image magic: " + std::to_string(magic));
    uint32_t N    = read_be32(in);
    uint32_t H    = read_be32(in);
    uint32_t W    = read_be32(in);

    // 행: 픽셀 개수, 열: 샘플 수
    d_matrix_ver2::d_matrix_2<double> mat(N, H * W);
    // host 메모리 채우기
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t p = 0; p < H*W; ++p) {
            uint8_t pixel;
            in.read(reinterpret_cast<char*>(&pixel), 1);
            if (!in) throw std::runtime_error("Unexpected EOF while reading MNIST pixels");
            mat(i, p) = double(pixel) / 255.0f;
        }
    }
    // 디바이스로 복사
    mat.cpyToDev();
    return mat;
}

// MNIST 레이블 파일(.idx1-ubyte) 로드 -> 원-핫 인코딩 d_matrix_2<double> (행: 클래스, 열: 샘플)
inline d_matrix_ver2::d_matrix_2<double> load_labels_matrix(const std::string &path, int num_classes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open MNIST label file: " + path);
    uint32_t magic = read_be32(in);
    if (magic != 0x00000801) throw std::runtime_error("Invalid MNIST label magic: " + std::to_string(magic));
    uint32_t N = read_be32(in);

    // 행: 클래스 수, 열: 샘플 수
    d_matrix_ver2::d_matrix_2<double> mat(N, num_classes);
    // 호스트에 0으로 초기화
    for (int c = 0; c < num_classes; ++c) {
        for (uint32_t i = 0; i < N; ++i) mat(i, c) = 0.0l;
    }
    // 레이블 읽어 채우기
    for (uint32_t i = 0; i < N; ++i) {
        uint8_t lbl;
        in.read(reinterpret_cast<char*>(&lbl), 1);
        if (!in) throw std::runtime_error("Unexpected EOF while reading MNIST labels");
        if (lbl >= num_classes) throw std::runtime_error("MNIST label out of range: " + std::to_string(lbl));
        mat(i, static_cast<int>(lbl)) = 1.0f;
    }
    // 디바이스로 복사
    mat.cpyToDev();
    return mat;
}


