#pragma once
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <cudnn.h>
#include <memory>
#include <fstream>
#include <string>
#include <d_matrix_2.hpp>
#include <thrust/device_ptr.h>

#define CHK_CUDNN(call) if((call)!=CUDNN_STATUS_SUCCESS) throw std::runtime_error(cudnnGetErrorString(call));

namespace d2 = d_matrix_ver2;

namespace perceptron_2 {
    
    cudaError_t err;

    static void getErr(){
        err = cudaGetLastError();  
        if (err != cudaSuccess) {  
          std::cerr << "[CUDA ERR] " << cudaGetErrorString(err)  
                    << " at " << __FILE__ << ":" << __LINE__ << "\n";  
          std::terminate();  
        }
    }

    enum class ActType {
        ReLU,
        LReLU,
        ELU,
        SELU,
        Tanh,
        Identity,
        Sigmoid,
        Softplus,
        Softsign,
        Swish,
        Softmax
    };
    
    enum class LossType {
        MSE,
        CrossEntropy,
        BCEWithLogits
    };

    enum class layerType{
        perceptron,
        conv
    };

    enum class optType{
        Adam,
        SGD
    };

    class handleStream{
        public:
            cudaStream_t model_str;
            handleStream(){
                cudaStreamCreate(&model_str);
            }
            ~handleStream() noexcept {
                cudaStreamDestroy(model_str);
            }
    };

    static __global__ void setGradThreshold(double* d_g, double* d_out, double threshold, int N){ // N = d_g row*col
        int idx = blockDim.x*blockIdx.x+threadIdx.x;
        if(idx >= N) return;

        double gi = d_g[idx];
        const double clip_threshold = threshold; // gradient clipping threshold
        if (isnan(gi) || isinf(gi)) {
            gi = 0.0; // NaN/Inf를 0으로 클리핑
        } else if (gi > clip_threshold) {
            gi = clip_threshold;
        } else if (gi < -clip_threshold) {
            gi = -clip_threshold;
        }
        d_out[idx] = gi;
    }

    class optimizer{
        public:
            virtual ~optimizer() = default;
            virtual void update(d2::d_matrix_2<double>& W, d2::d_matrix_2<double>& B, const d2::d_matrix_2<double>& gW, const d2::d_matrix_2<double>& gB, cudaStream_t str) = 0;
    };

    class SGD : public optimizer {
        double lr, th;
    public:
        explicit SGD(double lr_, double threshold=5.0) : lr(lr_), th(threshold) {}
        void update(d2::d_matrix_2<double>& W, d2::d_matrix_2<double>& B, const d2::d_matrix_2<double>& gW, const d2::d_matrix_2<double>& gB, cudaStream_t str) override {
            d2::d_matrix_2<double> gW_modified(gW.getRow(), gW.getCol(), str);
            d2::d_matrix_2<double> gB_modified(1, gB.getCol(), str);
            int gW_N = gW_modified.size();
            int gB_N = gB_modified.size();
            setGradThreshold<<<(gW_N + 32 -1)/32, 32, 0, str>>>(gW.getDevPointer(), gW_modified.getDevPointer(), th, gW_N);
            setGradThreshold<<<(gB_N + 32 -1)/32, 32, 0, str>>>(gB.getDevPointer(), gB_modified.getDevPointer(), th, gB_N);
            W = d2::matrixPlus(W, d2::ScalaProduct(gW_modified, -lr, str), str);
            B = d2::matrixPlus(B, d2::ScalaProduct(gB_modified, -lr, str), str);
        }
    };

    static __global__ void CalcAdam(
        const double* __restrict__ d_g,
        double*       __restrict__ d_W,
        double*       __restrict__ d_M,
        double*       __restrict__ d_V,
        double        beta1,
        double        beta2,
        double        lr,
        double        eps,
        double        bc1,
        double        bc2,
        int           N,      // 총 원소 수 = row*col
        double threshold
    ) {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx >= N) return;
    
        // 1) gradient clipping 추가 (NaN 방지)
        double gi = d_g[idx];
        const double clip_threshold = threshold; // gradient clipping threshold
        if (isnan(gi) || isinf(gi)) {
            gi = 0.0; // NaN/Inf를 0으로 클리핑
        } else if (gi > clip_threshold) {
            gi = clip_threshold;
        } else if (gi < -clip_threshold) {
            gi = -clip_threshold;
        }
    
        // 2) 모멘텀·분산 업데이트
        double m  = d_M[idx] = d_M[idx]*beta1 + gi*(1.0-beta1);
        double v  = d_V[idx] = d_V[idx]*beta2 + gi*gi*(1.0-beta2);
    
        // 3) 편향 보정된 모멘텀·분산
        double m_hat = m / bc1;
        double v_hat = v / bc2;
    
        // 4) 파라미터 업데이트
        double inv = 1.0 / (sqrt(v_hat) + eps); // 1/(sqrt(v_hat)+eps)
        d_W[idx]   -= lr * m_hat * inv;
    }
    
    class Adam : public optimizer {
        double lr, beta1, beta2, eps, th;
        int t;
        // weight: row×col, bias: 1×col
        d2::d_matrix_2<double> mW, vW;  // same shape as W
        d2::d_matrix_2<double> mB, vB;  // 1×col
        layerType layer;
    public:
        Adam(int row, int col, double lr_, layerType l = layerType::perceptron, cudaStream_t str=0, double b1=0.9, double b2=0.999, double e=1e-8, double threshold=5.0)
          : lr(lr_), beta1(b1), beta2(b2), eps(e), t(0), layer(l), th(threshold),
            mW(row, col, str), vW(row, col, str),
            mB(1, (layer == layerType::conv ? row : col), str),
            vB(1, (layer == layerType::conv ? row : col), str)
        {
            mW.fill(0.0);  vW.fill(0.0);
            mB.fill(0.0);  vB.fill(0.0);
        }
    
        void update(
            d2::d_matrix_2<double>& W,
            d2::d_matrix_2<double>& B,
            const d2::d_matrix_2<double>& gW,
            const d2::d_matrix_2<double>& gB,
            cudaStream_t str
        ) override {
            ++t;
            double bc1 = 1.0 - std::pow(beta1, t);
            double bc2 = 1.0 - std::pow(beta2, t);
    
            int rows = W.getRow(), cols = W.getCol();
            int Nw   = rows * cols;  // total weight elements
            int Nb   = mB.getCol();         // total bias elements (1×col)
    
            const int blockSize = 256;
            int gridW = (Nw + blockSize - 1) / blockSize;
            int gridB = (Nb + blockSize - 1) / blockSize;
    
            // 1) weight 업데이트
            CalcAdam<<<gridW, blockSize, 0, str>>>(
                gW.getDevPointer(),
                W.getDevPointer(),
                mW.getDevPointer(),
                vW.getDevPointer(),
                beta1, beta2, lr, eps,
                bc1, bc2,
                Nw,
                th
            );
            
            // 2) bias 업데이트 (이제 B는 1×col 이므로 N_B = cols)
            CalcAdam<<<gridB, blockSize, 0, str>>>(
                gB.getDevPointer(),
                B.getDevPointer(),
                mB.getDevPointer(),
                vB.getDevPointer(),
                beta1, beta2, lr, eps,
                bc1, bc2,
                Nb,
                th
            );
    
            cudaStreamSynchronize(str);
        }
    
        ~Adam() {}
    };

    static __global__ void addBias(double* d_input, double* d_bias, double* d_output, int row, int col){
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if(idx >= col) return;

        for(int i = 0; i < row; i++) d_output[i*col+idx] = d_input[i*col+idx] + d_bias[idx];
    }

    constexpr int TILE = 32;

    class PerceptronLayer {
        protected:
            int inputSize;
            int outputSize;
            d2::d_matrix_2<double> input;
            d2::d_matrix_2<double> weight;
            d2::d_matrix_2<double> bias;
            d2::d_matrix_2<double> output;
            d2::d_matrix_2<double> delta;
            d2::d_matrix_2<double> gradW;
            d2::d_matrix_2<double> gradB;
            optimizer* opt;

            int deviceId;
            cudaDeviceProp props;
            size_t threadsPerBlock;
            size_t numberOfBlocks;
            
            d2::d_matrix_2<double> w_t;
            d2::d_matrix_2<double> i_t;
            d2::d_matrix_2<double> dX;
            int sample_num;

            bool useBias;
        public:
            inline dim3 grid2d(int rows, int cols) {
              return dim3(
                (cols + TILE-1)/TILE,   // x-direction = #tiles across columns
                (rows + TILE-1)/TILE    // y-direction = #tiles across rows
              );
            }
    
            inline dim3 block2d() { return dim3(TILE, TILE); }

            PerceptronLayer() {}//빈 layer생성자.
    
            PerceptronLayer(int n, int i, int o, optType optimizer, d2::InitType init, double lr, bool useB=true, cudaStream_t str=0)
                : inputSize(i), outputSize(o), sample_num(n),
                  input(n, i, str), weight(i, o, str), bias(1, o, str), dX(n, i, str),
                  output(n, o, str), delta(n, o, str), gradW(i, o, str), gradB(1, o, str){
                cudaGetDevice(&deviceId);
                cudaGetDeviceProperties(&props, deviceId);
                threadsPerBlock = props.maxThreadsPerBlock;
                numberOfBlocks = props.multiProcessorCount;
                weight = d2::InitWeight<double>(i, o, init, str);

                if(useB == true) bias.fill(0.01, str);
                else bias.fill(0.00l, str);

                switch (optimizer){
                    case optType::SGD:
                        opt = new SGD(lr);
                        break;
                    case optType::Adam:
                        opt = new Adam(i, o, lr, layerType::perceptron, str);
                        break;
                    default:
                        break;
                }
            }
    
            const d2::d_matrix_2<double>& getWeight() const { return weight; }
            const d2::d_matrix_2<double>& getDelta()  const { return delta; }
        
            void feedforward(const d2::d_matrix_2<double>& in, cudaStream_t str) {
                input = in;
                d2::MPinKernel<double><<<grid2d(outputSize, sample_num), block2d(), 0, str>>>(input.getDevPointer(), weight.getDevPointer(), output.getDevPointer(), sample_num, outputSize, inputSize);//2^5, 2^5, 2개
                addBias<<<numberOfBlocks, threadsPerBlock, 0, str>>>(output.getDevPointer(), bias.getDevPointer(), output.getDevPointer(), sample_num, outputSize); 
                cudaStreamSynchronize(str);
                
            }
        
            d2::d_matrix_2<double> backprop(const d2::d_matrix_2<double>& ext_delta, const d2::d_matrix_2<double>& act_deriv, cudaStream_t str) {
                d2::d_matrix_2<double> grad_input = ext_delta;
    
                {
                    //delta
                    d2::HPinKernel_1dx<double><<<numberOfBlocks, threadsPerBlock, 0, str>>>(grad_input.getDevPointer(), act_deriv.getDevPointer(), delta.getDevPointer(), sample_num, outputSize);

    
                    //dW
                    d2::TransInKernel<double><<<grid2d(sample_num, inputSize), block2d(), 0, str>>>(input.getDevPointer(), i_t.getDevPointer(), sample_num, inputSize);
                    d2::MPinKernel<double><<<grid2d(outputSize, inputSize), block2d(), 0, str>>>(i_t.getDevPointer(), delta.getDevPointer(), gradW.getDevPointer(), inputSize, outputSize, sample_num);
                    d2::ScalainKernel<double><<<grid2d(inputSize, outputSize), block2d(), 0, str>>>(gradW.getDevPointer(), 1/static_cast<double>(sample_num), gradW.getDevPointer(), inputSize, outputSize);

    
                    //dB
                    d2::reduceRows<double><<<numberOfBlocks, threadsPerBlock, 0, str>>>(delta.getDevPointer(), gradB.getDevPointer(), sample_num, outputSize);
                    d2::ScalainKernel<double><<<grid2d(1, outputSize), block2d(), 0, str>>>(gradB.getDevPointer(), 1/static_cast<double>(sample_num), gradB.getDevPointer(), 1, outputSize);

    
                    //dX
                    int tR = weight.getCol();
                    int tC = weight.getRow();
                    w_t.resize(tC, tR);
                    d2::TransInKernel<double><<<grid2d(tR, tC), block2d(), 0, str>>>(weight.getDevPointer(), w_t.getDevPointer(), tR, tC);
                    d2::MPinKernel<double><<<grid2d(inputSize, sample_num), block2d(), 0, str>>>(delta.getDevPointer(), w_t.getDevPointer(), dX.getDevPointer(), sample_num, inputSize, outputSize);

                }
                opt->update(weight, bias, gradW, gradB, str);
                cudaStreamSynchronize(str);
                return dX;
            }
        
            d2::d_matrix_2<double>& getOutput(){ return output; }
    
            ~PerceptronLayer() noexcept {
                delete opt;  // optimizer 해제
            }

            // ===== 가중치 저장/로드 (바이너리) =====
            // 파일 스트림 버전
            void saveBinary(std::ofstream& out) const {
                const char tag[2] = {'F','C'}; // Fully Connected
                out.write(tag, sizeof(tag));
                int32_t in_sz  = static_cast<int32_t>(inputSize);
                int32_t out_sz = static_cast<int32_t>(outputSize);
                out.write(reinterpret_cast<const char*>(&in_sz), sizeof(in_sz));
                out.write(reinterpret_cast<const char*>(&out_sz), sizeof(out_sz));
                auto writeMatrix = [&](const d2::d_matrix_2<double>& m){
                    int32_t r = static_cast<int32_t>(m.getRow());
                    int32_t c = static_cast<int32_t>(m.getCol());
                    out.write(reinterpret_cast<const char*>(&r), sizeof(r));
                    out.write(reinterpret_cast<const char*>(&c), sizeof(c));
                    const_cast<d2::d_matrix_2<double>&>(m).cpyToHost();
                    size_t bytes = static_cast<size_t>(r) * c * sizeof(double);
                    out.write(reinterpret_cast<const char*>(m.getHostPointer()), static_cast<std::streamsize>(bytes));
                };
                writeMatrix(weight);
                writeMatrix(bias);
                if (!out) throw std::runtime_error("PerceptronLayer::saveBinary failed");
            }

            void loadBinary(std::ifstream& in, cudaStream_t str = 0){
                char tag[2];
                in.read(tag, sizeof(tag));
                if (!in || tag[0] != 'F' || tag[1] != 'C') throw std::runtime_error("Invalid FC layer tag in binary file");
                int32_t in_sz=0, out_sz=0;
                in.read(reinterpret_cast<char*>(&in_sz), sizeof(in_sz));
                in.read(reinterpret_cast<char*>(&out_sz), sizeof(out_sz));
                if (!in) throw std::runtime_error("PerceptronLayer::loadBinary header read error");
                auto readMatrix = [&](d2::d_matrix_2<double>& m){
                    int32_t r=0, c=0;
                    in.read(reinterpret_cast<char*>(&r), sizeof(r));
                    in.read(reinterpret_cast<char*>(&c), sizeof(c));
                    if (!in) throw std::runtime_error("PerceptronLayer::loadBinary matrix header error");
                    m.resize(static_cast<int>(r), static_cast<int>(c), str);
                    size_t bytes = static_cast<size_t>(r) * c * sizeof(double);
                    in.read(reinterpret_cast<char*>(m.getHostPointer()), static_cast<std::streamsize>(bytes));
                    if (!in) throw std::runtime_error("PerceptronLayer::loadBinary matrix payload error");
                    m.cpyToDev(str);
                };
                readMatrix(weight);
                readMatrix(bias);
                inputSize = weight.getRow();
                outputSize = weight.getCol();
                if (!in) throw std::runtime_error("PerceptronLayer::loadBinary failed");
            }

            // 파일 경로 버전
            bool saveBinary(const std::string& filepath) const {
                std::ofstream out(filepath, std::ios::binary);
                if (!out) return false;
                saveBinary(out);
                return static_cast<bool>(out);
            }

            bool loadBinary(const std::string& filepath, cudaStream_t str = 0){
                std::ifstream in(filepath, std::ios::binary);
                if (!in) return false;
                loadBinary(in, str);
                return static_cast<bool>(in);
            }
    };

    // ActivateLayer----------------------------------------------------------------------------------------------------------------------------

    // 활성화 계층
    // 사용법: pushInput()으로 입력, Active()로 활성화 적용, getOutput()으로 결과 반환
    // 지원: ReLU, LReLU, Identity, Sigmoid
    // d_Active: 미분값 반환
    class ActivateLayer{
        public:
            // 활성화 적용 (output = f(input))
            d2::d_matrix_2<double> Active(const d2::d_matrix_2<double>& z, ActType act, cudaStream_t str);
            // 활성화 미분값 반환 (f'(z))
            d2::d_matrix_2<double> d_Active(const d2::d_matrix_2<double>& z, ActType act, cudaStream_t str);
    };
    
    // LossLayer--------------------------------------------------------------------------------------------------------------------------------
    
    // 손실 계층
    // 사용법: pushTarget, pushOutput으로 데이터 입력 후 getLoss(), getGrad() 호출
    // 지원: MSE(평균제곱오차), CrossEntropy(크로스엔트로피)
    // getLoss: loss 반환, getGrad: dL/dz 반환
    class LossLayer{
        public:
            // 손실값 반환
            double getLoss(d2::d_matrix_2<double> out, d2::d_matrix_2<double> target, LossType Loss, cudaStream_t str);
            // 손실 미분 반환
            d2::d_matrix_2<double> getGrad(d2::d_matrix_2<double> out, d2::d_matrix_2<double> target, LossType Loss, cudaStream_t str);
    };
    

    // 활성화 적용 (output = f(input))
    // 지원: ReLU, LReLU, Identity, Sigmoid
    inline d2::d_matrix_2<double> ActivateLayer::Active(const d2::d_matrix_2<double>& z, ActType act, cudaStream_t str = 0){
        switch (act) {
            case ActType::ReLU:
                return d2::MatrixActivate<double, d2::relu>(z, str); break;
            case ActType::LReLU:
                return d2::MatrixActivate<double, d2::lrelu>(z, str); break;
            case ActType::Identity:
                return d2::MatrixActivate<double, d2::Identity>(z, str); break;
            case ActType::Sigmoid:
                return d2::MatrixActivate<double, d2::sigmoid>(z, str); break;
            case ActType::Tanh:
                return d2::MatrixActivate<double, d2::Tanh>(z, str); break;
            case ActType::ELU:
                return d2::MatrixActivate<double, d2::ELU>(z, str); break;
            case ActType::SELU:
                return d2::MatrixActivate<double, d2::SELU>(z, str); break;
            case ActType::Softplus:
                return d2::MatrixActivate<double, d2::Softplus>(z, str); break;
            case ActType::Softsign:
                return d2::MatrixActivate<double, d2::Softsign>(z, str); break;
            case ActType::Swish:
                return d2::MatrixActivate<double, d2::Swish>(z, str); break;
            case ActType::Softmax:
                return d2::softmax_efficient<double>(z, str);
            default:
                throw std::runtime_error("Unsupported ActivationType in perceptronLayer");
        }
    }
    
    // 활성화 함수 미분값 반환 (f'(z))
    // ReLU: 1(x>0), 0(x<=0)
    // LReLU: 1(x>0), 0.01(x<=0)
    // Identity: 1
    // Sigmoid: σ'(x) = σ(x)(1-σ(x))
    inline d2::d_matrix_2<double> ActivateLayer::d_Active(const d2::d_matrix_2<double>& z, ActType act, cudaStream_t str=0) {
        switch (act) {
            case ActType::ReLU:
                return d2::MatrixActivate<double, d2::d_relu>(z, str); 
            case ActType::LReLU:
                return d2::MatrixActivate<double, d2::d_lrelu>(z, str); 
            case ActType::Identity:
                return d2::MatrixActivate<double, d2::d_I>(z, str); 
            case ActType::Sigmoid:
                return d2::MatrixActivate<double, d2::d_sigmoid>(z, str); 
            case ActType::Tanh:
                return d2::MatrixActivate<double, d2::d_tanh>(z, str); 
            case ActType::ELU:
                return d2::MatrixActivate<double, d2::d_ELU>(z, str); 
            case ActType::SELU:
                return d2::MatrixActivate<double, d2::d_SELU>(z, str); 
            case ActType::Softplus:
                return d2::MatrixActivate<double, d2::sigmoid>(z, str);
            case ActType::Softsign:
                return d2::MatrixActivate<double, d2::d_Softsign>(z, str);
            case ActType::Swish:
                return d2::MatrixActivate<double, d2::d_Swish>(z, str);
            case ActType::Softmax:
                // Softmax derivative is handled in lossLayer::getGrad
                // For simplicity, return identity matrix
                return d2::MatrixActivate<double, d2::d_I>(z, str);
            default:
                throw std::runtime_error("Unsupported ActivationType in d_Active");
        }
    }
    

    static __global__ void getCrossEntropyLossInKernel(double* d_p, double* d_t, int N, int C, double* out){
        int rowIdx = blockIdx.x;
        int tid = threadIdx.x;
        
        if(rowIdx >= N) return;
        
        // 공유 메모리를 사용한 reduction
        extern __shared__ double sdata[];
        
        double thread_sum = 0.0;
        
        // 각 스레드가 여러 클래스를 담당할 수 있음
        for(int c = tid; c < C; c += blockDim.x) {
            if(d_t[rowIdx*C + c] == 1.0) {
                thread_sum += ::log(d_p[rowIdx*C + c] + 1e-9) * (-1);
            }
        }
        
        sdata[tid] = thread_sum;
        __syncthreads();
        
        // Reduction
        for(int s = blockDim.x / 2; s > 0; s >>= 1) {
            if(tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        
        if(tid == 0) {
            out[rowIdx] = sdata[0];
        }
    }


    static __global__ void getBinaryCrossEntropyLossInKernel(double* d_p, double* d_t, int N, int C, double* out){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = N * C;
        
        if(idx >= total) return;
        
        double p = d_p[idx];
        double t = d_t[idx];
        
        // BCE: -[y*log(p) + (1-y)*log(1-p)]
        out[idx] = -(t * ::log(p + 1e-9) + (1.0 - t) * ::log(1.0 - p + 1e-9));
    }
    
    // 손실값 반환
    // MSE: L = 1/n Σ(y-p)^2
    // CrossEntropy: L = -Σ y log(softmax(p))
    // BCEWithLogits: L = (1/N) Σ [ max(z,0) - z*y + log(1+exp(-|z|)) ] (sum over classes per sample)
    inline double LossLayer::getLoss(d2::d_matrix_2<double> out, d2::d_matrix_2<double> target, LossType Loss, cudaStream_t str = 0) {
        switch (Loss) {
            case LossType::MSE: {
                int N = out.getRow();
                int C = out.getCol();
    
                auto idx_begin = thrust::make_counting_iterator(0);
                auto idx_end   = thrust::make_counting_iterator(N*C);
    
                thrust::device_ptr<double> o_dev(out.getDevPointer());
                thrust::device_ptr<double> t_dev(target.getDevPointer());
                double* o_ptr = thrust::raw_pointer_cast(o_dev);
                double* t_ptr = thrust::raw_pointer_cast(t_dev);
    
                double sum = thrust::transform_reduce(
                    thrust::cuda_cub::par.on(str),
                    idx_begin, idx_end,
    
                    [o_ptr, t_ptr] __host__ __device__ (int i) -> double {
                        double d = o_ptr[i] - t_ptr[i];
                        return d * d;
                    },
    
                    0.0,
                    thrust::plus<double>()
                );
                cudaStreamSynchronize(str);
                return sum / (static_cast<double>(N*C));
            }
    
            case LossType::CrossEntropy: {
                int N = out.getRow();    // 샘플 수
                int C = out.getCol();    // 클래스 수
            
                d2::d_matrix_2<double> out_result(N, 1, str);

                // (2) 디바이스 포인터 확보
                double* out_ptr = out_result.getDevPointer();
                double* p_ptr = out.getDevPointer();
                double* t_ptr = target.getDevPointer();

                thrust::device_ptr<double> out_dev(out_ptr);
            
                auto idx_begin = thrust::make_counting_iterator(0);
                auto idx_end   = thrust::make_counting_iterator(N);

                int shared_mem_size = 512 * sizeof(double);
                getCrossEntropyLossInKernel<<<N, 512, shared_mem_size, str>>>(p_ptr, t_ptr, N, C, out_ptr);

                double sum = thrust::transform_reduce(
                    thrust::cuda_cub::par.on(str),
                    idx_begin, idx_end,
                    [out_ptr] __host__ __device__ (int idx) -> double {
                        return out_ptr[idx];
                    },
                    0.0, thrust::plus<double>()
                );
                cudaStreamSynchronize(str);
                // (3) 배치 평균
                return sum / static_cast<double>(N*C);
            }

            case LossType::BCEWithLogits: {
                int N = out.getRow();
                int C = out.getCol();
                int total = N * C;

                d2::d_matrix_2<double> loss_per_element(N, C, str);
                auto s = d2::MatrixActivate<double, d2::sigmoid>(out, str); // sigmoid(z)

                double* z_ptr = s.getDevPointer();
                double* y_ptr = target.getDevPointer();
                double* loss_ptr = loss_per_element.getDevPointer();

                int thread = 256;
                int block = (total + thread - 1) / thread;
                getBinaryCrossEntropyLossInKernel<<<block, thread, 0, str>>>(z_ptr, y_ptr, N, C, loss_ptr);

                auto idx_begin = thrust::make_counting_iterator(0);
                auto idx_end   = thrust::make_counting_iterator(N);

                 double sum = thrust::reduce(thrust::cuda_cub::par.on(str), 
                               thrust::device_ptr<double>(loss_ptr),
                               thrust::device_ptr<double>(loss_ptr + total),
                               0.0, thrust::plus<double>());
                cudaStreamSynchronize(str);
                // 평균은 배치 크기 N으로 나눔 (클래스 합은 샘플당 합산)
                return sum / static_cast<double>(total);
            }
    
            default:
                throw std::runtime_error("Unsupported LossType in getLoss");
        }
    }
    // 손실 미분 반환
    // MSE: dL/dz = 2(y-p)
    // CrossEntropy: dL/dz = softmax(p) - y
    inline d2::d_matrix_2<double> LossLayer::getGrad(d2::d_matrix_2<double> out, d2::d_matrix_2<double> target, LossType Loss, cudaStream_t str = 0) {
    
        switch (Loss) {
            case LossType::MSE: {
                // L = (1/N) Σ (o - t)^2  이므로  dL/dz = 2*(o - t)/N
                int N = out.getRow();
                // diff = output - target
                d2::d_matrix_2<double> diff = matrixPlus(out, ScalaProduct(target, -1.0, str), str);
                auto result = ScalaProduct(diff, 2.0 / static_cast<double>(N), str);
                cudaStreamSynchronize(str);
                return result;
            }
            case LossType::CrossEntropy: {
                int N = out.getRow();
                // 2) 소프트맥스 확률 계산
                d2::d_matrix_2<double> p = softmax_efficient(out, str);
    
                // 3) gradient = (p - y) / N  -- 중요: N으로 정규화 필요!
                d2::d_matrix_2<double> grad = matrixPlus(p, ScalaProduct(target, -1.0, str), str);
                auto result = ScalaProduct(grad, 1.0 / static_cast<double>(N), str);
                cudaStreamSynchronize(str);
                return result;
            }

            case LossType::BCEWithLogits: {
                int N = out.getRow();

                auto s = d2::MatrixActivate<double, d2::sigmoid>(out, str); // sigmoid(z)
                // grad = (s - y) / N
                d2::d_matrix_2<double> grad = matrixPlus(s, ScalaProduct(target, -1.0, str), str);
                auto result = ScalaProduct(grad, 1.0 / static_cast<double>(N), str);
                cudaStreamSynchronize(str);
                return result;
            }
    
            default:
                throw std::runtime_error("Unsupported LossType in getGrad");
        }
    }

    class convLayer {
        // cuDNN 오브젝트
        cudnnHandle_t                _handle;
        cudnnTensorDescriptor_t      _xDesc, _yDesc, _biasDesc;
        cudnnFilterDescriptor_t      _wDesc;
        cudnnConvolutionDescriptor_t _convDesc;
        cudnnConvolutionFwdAlgo_t    _bestAlgo;
        cudnnConvolutionBwdDataAlgo_t _bestBwdDAlgo;
        cudnnConvolutionBwdFilterAlgo_t _bestBwdFAlgo;
        size_t                       _workspaceBytes;
        void*                        _workspace = nullptr;
        
        int deviceId;
        cudaDeviceProp props;
        size_t threadsPerBlock;
        size_t numberOfBlocks;
    
        // 호스트/디바이스 저장용
        int N,C,H,W,K,R,S,pad_h,pad_w,stride_h,stride_w;
        d2::d_matrix_2<double> input, kernel, bias;
        d2::d_matrix_2<double> output, delta, gradW, gradB, dX;
    
        optimizer* opt;
    
      public:
        
        convLayer(){}//빈 layer생성자.

        convLayer(int N_, int C_, int H_, int W_,
                  int K_, int R_, int S_,
                  int pad_h_, int pad_w_,
                  int stride_h_, int stride_w_,
                  optType o, d2::InitType init,double lr, cudaStream_t str=0)
          : N(N_), C(C_), H(H_), W(W_)
          , K(K_), R(R_), S(S_)
          , pad_h(pad_h_), pad_w(pad_w_)
          , stride_h(stride_h_), stride_w(stride_w_)
          , input(N_, C_*H_*W_)
          , kernel(K_, C_*R_*S_)
          , bias(1, K_)
          , output(N_, K_*((H_+2*pad_h_-R_)/stride_h_+1)*((W_+2*pad_w_-S_)/stride_w_+1))
          , delta(output.getRow(), output.getCol())
          , gradW(kernel.getRow(), kernel.getCol())
          , gradB(bias.getRow(), bias.getCol())
          , dX(N_, C_*H_*W_)
        {
            cudaGetDevice(&deviceId);
            cudaGetDeviceProperties(&props, deviceId);
            threadsPerBlock = props.maxThreadsPerBlock;
            numberOfBlocks = props.multiProcessorCount;

            kernel = d2::InitWeight<double>(K_, C_*R_*S_, init, str);
            bias.fill(0.0, str);  // bias를 0으로 초기화 (더 안정적)

            switch (o)
            {
                case optType::SGD:
                    opt = new SGD(lr);
                    break;
                case optType::Adam:
                    opt = new Adam(K_, C_*R_*S_, lr, layerType::conv, str);
                    break;
                default:
                    break;
            }

            // — cuDNN 생략 없이 에러 체크까지 —
            CHK_CUDNN(cudnnCreate(&_handle));
            CHK_CUDNN(cudnnCreateTensorDescriptor(&_xDesc));
            CHK_CUDNN(cudnnCreateTensorDescriptor(&_yDesc));
            CHK_CUDNN(cudnnCreateTensorDescriptor(&_biasDesc));
            CHK_CUDNN(cudnnCreateFilterDescriptor(&_wDesc));
            CHK_CUDNN(cudnnCreateConvolutionDescriptor(&_convDesc));

            CHK_CUDNN(cudnnSetStream(_handle, str));
            CHK_CUDNN(cudnnSetTensor4dDescriptor(_xDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N,C,H,W));
            CHK_CUDNN(cudnnSetFilter4dDescriptor(_wDesc,
                CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K,C,R,S));
            CHK_CUDNN(cudnnSetConvolution2dDescriptor(_convDesc,
                pad_h, pad_w, stride_h, stride_w, 1,1,
                CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE));
      
            int outN,outC,outH,outW;
            CHK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
                _convDesc, _xDesc, _wDesc, &outN,&outC,&outH,&outW));
            CHK_CUDNN(cudnnSetTensor4dDescriptor(_yDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, outN,outC,outH,outW));
            CHK_CUDNN(cudnnSetTensor4dDescriptor(_biasDesc,
                CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1,outC,1,1));
      
            // fastest algo 한 개만 뽑기 (수정:알고리즘 여려개 뽐기로 수정함)
            int algoCount = 0;
            cudnnConvolutionFwdAlgoPerf_t perf[ CUDNN_CONVOLUTION_FWD_ALGO_COUNT ];
            CHK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(_handle, _xDesc, _wDesc, _convDesc, _yDesc, CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &algoCount, perf));
            _bestAlgo      = perf[0].algo;
            _workspaceBytes= perf[0].memory;

            cudnnConvolutionBwdFilterAlgoPerf_t perfF[ CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT ];
            int returnedF = 0;
            CHK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(_handle, _xDesc, _yDesc, _convDesc, _wDesc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT, &returnedF, perfF));
            _workspaceBytes  = max(_workspaceBytes, perfF[0].memory);
  
            cudnnConvolutionBwdDataAlgoPerf_t perfD[ CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT ];
            int returnedD = 0;
            CHK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(_handle, _wDesc, _yDesc, _convDesc, _xDesc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT, &returnedD, perfD));
            _workspaceBytes  = max(_workspaceBytes, perfD[0].memory);

            _bestBwdFAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;  // 기본값
            for(int i = 0; i < returnedF; ++i) {
                if (perfF[i].status == CUDNN_STATUS_SUCCESS && perfF[i].memory <= _workspaceBytes) {
                    _bestBwdFAlgo = perfF[i].algo;
                    _workspaceBytes = std::max(_workspaceBytes, perfF[i].memory);
                    break;
                }
            }
            _bestBwdDAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;  // 기본값
            for(int i = 0; i < returnedD; ++i) {
                if (perfD[i].status == CUDNN_STATUS_SUCCESS && perfD[i].memory <= _workspaceBytes) {
                    _bestBwdDAlgo = perfD[i].algo;
                    _workspaceBytes = std::max(_workspaceBytes, perfD[i].memory);
                    break;
                }
            }
  
            if (_workspaceBytes > 0) {
            err = cudaMallocAsync(&_workspace, _workspaceBytes, str);
              if (err != cudaSuccess) {
                throw std::runtime_error(std::string("cudaMalloc failed for workspace: ") + cudaGetErrorString(err));
              }
              // workspace 메모리 초기화 (NaN 방지)
              cudaMemsetAsync(_workspace, 0, _workspaceBytes, str);
            }
  
            std::cout << "[conv] fwdAlgo=" << _bestAlgo
            << " bwdF=" << _bestBwdFAlgo
            << " bwdD=" << _bestBwdDAlgo
            << " workspace=" << (_workspaceBytes/1024/1024) << "MB\n";
        }

        inline dim3 grid2d(int rows, int cols) {
          return dim3(
            (cols + TILE-1)/TILE,   // x-direction = #tiles across columns
            (rows + TILE-1)/TILE    // y-direction = #tiles across rows
          );
        }

        inline dim3 block2d() { return dim3(TILE, TILE); }
    
        // 순전파 (호스트에서 호출)
        // x: (N × C×H×W)  kernel: (K × C×R×S)  bias: (1×K×1×1)
        d2::d_matrix_2<double> forward(const d2::d_matrix_2<double>& x_dev, cudaStream_t str) {
            // 1) 입력 복사
            input = x_dev;
            // 2) convolution
            const double alpha=1.0, beta=0.0;
            CHK_CUDNN(cudnnSetStream(_handle, str));
            CHK_CUDNN(cudnnConvolutionForward(
                _handle, &alpha,
                _xDesc, input.getDevPointer(),
                _wDesc, kernel.getDevPointer(),
                _convDesc, _bestAlgo,
                _workspace, _workspaceBytes,
                &beta,    _yDesc, output.getDevPointer()));
            // 3) bias 추가
            CHK_CUDNN(cudnnAddTensor(
                _handle, &alpha,
                _biasDesc, bias.getDevPointer(),
                &alpha,
                _yDesc, output.getDevPointer()));
            // 동기화 제거 - 스트림을 사용하므로 불필요
            return output;  // (N × K×Ho×Wo)
        }
    
        // 역전파 (호스트에서 호출)
        // next==nullptr 이면 맨 앞 레이어
        // dY: Loss 에서 넘어온 dL/dZ  act_deriv: 활성화 함수 도함수(z)
        d2::d_matrix_2<double> backward(const d2::d_matrix_2<double>& dY_dev, const d2::d_matrix_2<double>& act_deriv_dev, cudaStream_t str)
        {
            d2::d_matrix_2<double> grad_input = dY_dev;
            // 2) 활성화 미분 곱하기
            int Rr=delta.getRow(), Cc=delta.getCol();
            d2::HPinKernel_1dx<double><<<numberOfBlocks, threadsPerBlock, 0, str>>>(grad_input.getDevPointer(), act_deriv_dev.getDevPointer(), delta.getDevPointer(), Rr, Cc);
      
            // 3) gradW 계산
            const double alpha=1.0, beta=0.0; // alpha를 1.0으로 변경 (gradient scaling 완화)
            CHK_CUDNN(cudnnSetStream(_handle, str));
            CHK_CUDNN(cudnnConvolutionBackwardFilter(
                _handle, &alpha,
                _xDesc, input.getDevPointer(),
                _yDesc, delta.getDevPointer(),
                _convDesc, _bestBwdFAlgo,
                _workspace, _workspaceBytes,
                &beta,
                _wDesc, gradW.getDevPointer()));
            // 4) gradB 계산
            CHK_CUDNN(cudnnSetStream(_handle, str));
            CHK_CUDNN(cudnnConvolutionBackwardBias(
                _handle, &alpha,
                _yDesc, delta.getDevPointer(),
                &beta,
                _biasDesc, gradB.getDevPointer()));
            // 5) dX 계산 (이전 레이어로 전파할 델타)
            CHK_CUDNN(cudnnSetStream(_handle, str));
            CHK_CUDNN(cudnnConvolutionBackwardData(
                _handle, &alpha,
                _wDesc, kernel.getDevPointer(),
                _yDesc, delta.getDevPointer(),
                _convDesc, _bestBwdDAlgo,
                _workspace, _workspaceBytes,
                &beta,
                _xDesc, dX.getDevPointer()));
            // 6) 파라미터 업데이트
            opt->update(kernel, bias, gradW, gradB, str);
            cudaStreamSynchronize(str);
      
            // 최종적으로 이전 레이어로 보낼 델타 반환
            // (already device 에 있으므로 Host 로 안 옮겨도 됩니다)
            return dX;
        }
    
        ~convLayer(){
            if(_workspace)  cudaFree(_workspace);
            cudnnDestroyConvolutionDescriptor(_convDesc);
            cudnnDestroyFilterDescriptor(_wDesc);
            cudnnDestroyTensorDescriptor(_xDesc);
            cudnnDestroyTensorDescriptor(_yDesc);
            cudnnDestroyTensorDescriptor(_biasDesc);
            cudnnDestroy(_handle);
            delete opt;  // optimizer 해제
        }
    
        // 필요하다면 getter들도 추가
        const d2::d_matrix_2<double>& getOutput() const { return output; }
        const d2::d_matrix_2<double>& getDelta () const { return delta;  }
        const d2::d_matrix_2<double>& getGradW () const { return gradW;  }
        const d2::d_matrix_2<double>& getGradB () const { return gradB;  }

        // ===== 가중치 저장/로드 (바이너리) =====
        void saveBinary(std::ofstream& out) const {
            const char tag[2] = {'C','V'}; // Convolution
            out.write(tag, sizeof(tag));
            int32_t K_ = static_cast<int32_t>(kernel.getRow());
            int32_t CRS_= static_cast<int32_t>(kernel.getCol());
            out.write(reinterpret_cast<const char*>(&K_), sizeof(K_));
            out.write(reinterpret_cast<const char*>(&CRS_), sizeof(CRS_));
            auto writeMatrix = [&](const d2::d_matrix_2<double>& m){
                int32_t r = static_cast<int32_t>(m.getRow());
                int32_t c = static_cast<int32_t>(m.getCol());
                out.write(reinterpret_cast<const char*>(&r), sizeof(r));
                out.write(reinterpret_cast<const char*>(&c), sizeof(c));
                const_cast<d2::d_matrix_2<double>&>(m).cpyToHost();
                size_t bytes = static_cast<size_t>(r) * c * sizeof(double);
                out.write(reinterpret_cast<const char*>(m.getHostPointer()), static_cast<std::streamsize>(bytes));
            };
            writeMatrix(kernel);
            writeMatrix(bias);
            if (!out) throw std::runtime_error("convLayer::saveBinary failed");
        }

        void loadBinary(std::ifstream& in, cudaStream_t str = 0){
            char tag[2];
            in.read(tag, sizeof(tag));
            if (!in || tag[0] != 'C' || tag[1] != 'V') throw std::runtime_error("Invalid Conv layer tag in binary file");
            int32_t K_=0, CRS_=0;
            in.read(reinterpret_cast<char*>(&K_), sizeof(K_));
            in.read(reinterpret_cast<char*>(&CRS_), sizeof(CRS_));
            if (!in) throw std::runtime_error("convLayer::loadBinary header read error");
            auto readMatrix = [&](d2::d_matrix_2<double>& m){
                int32_t r=0, c=0;
                in.read(reinterpret_cast<char*>(&r), sizeof(r));
                in.read(reinterpret_cast<char*>(&c), sizeof(c));
                if (!in) throw std::runtime_error("convLayer::loadBinary matrix header error");
                m.resize(static_cast<int>(r), static_cast<int>(c), str);
                size_t bytes = static_cast<size_t>(r) * c * sizeof(double);
                in.read(reinterpret_cast<char*>(m.getHostPointer()), static_cast<std::streamsize>(bytes));
                if (!in) throw std::runtime_error("convLayer::loadBinary matrix payload error");
                m.cpyToDev(str);
            };
            readMatrix(kernel);
            readMatrix(bias);
            if (!in) throw std::runtime_error("convLayer::loadBinary failed");
        }

        bool saveBinary(const std::string& filepath) const {
            std::ofstream out(filepath, std::ios::binary);
            if (!out) return false;
            saveBinary(out);
            return static_cast<bool>(out);
        }

        bool loadBinary(const std::string& filepath, cudaStream_t str = 0){
            std::ifstream in(filepath, std::ios::binary);
            if (!in) return false;
            loadBinary(in, str);
            return static_cast<bool>(in);
        }
    };

}//namespace perceptron_2

