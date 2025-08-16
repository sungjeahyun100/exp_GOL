/**
 * Game of Life Database Generator - d_matrix_2 version
 * Conway's Game of Life 패턴 생성 및 라벨링을 위한 데이터베이스 생성기
 */

#include "GOLdatabase_2.hpp"

namespace GOL_2 {
    using namespace d_matrix_ver2;

    #define MAXGEN 2500
    namespace fs = std::filesystem;

    const int BOARDWIDTH = 100;
    const int BOARDHEIGHT = 100;
    const int WIDTH = 10;
    const int HEIGHT = 10;

    // CUDA kernel: Game of Life 다음 세대 계산
    __global__ void nextGenKernel(int* current, int* next, int width, int height) {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < height && j < width) {
            int alive = 0;
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    if(dx == 0 && dy == 0) continue;
                    int ni = i + dx;
                    int nj = j + dy;
                    if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        alive += current[ni * width + nj];
                    }
                }
            }

            int idx = i * width + j;
            if (current[idx] == 1) {
                next[idx] = (alive == 2 || alive == 3) ? 1 : 0;
            } else {
                next[idx] = (alive == 3) ? 1 : 0;
            }
        }
    }

    d_matrix_2<int> nextGen(const d_matrix_2<int>& current, cudaStream_t str) {
        d_matrix_2<int> next(current.getRow(), current.getCol(), str);
        int* d_curr = current.getDevPointer();
        int* d_next = next.getDevPointer();

        dim3 blockSize(32, 32);
        dim3 gridSize((current.getCol() + 31) / 32, (current.getRow() + 31) / 32);

        nextGenKernel<<<gridSize, blockSize, 0, str>>>(d_curr, d_next, current.getCol(), current.getRow());
        cudaStreamSynchronize(str);
        
        return next;
    }

    __global__ void placePatternKernel(int* board, int* pattern, int fullHeight, int fullWidth,
        int patternHeight, int patternWidth,
        int startRow, int startCol) {
        int i = blockIdx.y * blockDim.y + threadIdx.y; // pattern row
        int j = blockIdx.x * blockDim.x + threadIdx.x; // pattern col

        if (i < patternHeight && j < patternWidth) {
            int boardIdx = (startRow + i) * fullWidth + (startCol + j);
            int patternIdx = i * patternWidth + j;
            board[boardIdx] = pattern[patternIdx];
        }
    }

    d_matrix_2<int> generateFixedRatioPatternWithPadding(int fullHeight, int fullWidth, int patternHeight, int patternWidth, double aliveRatio, cudaStream_t str) {
        // 1. CPU에서 pattern 배열 셔플
        int totalPatternCells = patternHeight * patternWidth;
        int aliveCells = static_cast<int>(totalPatternCells * aliveRatio);
        std::vector<int> host_pattern(totalPatternCells, 0);
        std::fill_n(host_pattern.begin(), aliveCells, 1);

        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::shuffle(host_pattern.begin(), host_pattern.end(), gen);

        // 2. GPU 메모리로 복사
        thrust::device_vector<int> d_pattern = host_pattern;
        d_matrix_2<int> board(fullHeight, fullWidth, str); // 전체 보드
        board.fill(0, str); // 0으로 초기화

        int startRow = (fullHeight - patternHeight) / 2;
        int startCol = (fullWidth - patternWidth) / 2;

        // 3. 커널로 중앙에 패턴 복사
        dim3 blockSize(16, 16);
        dim3 gridSize((patternWidth + 15) / 16, (patternHeight + 15) / 16);

        placePatternKernel<<<gridSize, blockSize, 0, str>>>(
            board.getDevPointer(), 
            thrust::raw_pointer_cast(d_pattern.data()), 
            fullHeight, fullWidth, 
            patternHeight, patternWidth, 
            startRow, startCol
        );

        cudaStreamSynchronize(str);
        return board;
    }

    d_matrix_2<int> generateFixedRatioPatternWithSeed(int fullHeight, int fullWidth, int patternHeight, int patternWidth, double aliveRatio, int seed, cudaStream_t str) {
        // 1. CPU에서 pattern 배열 셔플
        int totalPatternCells = patternHeight * patternWidth;
        int aliveCells = static_cast<int>(totalPatternCells * aliveRatio);
        std::vector<int> host_pattern(totalPatternCells, 0);
        std::fill_n(host_pattern.begin(), aliveCells, 1);

        std::mt19937_64 gen(seed);
        std::shuffle(host_pattern.begin(), host_pattern.end(), gen);

        // 2. GPU 메모리로 복사
        thrust::device_vector<int> d_pattern = host_pattern;
        d_matrix_2<int> board(fullHeight, fullWidth, str); // 전체 보드
        board.fill(0, str); // 0으로 초기화

        int startRow = (fullHeight - patternHeight) / 2;
        int startCol = (fullWidth - patternWidth) / 2;

        // 3. 커널로 중앙에 패턴 복사
        dim3 blockSize(16, 16);
        dim3 gridSize((patternWidth + 15) / 16, (patternHeight + 15) / 16);

        placePatternKernel<<<gridSize, blockSize, 0, str>>>(
            board.getDevPointer(), 
            thrust::raw_pointer_cast(d_pattern.data()), 
            fullHeight, fullWidth, 
            patternHeight, patternWidth, 
            startRow, startCol
        );

        cudaStreamSynchronize(str);
        return board;
    }

    __global__ void countAliveKernel(int* mat, int* partialSums, int totalSize) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        int localSum = 0;

        for (int i = tid; i < totalSize; i += stride) {
            localSum += mat[i];
        }

        if (tid < totalSize) {
            partialSums[tid] = localSum;
        }
    }

    // Fast alive-cell counter using thrust reduction (no per-iteration allocations)
    int countAlive(const d_matrix_2<int>& mat, cudaStream_t str) {
        int totalSize = mat.getRow() * mat.getCol();
        thrust::device_ptr<const int> ptr(mat.getDevPointer());
        // Sum 0/1 values directly on device
        int total = thrust::reduce(
            thrust::cuda::par.on(str),
            ptr, ptr + totalSize,
            0, thrust::plus<int>()
        );
        // Ensure reduction is complete before returning
        cudaStreamSynchronize(str);
        return total;
    }

    // Optimized simulation using ping-pong device buffers (avoids per-step allocations)
    int simulateAndLabel(const d_matrix_2<int>& initialPattern, int fileId, cudaStream_t str) {
        const int H = initialPattern.getRow();
        const int W = initialPattern.getCol();

        // Allocate a second device buffer for next state (track separately)
        int* d_curr = initialPattern.getDevPointer();
        int* d_next = nullptr;
        int* d_alt  = nullptr; // always points to allocated buffer to free later
        cudaMallocAsync(&d_alt, sizeof(int) * H * W, str);
        d_next = d_alt;

        std::deque<int> history; // 최근 50개 alive 수 저장
        const int window = 50;

        int constantCount = 0;
        int prev = -1;
        bool strictlyIncreasing = true;
        int gen = 0;

        dim3 blockSize(32, 32);
        dim3 gridSize((W + 31) / 32, (H + 31) / 32);

        while (gen < MAXGEN) {
            // Count alive cells on device
            int alive = 0;
            {
                thrust::device_ptr<const int> ptr(d_curr);
                alive = thrust::reduce(thrust::cuda::par.on(str), ptr, ptr + H * W, 0, thrust::plus<int>());
                cudaStreamSynchronize(str);
            }

            // history 갱신
            if (static_cast<int>(history.size()) >= window) history.pop_front();
            history.push_back(alive);

            if (prev == alive) constantCount++;
            else constantCount = 0;

            if (prev != -1 && alive <= prev) strictlyIncreasing = false;
            if (constantCount >= 100 || (strictlyIncreasing && gen >= 100)) break;

            prev = alive;

            // Next generation in-place to d_next, then swap pointers
            nextGenKernel<<<gridSize, blockSize, 0, str>>>(d_curr, d_next, W, H);
            cudaStreamSynchronize(str);
            std::swap(d_curr, d_next);
            gen++;
        }

        // Final alive count on the current buffer
        int final_alive = 0;
        {
            thrust::device_ptr<const int> ptr(d_curr);
            final_alive = thrust::reduce(thrust::cuda::par.on(str), ptr, ptr + H * W, 0, thrust::plus<int>());
            cudaStreamSynchronize(str);
        }

    if (d_alt) cudaFreeAsync(d_alt, str);
        cudaStreamSynchronize(str);
        return final_alive;
    }

    d_matrix_2<int> simulateAndLabelingtopattern(const d_matrix_2<int>& initialPattern, int fileId, cudaStream_t str) {
        const int H = initialPattern.getRow();
        const int W = initialPattern.getCol();

        int* d_curr = initialPattern.getDevPointer();
        int* d_next = nullptr;
        int* d_alt  = nullptr;
        cudaMallocAsync(&d_alt, sizeof(int) * H * W, str);
        d_next = d_alt;

        std::deque<int> history; // 최근 50개 alive 수 저장
        const int window = 50;

        int constantCount = 0;
        int prev = -1;
        bool strictlyIncreasing = true;
        int gen = 0;

        dim3 blockSize(32, 32);
        dim3 gridSize((W + 31) / 32, (H + 31) / 32);

        while (gen < MAXGEN) {
            int alive = 0;
            {
                thrust::device_ptr<const int> ptr(d_curr);
                alive = thrust::reduce(thrust::cuda::par.on(str), ptr, ptr + H * W, 0, thrust::plus<int>());
                cudaStreamSynchronize(str);
            }

            if (static_cast<int>(history.size()) >= window) history.pop_front();
            history.push_back(alive);

            if (prev == alive) constantCount++;
            else constantCount = 0;

            if (prev != -1 && alive <= prev) strictlyIncreasing = false;
            if (constantCount >= 100 || (strictlyIncreasing && gen >= 100)) break;

            prev = alive;

            nextGenKernel<<<gridSize, blockSize, 0, str>>>(d_curr, d_next, W, H);
            cudaStreamSynchronize(str);
            std::swap(d_curr, d_next);
            gen++;
        }

        // Copy final board into d_matrix_2
        d_matrix_2<int> final_board(H, W, str);
        cudaMemcpyAsync(final_board.getDevPointer(), d_curr, sizeof(int) * H * W, cudaMemcpyDeviceToDevice, str);
        cudaStreamSynchronize(str);

        if (d_alt) cudaFreeAsync(d_alt, str);
        cudaStreamSynchronize(str);
        return final_board;
    }

    // 최적화: 단일 시뮬레이션으로 패턴과 라벨을 동시에 반환
    std::pair<d_matrix_2<int>, int> simulateAndGetBoth(const d_matrix_2<int>& initialPattern, int fileId, cudaStream_t str) {
        const int H = initialPattern.getRow();
        const int W = initialPattern.getCol();

        int* d_curr = initialPattern.getDevPointer();
        int* d_next = nullptr;
        int* d_alt  = nullptr;
        cudaMallocAsync(&d_alt, sizeof(int) * H * W, str);
        d_next = d_alt;

        std::deque<int> history; // 최근 50개 alive 수 저장
        const int window = 50;

        int constantCount = 0;
        int prev = -1;
        bool strictlyIncreasing = true;
        int gen = 0;

        dim3 blockSize(32, 32);
        dim3 gridSize((W + 31) / 32, (H + 31) / 32);

        while (gen < MAXGEN) {
            int alive = 0;
            {
                thrust::device_ptr<const int> ptr(d_curr);
                alive = thrust::reduce(thrust::cuda::par.on(str), ptr, ptr + H * W, 0, thrust::plus<int>());
                cudaStreamSynchronize(str);
            }

            if (static_cast<int>(history.size()) >= window) history.pop_front();
            history.push_back(alive);

            if (prev == alive) constantCount++;
            else constantCount = 0;

            if (prev != -1 && alive <= prev) strictlyIncreasing = false;

            // 더 빠른 조기 종료: 안정화 감지 개선
            if (constantCount >= 30) break;  // 30 세대 연속 동일 → 안정화
            if (strictlyIncreasing && gen >= 50) break;  // 50 세대 연속 증가 → 발산
            if (alive == 0) break;  // 모든 셀 사망 → 소멸

            // 진동 패턴 감지: 최근 기록에서 반복 확인
            if (history.size() >= 20) {
                bool oscillating = true;
                int period = 2;  // 2주기 진동 확인
                for (int i = 0; i < 10 && oscillating; i++) {
                    if (history[history.size()-1-i] != history[history.size()-1-i-period]) {
                        oscillating = false;
                    }
                }
                if (oscillating) break;  // 진동 패턴 감지 시 조기 종료
            }

            prev = alive;
            nextGenKernel<<<gridSize, blockSize, 0, str>>>(d_curr, d_next, W, H);
            cudaStreamSynchronize(str);
            std::swap(d_curr, d_next);
            gen++;
        }

        int final_count = 0;
        {
            thrust::device_ptr<const int> ptr(d_curr);
            final_count = thrust::reduce(thrust::cuda::par.on(str), ptr, ptr + H * W, 0, thrust::plus<int>());
            cudaStreamSynchronize(str);
        }

        d_matrix_2<int> final_board(H, W, str);
        cudaMemcpyAsync(final_board.getDevPointer(), d_curr, sizeof(int) * H * W, cudaMemcpyDeviceToDevice, str);
        cudaStreamSynchronize(str);

    if (d_alt) cudaFreeAsync(d_alt, str);
        cudaStreamSynchronize(str);
        return {std::move(final_board), final_count};
    }

    void generateGameOfLifeData(int filenum, double ratio, int seed, dataset_id info) {
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            std::cerr << "[FATAL] No CUDA device: " << cudaGetErrorString(err) << std::endl;
            exit(1);
        }
        cudaSetDevice(0);

        // 스트림 생성
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        std::string datasetPath = "../" + getDatasetId(info) + "/";

        
        if (fs::exists(datasetPath)) {
            std::cout << "[INFO] Dataset directory already exists: " << datasetPath << std::endl;
            return;
        }else {
            fs::create_directories(datasetPath);
        }
        int totalFiles = filenum;
        double aliveratio = ratio;

        std::cout << "totalFiles:" << totalFiles << " (file direction: " << datasetPath << ")" << std::endl;
        std::cout << "aliveratio:" << aliveratio << std::endl;
        std::cout << "max generation:" << MAXGEN << std::endl;
        std::cout << "pattern size:" << HEIGHT << " * " << WIDTH << std::endl;
        std::cout << "board size:" << BOARDHEIGHT << " * " << BOARDWIDTH << std::endl;

        auto startTime = std::chrono::steady_clock::now();

        // 고정 시드 기반 난수 엔진 (파일 ID 오프셋으로 각 샘플을 유일화하되 결정성 유지)
        std::mt19937_64 global_gen(static_cast<uint64_t>(seed));
        std::uniform_int_distribution<int> offset_dist(0, std::numeric_limits<int>::max());

        for (int fileId = 1; fileId <= totalFiles; ++fileId) {
            // 각 샘플에 대해 고유하지만 결정적인 시드를 생성
            // 시드 충돌 줄이기 위해 64비트 혼합
            uint64_t file_seed = static_cast<uint64_t>(seed) ^ (static_cast<uint64_t>(fileId) * 0x9E3779B97F4A7C15ULL);
            // 패턴 생성에 시드 적용
            d_matrix_2<int> pattern = generateFixedRatioPatternWithSeed(
                BOARDHEIGHT, BOARDWIDTH, HEIGHT, WIDTH, aliveratio, static_cast<int>(file_seed & 0x7fffffff), stream);

            pattern.cpyToHost(stream);
            
            // 단일 시뮬레이션으로 최종 패턴과 라벨을 동시에 얻음
            int label = simulateAndLabel(pattern, fileId, stream);

            std::ofstream fout(datasetPath + "sample" + std::to_string(fileId) + ".txt");

            int startRow = (BOARDHEIGHT - HEIGHT) / 2;
            int startCol = (BOARDWIDTH - WIDTH) / 2;
            
            // GPU 작업이 완료될 때까지 대기
            cudaStreamSynchronize(stream);

            // 초기 패턴 저장
            for (int i = startRow; i < startRow + HEIGHT; ++i) {
                for (int j = startCol; j < startCol + WIDTH; ++j) {
                    fout << pattern(i, j);
                }
                fout << '\n';
            }

            fout << label << '\n';
            fout << '\n';

            // 최종 패턴을 호스트로 복사 (비동기)
            //last_pattern.cpyToHost(stream);
            //cudaStreamSynchronize(stream);
            //// 최종 패턴 저장
            //for(int i = 0; i < BOARDHEIGHT; i++){
            //    for(int j = 0; j < BOARDWIDTH; j++){
            //        fout << last_pattern(i, j);
            //    }
            //    fout << '\n';
            //}

            fout.close();
            
            // 진행률 표시 최적화: 매 10번째마다만 업데이트
            if (fileId % 10 == 0 || fileId == totalFiles) {
                std::string prograss_name = "GOL data generating... " + std::to_string(fileId) + "/" + std::to_string(totalFiles);
                printProgressBar(fileId, totalFiles, startTime, prograss_name);
            }
        }
        
        std::cout << std::endl << "[Done] Dataset generation complete." << std::endl;

        auto totalElapsed = std::chrono::steady_clock::now() - startTime;
        int totalSec = std::chrono::duration_cast<std::chrono::seconds>(totalElapsed).count();
        std::cout << "총 실행 시간: " << totalSec << " 초" << std::endl;

        cudaStreamDestroy(stream);
    }

    std::pair<d_matrix_2<double>, d_matrix_2<double>> LoadingDataBatch(dataset_id info, cudaStream_t str) {
        std::vector<std::pair<d_matrix_2<double>, d_matrix_2<double>>> temp_dataset;
        temp_dataset.reserve(5000);  // 4000개 + 여유분

        std::string datasetPath = "../" + getDatasetId(info) + "/";

        // 파일들을 읽어서 임시 벡터에 저장
        for (const auto& entry : fs::directory_iterator(datasetPath)) {
            if (entry.path().extension() != ".txt") continue;

            std::ifstream fin(entry.path());
            if (!fin) {
                std::cerr << "파일 열기 실패: " << entry.path() << '\n';
                continue;
            }

            d_matrix_2<double> input(WIDTH*HEIGHT, 1, str);
            std::string line;
            int row = 0;
            while (row < WIDTH && std::getline(fin, line)) {
                int len = std::min(HEIGHT, static_cast<int>(line.size()));
                for (int col = 0; col < len; ++col) {
                    input(row * HEIGHT + col, 0) = line[col] - '0';
                }
                row++;
            }

            int label_index = -1;
            if (std::getline(fin, line)) label_index = std::stoi(line);

            d_matrix_2<double> label(BIT_WIDTH, 1, str);
            // 1) 모두 0으로 초기화
            label.fill(0.0, str);
            // 2) 각 비트 위치에 0/1 설정 (LSB부터)
            for (int b = 0; b < BIT_WIDTH; ++b) {
                label(b, 0) = (label_index >> b) & 1;
            }

            input.cpyToDev(str);
            label.cpyToDev(str);
            temp_dataset.emplace_back(std::move(input), std::move(label));
        }

        int N = temp_dataset.size();
        std::cout << "로드된 데이터 개수: " << N << std::endl;

        // 배치 친화적인 형태로 변환: (N, features) 형태의 행렬 2개
        d_matrix_2<double> X(N, WIDTH*HEIGHT, str);  // 입력 행렬: (샘플수, 특성수)
        d_matrix_2<double> Y(N, BIT_WIDTH, str);     // 라벨 행렬: (샘플수, 클래스수)

        // 데이터 복사
        for (int i = 0; i < N; i++) {
            temp_dataset[i].first.cpyToHost(str);
            temp_dataset[i].second.cpyToHost(str);
            cudaStreamSynchronize(str);

            // 입력 데이터 복사
            for (int j = 0; j < WIDTH*HEIGHT; j++) {
                X(i, j) = temp_dataset[i].first.getHostPointer()[j];
            }

            // 라벨 데이터 복사
            for (int j = 0; j < BIT_WIDTH; j++) {
                Y(i, j) = temp_dataset[i].second.getHostPointer()[j];
            }
        }

        // GPU로 전송
        X.cpyToDev(str);
        Y.cpyToDev(str);
        cudaStreamSynchronize(str);

        return {std::move(X), std::move(Y)};
    }

} // namespace GOL_2
