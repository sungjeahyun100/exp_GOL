import pygame
import numpy as np
import ctypes
import os

# 공유 라이브러리 경로
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "mercury_python.so")

# 라이브러리 로드
lib = ctypes.CDLL(lib_path)

# 함수 프로토타입 설정
lib.initModel.argtypes = [ctypes.c_char_p]
lib.initModel.restype = ctypes.c_void_p

lib.createNewModel.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int]
lib.createNewModel.restype = ctypes.c_void_p

lib.trainWithUserData.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
lib.trainWithUserData.restype = None

lib.saveModelToFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.saveModelToFile.restype = ctypes.c_bool

lib.getTrainingInfo.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
lib.getTrainingInfo.restype = None

lib.predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
lib.predict.restype = None

lib.freeModel.argtypes = [ctypes.c_void_p]
lib.freeModel.restype = None

# 게임 설정
WINDOW_SIZE = 600
GRID_SIZE = 10
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
PADDING = 20
SIDEBAR_WIDTH = 300  # 사이드바 폭 증가 (학습 UI 위해)

# 색상 정의
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
GREEN = (0, 255, 100)
DARK_GREEN = (0, 180, 70)
DARK_GRAY = (40, 40, 40)
BLUE = (50, 150, 255)
YELLOW = (255, 255, 0)
RED = (255, 100, 100)
PURPLE = (200, 100, 255)

# 모델 경로
MODEL_PATH = b"../model_save/mercury_custom_conv3-LReLU-He_fc5-Tanh-Xavier_BCEWithLogits_Adam-1000-50-0.000001-2025-08-17_193828"

# 게임 클래스
class GameOfLifeAI:
    def __init__(self):
        # 파이게임 초기화
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE + SIDEBAR_WIDTH, WINDOW_SIZE))
        pygame.display.set_caption("Game of Life with AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 16)
        self.large_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 12)
        
        # 그리드 초기화
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        
        # 모델 상태
        self.model = None
        self.predictions = np.zeros(8, dtype=np.float32)
        self.is_training = False
        self.training_epochs = 100
        self.training_progress = 0
        self.loss_history = []
        
        # 사용자 학습 데이터 수집
        self.user_training_data = []  # [(input_pattern, target_label), ...]
        self.collecting_data = False
        
        # UI 상태
        self.mode = "predict"  # "predict", "train", "collect"
        
        # 모델 로드 시도
        self.try_load_model()
    
    def try_load_model(self):
        """Try to load existing model"""
        try:
            self.model = lib.initModel(MODEL_PATH)
            print("AI model loaded successfully!")
            self.mode = "predict"
        except Exception as e:
            print(f"Model loading failed: {e}")
            self.model = None
            self.mode = "train"
    
    def create_new_model(self, batch_size=50, learning_rate=1e-6, epochs=100):
        """Create new model"""
        try:
            if self.model:
                lib.freeModel(self.model)
            
            self.model = lib.createNewModel(batch_size, learning_rate, epochs)
            print(f"New model created (batch: {batch_size}, lr: {learning_rate}, epochs: {epochs})")
            self.mode = "train"
            return True
        except Exception as e:
            print(f"Model creation failed: {e}")
            return False
    
    def start_training(self, epochs=100):
        """Start model training"""
        if not self.model:
            print("No model available. Please create a new model first.")
            return
        
        try:
            print(f"Training started ({epochs} epochs)...")
            self.is_training = True
            self.training_epochs = epochs
            self.loss_history = np.zeros(epochs, dtype=np.float32)
            
            # Run training in separate thread to prevent UI blocking
            import threading
            training_thread = threading.Thread(
                target=self._train_worker, 
                args=(epochs,)
            )
            training_thread.daemon = True
            training_thread.start()
            
        except Exception as e:
            print(f"Training start failed: {e}")
            self.is_training = False
    
    def start_user_data_training(self, epochs=100):
        """Start training with user data"""
        if not self.model:
            print("No model available. Please create a new model first.")
            return
        
        if len(self.user_training_data) == 0:
            print("No training data available. Please collect data first.")
            return
        
        try:
            print(f"User data training started ({len(self.user_training_data)} samples, {epochs} epochs)...")
            self.is_training = True
            self.training_epochs = epochs
            self.loss_history = np.zeros(epochs, dtype=np.float32)
            
            # Run training in separate thread
            import threading
            training_thread = threading.Thread(
                target=self._train_user_data_worker, 
                args=(epochs,)
            )
            training_thread.daemon = True
            training_thread.start()
            
        except Exception as e:
            print(f"User data training start failed: {e}")
            self.is_training = False
    
    def add_training_sample(self):
        """Add current grid to training samples"""
        if np.sum(self.grid) == 0:
            print("Empty grid cannot be added as training data.")
            return
        
        # Current grid pattern
        input_pattern = self.grid.flatten()
        
        # Calculate actual GOL result
        try:
            real_result = np.zeros(8, dtype=np.float32)
            lib.getRealGOLnumber(
                input_pattern.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                real_result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
            
            # Add to training data
            self.user_training_data.append((input_pattern.copy(), real_result.copy()))
            print(f"Training sample added (total: {len(self.user_training_data)})")
            
        except Exception as e:
            print(f"Failed to add training sample: {e}")
    
    def clear_training_data(self):
        """Clear collected training data"""
        self.user_training_data.clear()
        print("Training data cleared.")
    
    def toggle_data_collection(self):
        """Toggle data collection mode"""
        if self.mode == "collect":
            self.mode = "predict" if self.model else "train"
            self.collecting_data = False
            print("Data collection mode ended")
        else:
            self.mode = "collect"
            self.collecting_data = True
            print("Data collection mode started")
    
    def _train_user_data_worker(self, epochs):
        """Worker function to perform actual training with user data"""
        try:
            # Prepare data
            num_samples = len(self.user_training_data)
            input_data = np.zeros((num_samples * 100,), dtype=np.float32)
            target_data = np.zeros((num_samples * 8,), dtype=np.float32)
            
            for i, (input_pattern, target_label) in enumerate(self.user_training_data):
                input_data[i*100:(i+1)*100] = input_pattern
                target_data[i*8:(i+1)*8] = target_label
            
            # Execute training
            lib.trainWithUserData(
                self.model,
                input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                target_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                num_samples,
                epochs,
                self.loss_history.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
            
            print("User data training completed!")
            self.is_training = False
            self.mode = "predict"
            
        except Exception as e:
            print(f"Error during user data training: {e}")
            self.is_training = False
    
    def _train_worker(self, epochs):
        """Worker function to perform actual training"""
        try:
            lib.trainModel(
                self.model,
                epochs,
                self.loss_history.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
            print("Training completed!")
            self.is_training = False
            self.mode = "predict"
        except Exception as e:
            print(f"Error during training: {e}")
            self.is_training = False
    
    def save_model(self, filepath=None):
        """Save model"""
        if not self.model:
            print("No model to save.")
            return False
        
        if filepath is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filepath = f"../model_save/mercury_trained_{timestamp}"
        
        try:
            success = lib.saveModelToFile(self.model, filepath.encode('utf-8'))
            if success:
                print(f"Model saved: {filepath}")
                return True
            else:
                print("Model save failed")
                return False
        except Exception as e:
            print(f"Model save error: {e}")
            return False
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                # 키보드 단축키
                if event.key == pygame.K_n:  # N: 새 모델 생성
                    self.create_new_model()
                elif event.key == pygame.K_t:  # T: 기존 데이터셋으로 학습 시작
                    self.start_training()
                elif event.key == pygame.K_u:  # U: 사용자 데이터로 학습 시작
                    self.start_user_data_training()
                elif event.key == pygame.K_d:  # D: 데이터 수집 모드 토글
                    self.toggle_data_collection()
                elif event.key == pygame.K_a:  # A: 현재 패턴을 학습 데이터로 추가
                    if self.mode == "collect":
                        self.add_training_sample()
                elif event.key == pygame.K_x:  # X: 수집된 학습 데이터 삭제
                    self.clear_training_data()
                elif event.key == pygame.K_s:  # S: 모델 저장
                    self.save_model()
                elif event.key == pygame.K_c:  # C: 그리드 클리어
                    self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
                elif event.key == pygame.K_r:  # R: 랜덤 그리드
                    self.grid = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE)).astype(np.float32)
                    if self.model and self.mode == "predict":
                        self.update_prediction()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                
                # 그리드 영역 내 클릭 처리
                if x < WINDOW_SIZE and y < WINDOW_SIZE:
                    grid_x = x // CELL_SIZE
                    grid_y = y // CELL_SIZE
                    
                    # 셀 상태 토글
                    self.grid[grid_y, grid_x] = 1 - self.grid[grid_y, grid_x]
                    
                    # 예측 모드에서만 AI 예측 업데이트
                    if self.model and self.mode == "predict":
                        self.update_prediction()
        
        return True
    
    def update_prediction(self):
        if self.model:
            try:
                # 예측 함수 호출
                lib.predict(
                    self.model,
                    self.grid.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    self.predictions.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                )
            except Exception as e:
                print(f"Prediction error: {e}")
    
    def get_binary_prediction(self):
        # Convert prediction values to 0, 1 (threshold: 0.5)
        binary = [1 if pred > 0.5 else 0 for pred in self.predictions]
        binary.reverse()
        return binary
    
    def binary_to_decimal(self, binary):
        # Convert 8-bit binary to decimal
        decimal = 0
        for bit in binary:
            decimal = (decimal << 1) | bit
        return decimal
    
    def draw_grid(self):
        # Draw background
        self.screen.fill(BLACK)
        
        # Draw grid
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                # Cell color (active/inactive)
                if self.grid[y, x] > 0:
                    pygame.draw.rect(self.screen, GREEN, rect)
                else:
                    pygame.draw.rect(self.screen, DARK_GRAY, rect)
                
                # Border
                pygame.draw.rect(self.screen, GRAY, rect, 1)
        
        # Sidebar background
        sidebar_rect = pygame.Rect(WINDOW_SIZE, 0, SIDEBAR_WIDTH, WINDOW_SIZE)
        pygame.draw.rect(self.screen, DARK_GRAY, sidebar_rect)
        
        # Display UI
        self.draw_ui()
    
    def draw_ui(self):
        y_offset = 20
        
        # Display current mode
        mode_text = f"Mode: {'Predict' if self.mode == 'predict' else ('Train' if self.mode == 'train' else 'Collect Data')}"
        mode_color = GREEN if self.mode == "predict" else (YELLOW if self.mode == "train" else PURPLE)
        text = self.font.render(mode_text, True, mode_color)
        self.screen.blit(text, (WINDOW_SIZE + 10, y_offset))
        y_offset += 30
        
        # Keyboard shortcuts
        shortcuts = [
            "Shortcuts:",
            "N: Create new model",
            "T: Train with dataset",
            "U: Train with user data",
            "D: Data collection mode",
            "A: Add current pattern",
            "X: Clear training data",
            "S: Save model",
            "C: Clear grid",
            "R: Random grid"
        ]
        
        for shortcut in shortcuts:
            color = WHITE if shortcut == "Shortcuts:" else GRAY
            font_to_use = self.small_font if shortcut != "Shortcuts:" else self.font
            text = font_to_use.render(shortcut, True, color)
            self.screen.blit(text, (WINDOW_SIZE + 10, y_offset))
            y_offset += 16 if shortcut != "Shortcuts:" else 20
        
        y_offset += 10
        
        # Display collected data count
        data_count_text = f"Collected data: {len(self.user_training_data)} samples"
        data_color = GREEN if len(self.user_training_data) > 0 else GRAY
        text = self.font.render(data_count_text, True, data_color)
        self.screen.blit(text, (WINDOW_SIZE + 10, y_offset))
        y_offset += 25
        
        # Display training status
        if self.is_training:
            train_text = self.font.render("Training...", True, YELLOW)
            self.screen.blit(train_text, (WINDOW_SIZE + 10, y_offset))
            y_offset += 25
        
        # Mode-specific UI display
        if self.model and self.mode == "predict":
            self.draw_prediction_results(y_offset)
        elif self.mode == "train":
            self.draw_training_info(y_offset)
        elif self.mode == "collect":
            self.draw_collection_info(y_offset)
    
    def draw_prediction_results(self, y_start):
        title = self.font.render("AI Prediction:", True, WHITE)
        self.screen.blit(title, (WINDOW_SIZE + 10, y_start))
        y_start += 25
        
        # Display each bit value
        for i, pred in enumerate(self.predictions):
            value = f"Bit {i}: {pred:.4f} -> {'1' if pred > 0.5 else '0'}"
            color = GREEN if pred > 0.5 else WHITE
            text = self.small_font.render(value, True, color)
            self.screen.blit(text, (WINDOW_SIZE + 10, y_start + i*18))
        
        y_start += 8 * 18 + 10
        
        # Display binary
        binary = self.get_binary_prediction()
        binary_str = ''.join(map(str, binary))
        binary_text = self.font.render(f"Binary: {binary_str}", True, YELLOW)
        self.screen.blit(binary_text, (WINDOW_SIZE + 10, y_start))
        
        # Display decimal conversion
        decimal = self.binary_to_decimal(binary)
        decimal_text = self.large_font.render(f"Decimal: {decimal}", True, YELLOW)
        self.screen.blit(decimal_text, (WINDOW_SIZE + 10, y_start + 25))
    
    def draw_training_info(self, y_start):
        if not self.model:
            no_model_text = self.font.render("No model", True, RED)
            self.screen.blit(no_model_text, (WINDOW_SIZE + 10, y_start))
            
            create_text = self.font.render("Press N to create new model", True, WHITE)
            self.screen.blit(create_text, (WINDOW_SIZE + 10, y_start + 25))
        else:
            train_info_text = self.font.render("Ready to train", True, GREEN)
            self.screen.blit(train_info_text, (WINDOW_SIZE + 10, y_start))
            
            start_text = self.font.render("T: Train with dataset", True, WHITE)
            self.screen.blit(start_text, (WINDOW_SIZE + 10, y_start + 25))
            
            user_text = self.font.render("U: Train with user data", True, WHITE)
            self.screen.blit(user_text, (WINDOW_SIZE + 10, y_start + 45))
            
            # Loss graph (simple text form)
            if len(self.loss_history) > 0:
                recent_losses = self.loss_history[-10:]  # Recent 10 loss values
                loss_text = f"Recent loss: {recent_losses[-1]:.6f}"
                loss_display = self.small_font.render(loss_text, True, PURPLE)
                self.screen.blit(loss_display, (WINDOW_SIZE + 10, y_start + 70))
    
    def draw_collection_info(self, y_start):
        collect_title = self.font.render("Data Collection Mode", True, PURPLE)
        self.screen.blit(collect_title, (WINDOW_SIZE + 10, y_start))
        
        instructions = [
            "1. Draw pattern with mouse",
            "2. Press A to add pattern",
            "3. Collect multiple patterns",
            "4. Press U to start training",
            "",
            "D: Exit collection mode",
            "X: Clear data"
        ]
        
        for i, instruction in enumerate(instructions):
            color = WHITE if instruction else GRAY
            if instruction:
                text = self.small_font.render(instruction, True, color)
                self.screen.blit(text, (WINDOW_SIZE + 10, y_start + 25 + i*18))
    
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.draw_grid()
            pygame.display.flip()
            self.clock.tick(60)
        
        # Terminate model memory when exiting
        if self.model:
            lib.freeModel(self.model)
        
        pygame.quit()

# Run game
if __name__ == "__main__":
    game = GameOfLifeAI()
    game.run()