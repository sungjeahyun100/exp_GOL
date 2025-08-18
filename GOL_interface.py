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

lib.predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
lib.predict.restype = None

lib.freeModel.argtypes = [ctypes.c_void_p]
lib.freeModel.restype = None

# 게임 설정
WINDOW_SIZE = 600
GRID_SIZE = 10
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
PADDING = 20
SIDEBAR_WIDTH = 200

# 색상 정의
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
GREEN = (0, 255, 100)
DARK_GREEN = (0, 180, 70)
DARK_GRAY = (40, 40, 40)
BLUE = (50, 150, 255)
YELLOW = (255, 255, 0)

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
        
        # 그리드 초기화
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        
        # 모델 로드
        try:
            self.model = lib.initModel(MODEL_PATH)
            print("AI 모델 로드 성공!")
            self.predictions = np.zeros(8, dtype=np.float32)
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            self.model = None
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                
                # 그리드 영역 내 클릭 처리
                if x < WINDOW_SIZE and y < WINDOW_SIZE:
                    grid_x = x // CELL_SIZE
                    grid_y = y // CELL_SIZE
                    
                    # 셀 상태 토글
                    self.grid[grid_y, grid_x] = 1 - self.grid[grid_y, grid_x]
                    
                    # AI 예측 업데이트
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
                print(f"예측 오류: {e}")
    
    def get_binary_prediction(self):
        # 예측값을 0, 1로 변환 (임계값: 0.5)
        binary = [1 if pred > 0.5 else 0 for pred in self.predictions]
        return binary
    
    def binary_to_decimal(self, binary):
        # 8비트 이진수를 십진수로 변환
        decimal = 0
        for bit in binary:
            decimal = (decimal << 1) | bit
        return decimal
    
    def draw_grid(self):
        # 배경 그리기
        self.screen.fill(BLACK)
        
        # 그리드 그리기
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                # 셀 색상 (활성/비활성)
                if self.grid[y, x] > 0:
                    pygame.draw.rect(self.screen, GREEN, rect)
                else:
                    pygame.draw.rect(self.screen, DARK_GRAY, rect)
                
                # 테두리
                pygame.draw.rect(self.screen, GRAY, rect, 1)
        
        # 사이드바 배경
        sidebar_rect = pygame.Rect(WINDOW_SIZE, 0, SIDEBAR_WIDTH, WINDOW_SIZE)
        pygame.draw.rect(self.screen, DARK_GRAY, sidebar_rect)
        
        # 예측 결과 표시
        if self.model:
            title = self.font.render("AI predict:", True, WHITE)
            self.screen.blit(title, (WINDOW_SIZE + 10, 20))
            
            # 각 비트 값 표시
            for i, pred in enumerate(self.predictions):
                value = f"Bit {i}: {pred:.4f} -> {'1' if pred > 0.5 else '0'}"
                color = GREEN if pred > 0.5 else WHITE
                text = self.font.render(value, True, color)
                self.screen.blit(text, (WINDOW_SIZE + 10, 50 + i*25))
            
            # 이진수 표시
            binary = self.get_binary_prediction()
            binary_str = ''.join(map(str, binary))
            binary_text = self.font.render(f"binary: {binary_str}", True, YELLOW)
            self.screen.blit(binary_text, (WINDOW_SIZE + 10, 260))
            
            # 십진수 변환 표시
            decimal = self.binary_to_decimal(binary)
            decimal_text = self.large_font.render(f"decimal: {decimal}", True, YELLOW)
            self.screen.blit(decimal_text, (WINDOW_SIZE + 10, 290))
    
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.draw_grid()
            pygame.display.flip()
            self.clock.tick(60)
        
        # 종료 시 모델 메모리 해제
        if self.model:
            lib.freeModel(self.model)
        
        pygame.quit()

# 게임 실행
if __name__ == "__main__":
    game = GameOfLifeAI()
    game.run()