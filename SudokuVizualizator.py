import pygame
import sudoku_solver


SCREEN_WIDTH = 542
SCREEN_HEIGHT = 600

GRID_SIZE = 9
CELL_SIZE = 60

GRID_COLOR = (0, 0, 0)
TEXT_COLOR = (0, 0, 0)
ENTER_TEXT_COLOR = (100, 100, 100)
BG_COLOR = (255, 255, 255)
NEXT_MOVE_EVENT =  pygame.USEREVENT + 1
NEXT_MOVE_EVENT_DURATION = 300 #ms

class SudokuWindow():
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Plansza do Sudoku")
    
    def draw_board(self):
        self.screen.fill(BG_COLOR)
        self.draw_grid()
    
    def draw_grid(self):
        for i in range(GRID_SIZE + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (0, i * CELL_SIZE), (9*CELL_SIZE, i * CELL_SIZE), 2 if i % 3 == 0 else 1)


        for j in range(GRID_SIZE + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (j * CELL_SIZE, 0), (j * CELL_SIZE, SCREEN_HEIGHT - CELL_SIZE), 2 if j % 3 == 0 else 1)
    
    def draw_numbers(self, grid, color = TEXT_COLOR):
        font = pygame.font.Font(None, 40)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if grid[i][j] != 0:
                    text = font.render(str(grid[i][j]), True, color)
                    text_rect = text.get_rect(center=(j * CELL_SIZE + CELL_SIZE / 2, i * CELL_SIZE + CELL_SIZE / 2))
                    self.screen.blit(text, text_rect)

    def draw_posibles(self, grid, color = TEXT_COLOR):
        font = pygame.font.Font(None, CELL_SIZE//3)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if len(grid[i][j]) > 0:
                    for num in grid[i][j]:
                        x, y = (num-1)%3-1, (num-1)//3-1
                        x, y = x/3, y/3
                        text = font.render(str(num), True, color)
                        text_rect = text.get_rect(center=(j * CELL_SIZE + CELL_SIZE / 2 + x*CELL_SIZE, i * CELL_SIZE + CELL_SIZE / 2 + y*CELL_SIZE))
                        self.screen.blit(text, text_rect)
    def draw(self,original_grid, actual_grid, posibles_grid):
        self.draw_board()
        self.draw_numbers(actual_grid, ENTER_TEXT_COLOR)
        self.draw_numbers(original_grid, TEXT_COLOR)
        self.draw_posibles(posibles_grid)
    

        

class SudokuVizualizator():
    def __init__(self, init_grid, solver = "Traditional solver") -> None:
        pygame.init()
        self.sudokuWindow = SudokuWindow()
        self.clock = pygame.time.Clock()

        self.originalGrid = init_grid

        self.solver = sudoku_solver.solvers_map[solver](init_grid)
        pygame.time.set_timer(pygame.event.Event(NEXT_MOVE_EVENT), NEXT_MOVE_EVENT_DURATION)   
    
    def run(self):
        flag = True
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if not self.solver.insert_single():
                            if not self.solver.insert_last_remaining():
                                if not self.solver.correct_obvious_groups():
                                    if not self.solver.correct_line_in_squere():
                                        flag = False
                if event.type == NEXT_MOVE_EVENT:
                    if flag:
                        if not self.solver.insert_single():
                            if not self.solver.insert_last_remaining():
                                if not self.solver.correct_obvious_groups():
                                    if not self.solver.correct_line_in_squere():
                                        flag = False

            self.sudokuWindow.draw(self.originalGrid,self.solver.sudoku_grid,self.solver.posible_grid)            
            pygame.display.flip()
            self.clock.tick(15)
        pygame.quit()

if __name__ == "__main__":
    sw = SudokuWindow()
    sudoku_grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 9, 0, 0, 6, 0],
        [0, 0, 4, 0, 0, 5, 0, 0, 2],
        [0, 0, 2, 0, 0, 8, 0, 0, 1],
        [7, 3, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 9, 3, 0, 0],
        [0, 0, 0, 0, 7, 0, 0, 0, 3],
        [0, 0, 9, 0, 4, 0, 0, 0, 5],
        [0, 4, 7, 0, 6, 0, 2, 0, 0]
    ]

    SudokuVizualizator(sudoku_grid).run()

    