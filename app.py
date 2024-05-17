import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QRegExp  # Dodatkowy import modułu Qt
from PyQt5.QtGui import QFont, QIntValidator, QRegExpValidator # Dodatkowy import klasy QFont
from utilities import *
from SudokuVizualizator import SudokuVizualizator
from SudokuDetector import SudokuDetector, WorkerDetector
from sudoku import draw_solving
from sudoku_solver import solvers_map
import cv2
import os


class SudokuWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sudoku")
        # self.setGeometry(100, 100, 800, 600)
        # self.setGeometry(100, 100, 300, 300)

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self.solver_tab = QWidget()
        self.detector_tab = QWidget()

        self.tab_widget.addTab(self.solver_tab, "Solver")
        self.solver_tab.tabIndex = 0
        self.tab_widget.addTab(self.detector_tab, "Detector")
        self.detector_tab.tabIndex = 1


        self.init_solver_tab()
        self.init_detector_tab()


    def init_solver_tab(self):
        solver_tab_layout = QGridLayout()
        self.solver_tab.setLayout(solver_tab_layout)
        
        # self.solver_tab.setLayout(self.sudoku_board_layout)
        sudoku_board_layout = self.create_sudoku_board()

        solver_tab_layout.addLayout(sudoku_board_layout,0,0)

        options_layout = self.create_solver_tab_options()

        solver_tab_layout.addLayout(options_layout, 0,1) 

        sudoku_board_functions = self.create_sudoku_board_functions()

        solver_tab_layout.addLayout(sudoku_board_functions,1,0)

    def create_sudoku_board_functions(self):
        layout = QVBoxLayout()

        clear_button = QPushButton()
        clear_button.setText("Clear")
        clear_button.clicked.connect(self._clear_button_action)
        layout.addWidget(clear_button)

        set_zeros_button = QPushButton()
        set_zeros_button.setText("Set Zeros")
        set_zeros_button.clicked.connect(self._set_zeros_button_action)
        layout.addWidget(set_zeros_button)

        set_empty_button = QPushButton()
        set_empty_button.setText("Set Empty")
        set_empty_button.clicked.connect(self._set_empty_button_action)
        layout.addWidget(set_empty_button)

        return layout

    def create_solver_tab_options(self):
        options_layout = QVBoxLayout()

        self.comboBox_solver = QComboBox()
        for solver in solvers_map.keys():
            self.comboBox_solver.addItem(solver)
        
        label = QLabel("Solving Method")

        temp_layout = QHBoxLayout()
        temp_layout.addWidget(label)
        temp_layout.addItem(QSpacerItem(10,1))
        temp_layout.addWidget(self.comboBox_solver)

        options_layout.addItem(temp_layout)

        solve_button = QPushButton()
        solve_button.setText("Solve")
        solve_button.clicked.connect(self._solve_button_action)
        self.solve_button = solve_button

        options_layout.addWidget(solve_button)

        return options_layout
    
    def get_sudoku_grid(self):
        sudoku_grid = []
        for row in range(9):
            sudoku_grid.append([])
            for col in range(9):
                cell = self.sudoku_board_cells[row][col]
                if cell.text() != "":
                    sudoku_grid[-1].append(int(cell.text()))
                else:
                    sudoku_grid[-1].append(0)
        return sudoku_grid
    def _solve_button_action(self):
        self.solve_button.setEnabled(False)
        sudoku_grid = self.get_sudoku_grid()

        sv = SudokuVizualizator(sudoku_grid)
        sv.run()
        self.solve_button.setEnabled(True)

    def _clear_button_action(self):
        for row in range(9):
            for col in range(9):
                self.sudoku_board_cells[row][col].setText("")
    
    def _set_zeros_button_action(self):
        for row in range(9):
            for col in range(9):
                if self.sudoku_board_cells[row][col].text() == "":
                    self.sudoku_board_cells[row][col].setText("0")

    def _set_empty_button_action(self):
        for row in range(9):
            for col in range(9):
                if self.sudoku_board_cells[row][col].text() == "0":
                    self.sudoku_board_cells[row][col].setText("")

    def _open_image_button_action(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if os.path.exists(file_path):
            self.sudoku_detector.load_image(file_path)
            self.sudoku_detector.detect_board()

    def _read_sudoku_button_action(self):
        self.read_sudoku_button.setEnabled(False)
        if self.sudoku_detector.read_sudoku():
            self._show_progress_dialog()
        else:
            self.read_sudoku_button.setEnabled(True)

        # t = ThreadWithReturnValue(target=self.sudoku_detector.read_sudoku())
        # t.start()
    
    def _update_reading_progress(self, value):
        self.readingProgressDialog.setValue(int(value*100))
    
    def _show_progress_dialog(self):
        self.readingProgressDialog = QProgressDialog("Reading board...", "Cancel", 0, 100, self)
        self.readingProgressDialog.setWindowTitle("Detector")
        self.readingProgressDialog.setWindowModality(Qt.WindowModal)
        self.readingProgressDialog.setMinimumDuration(0)
        self.readingProgressDialog.setValue(0)
        self.readingProgressDialog.canceled.connect(self._cancel_reading)

    def _cancel_reading(self):
        self.sudoku_detector.worker.terminate()
        self.readingProgressDialog.close()
        self.read_sudoku_button.setEnabled(True)

    def _set_readed_sudoku_board(self, sudoku):
        # sudoku = self.sudoku_detector.read_sudoku()
        
        if not sudoku is None:
            for row in range(9):
                for col in range(9):
                    self.sudoku_board_cells[row][col].setText(str(sudoku[row][col]))

        # to clear zeros
        self._set_empty_button_action()

        self.readingProgressDialog.close()
        self.read_sudoku_button.setEnabled(True)
        self.tab_widget.setCurrentIndex(self.solver_tab.tabIndex)

    def init_detector_tab(self):
        detector_tab_layout = QGridLayout()
        self.detector_tab.setLayout(detector_tab_layout)
        
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        detector_tab_layout.addWidget(self.canvas,0,0)

        self.sudoku_detector = SudokuDetector(self.canvas)

        options_layout = self.create_detector_tab_options()
        detector_tab_layout.addLayout(options_layout, 0,1) 

        self.sudoku_detector.worker.finished_event.connect(self._set_readed_sudoku_board)
        self.sudoku_detector.worker.progress_event.connect(self._update_reading_progress)

    def create_detector_tab_options(self):
        options_layout = QVBoxLayout()

        open_image_button = QPushButton("Open Image")
        open_image_button.clicked.connect(self._open_image_button_action)
        options_layout.addWidget(open_image_button)

        self.read_sudoku_button = QPushButton("Read Sudoku")
        
        self.read_sudoku_button.clicked.connect(self._read_sudoku_button_action)
        options_layout.addWidget(self.read_sudoku_button)


        return options_layout
        
    def create_sudoku_board(self):
        sudoku_board_layout = QGridLayout()
        sudoku_board_layout.setSpacing(0)
        regex = QRegExp("[0-9]")
        reg_validator = QRegExpValidator(regex)
        # int_validator = QIntValidator(1, 9)
        self.sudoku_board_cells = []
        for i in range(9):
            self.sudoku_board_cells.append([])
            for j in range(9):              
                cell = QLineEdit()
                cell.setFixedSize(30, 30)
                cell.setAlignment(Qt.AlignCenter)  # Centralne wyświetlanie tekstu
                cell.setValidator(reg_validator)
                cell.setFont(QFont("Arial", 14))  # Ustawienie czcionki
                cell.installEventFilter(self)
                sudoku_board_layout.addWidget(cell, i, j)
                self.sudoku_board_cells[-1].append(cell)

        for row in range(9):
            for col in range(9):
                next_ = row*9+col+1
                next_col = next_%9
                next_row = next_//9 if next_//9 < 9 else 0
                self.sudoku_board_cells[row][col].textChanged.connect(lambda text, col = next_col, row = next_row: self._go_to_cell(col,row) if text != "" else None)
        return sudoku_board_layout
    
    def _go_to_cell(self, col, row):
        self.sudoku_board_cells[row][col].setFocus()
        self.sudoku_board_cells[row][col].selectAll()
    
    def eventFilter(self, source, event):
        # moving on board using arrow keys
        if event.type() == event.KeyPress and isinstance(source, QLineEdit):
            current_index = self.get_cell_index(source)
            if current_index is None:
                return False
            
            if event.key() == Qt.Key_Up:
                new_index = (current_index[0], current_index[1]-1 if current_index[1] > 0 else 8)
                self._go_to_cell(new_index[0], new_index[1])
                return True
            elif event.key() == Qt.Key_Down:
                new_index = (current_index[0], current_index[1]+1 if current_index[1] < 8 else 0)
                self._go_to_cell(new_index[0], new_index[1])
                return True
            elif event.key() == Qt.Key_Left:
                new_index = (current_index[0]-1 if current_index[0] > 0 else 8, current_index[1])
                self._go_to_cell(new_index[0], new_index[1])
                return True
            elif event.key() == Qt.Key_Right:
                new_index = (current_index[0]+1 if current_index[0] < 8 else 0, current_index[1])
                self._go_to_cell(new_index[0], new_index[1])
                return True

        return super().eventFilter(source, event)

    
    def get_cell_index(self, cell):
        for row in range(9):
            if cell in self.sudoku_board_cells[row]:
                return (self.sudoku_board_cells[row].index(cell), row)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    sudoku_window = SudokuWindow()
    sudoku_window.show()
    sys.exit(app.exec_())
