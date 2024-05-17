import cv2
from sudokuDetectorFunctions import get_corners, pytesseract, get_divided_image, extract_from_cell
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from matplotlib.widgets import PolygonSelector
from PyQt5.QtWidgets import QProgressDialog, QMessageBox, QErrorMessage

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class WorkerDetector(QThread):
    finished_event = pyqtSignal(list)
    progress_event = pyqtSignal(float)

    def run(self):        
        sudoku_board = []
        for i, cell in enumerate(get_divided_image(self.original, self.corners)):
            if i%9 == 0:
                sudoku_board.append([])
            # if self.progress.wasCanceled():
            #     return

            self.progress_event.emit((i/81))

            num = extract_from_cell(cell)
            if(num == ""):
                num = 0
            else:
                num = int(num)
            
            sudoku_board[-1].append(num)

        self.finished_event.emit(sudoku_board)
    
class SudokuDetector():
    
    def __init__(self, canvas) -> None:
        self.canvas = canvas
        self.selector = PolygonSelector(canvas.axes, lambda *args: None)
        self.worker = WorkerDetector()
        self.image = None
        
        
    def draw(self):
        self.canvas.axes.imshow(self.image)
        self.canvas.draw()
    
    def set_image(self, image):
        self.image = image
        self.draw()

    def load_image(self, file_path):
        self.selector.clear()
        self.image = cv2.imread(file_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.draw()
    
    def set_tesseract_path(self, path):
        pytesseract.pytesseract.tesseract_cmd = path
    
    def detect_board(self):
        if self.image is None:
            return
        vert = get_corners(self.image)
        vert = vert

        self.selector.verts = vert
    
    def read_sudoku(self):
        if not self.image is None:
            self.worker.original = self.image
            self.worker.corners = self.selector.verts
            self.worker.start()
            return True
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("You have to load image first")
            # msg.setInformativeText('You have to load image first.')
            msg.setWindowTitle("Error")
            msg.exec_()
        return False
        # return worker.finished_event
            

    
    