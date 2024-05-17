import cv2
import numpy as np
from skimage.segmentation import clear_border
import pytesseract
import re
from collections import Counter

def cross_product(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def is_convex_hull(points):
    p1, p2, p3, p4 = points

    cp1 = cross_product(p1, p2, p3)
    cp2 = cross_product(p2, p3, p4)
    cp3 = cross_product(p3, p4, p1)
    cp4 = cross_product(p4, p1, p2)

    return (cp1 > 0 and cp2 > 0 and cp3 > 0 and cp4 > 0) or (cp1 < 0 and cp2 < 0 and cp3 < 0 and cp4 < 0)


def euclidian_distance(point1, point2):
    # Calcuates the euclidian distance between the point1 and point2
    #used to calculate the length of the four sides of the square 
    distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return distance


def order_corner_points(corners):
    # The points obtained from contours may not be in order because of the skewness  of the image, or
    # because of the camera angle. This function returns a list of corners in the right order 
    # sort_corners = [(corner[0][0], corner[0][1]) for corner in corners]
    if not is_convex_hull(corners) : 
        raise Exception("Corners must create convex hull")
    sort_corners = corners
    sort_corners = [list(ele) for ele in sort_corners]
    x, y = [], []

    for i in range(len(sort_corners[:])):
        x.append(sort_corners[i][0])
        y.append(sort_corners[i][1])

    centroid = [sum(x) / len(x), sum(y) / len(y)]

    for _, item in enumerate(sort_corners):
        if item[0] < centroid[0]:
            if item[1] < centroid[1]:
                top_left = item
            else:
                bottom_left = item
        elif item[0] > centroid[0]:
            if item[1] < centroid[1]:
                top_right = item
            else:
                bottom_right = item

    ordered_corners = [top_left, top_right, bottom_right, bottom_left]

    return np.array(ordered_corners, dtype="float32")
def image_preprocessing(image, corners):
    # This function undertakes all the preprocessing of the image and return  
    ordered_corners = order_corner_points(corners)
    # print("ordered corners: ", ordered_corners)
    top_left, top_right, bottom_right, bottom_left = ordered_corners

    # Determine the widths and heights  ( Top and bottom ) of the image and find the max of them for transform 

    width1 = euclidian_distance(bottom_right, bottom_left)
    width2 = euclidian_distance(top_right, top_left)

    height1 = euclidian_distance(top_right, bottom_right)
    height2 = euclidian_distance(top_left, bottom_right)

    width = max(int(width1), int(width2))
    height = max(int(height1), int(height2))

    # To find the matrix for warp perspective function we need dimensions and matrix parameters
    dimensions = np.array([[0, 0], [width, 0], [width, width],
                           [0, width]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    transformed_image = cv2.warpPerspective(image, matrix, (width, width))

    #Now, chances are, you may want to return your image into a specific size. If not, you may ignore the following line
    transformed_image = cv2.resize(transformed_image, (252, 252), interpolation=cv2.INTER_AREA)

    return transformed_image

    # main function

def get_square_box_from_image(image, corners = None):
    # This function returns the top-down view of the puzzle in grayscale.
    # 
    if corners is None:
        corners = get_corners()


    puzzle_image = image_preprocessing(image, corners)

    return puzzle_image

def get_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    adaptive_threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    corners = cv2.findContours(adaptive_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners = corners[0] if len(corners) == 2 else corners[1]
    corners = sorted(corners, key=cv2.contourArea, reverse=True)
    for corner in corners:
        length = cv2.arcLength(corner, True)
        approx = cv2.approxPolyDP(corner, 0.015 * length, True)
        break
    approx = [tuple(v[0]) for v in approx]
    return approx

# numbers detection

def remove_noise_and_keep_large_contours(binary_image, min_contour_area_percent):
    # Konwertowanie obrazu binarnego na obiekt Mat w OpenCV
    image_mat = np.array(binary_image, dtype=np.uint8)

    # Znajdź kontury na obrazie
    contours, _ = cv2.findContours(image_mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Oblicz całkowite pole powierzchni obrazu
    total_area = binary_image.shape[0] * binary_image.shape[1]

    # Oblicz minimalne pole powierzchni konturu na podstawie procentu całkowitego obszaru obrazu
    min_contour_area = total_area * min_contour_area_percent

    # Inicjalizuj obraz wynikowy jako czarny
    result_image = np.copy(binary_image)

    # Iteruj przez kontury
    for contour in contours:
        # Oblicz pole powierzchni konturu
        area = cv2.contourArea(contour)

        # Jeśli pole powierzchni konturu jest większe niż minimalne, narysuj go na obrazie wynikowym
        if area <= min_contour_area:
            cv2.drawContours(result_image, [contour], 0, 0, -1)

    return result_image

def keep_nearest_blob_only(binary_image, min_blob_size):
    # Znajdź kontury na obrazie binarnym
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Oblicz środek obrazu
    center = (binary_image.shape[1] // 2, binary_image.shape[0] // 2)

    # Inicjalizuj najbliższy blob jako None
    nearest_blob = None
    nearest_distance = float('inf')

    # Iteruj przez wszystkie znalezione kontury
    for contour in contours:
        # Oblicz pole powierzchni konturu
        area = cv2.contourArea(contour)

        # Jeśli pole powierzchni konturu jest większe niż minimalne
        if area >= min_blob_size:
            # Znajdź środek konturu
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # Oblicz odległość między środkiem konturu a środkiem obrazu
            distance = np.sqrt((cX - center[0]) ** 2 + (cY - center[1]) ** 2)

            # Jeśli obecny kontur jest bliżej środka niż poprzedni najlepszy kontur
            if distance < nearest_distance:
                nearest_blob = contour
                nearest_distance = distance

    # Utwórz obraz wynikowy zaznaczając tylko najbliższy blob
    result_image = np.copy(binary_image)
    for contour in contours:
        if contour.shape != nearest_blob.shape or (contour != nearest_blob).all():
            cv2.drawContours(result_image, [contour], -1, 0, thickness=cv2.FILLED)

    return result_image

def divide_image_with_buffer(image, buffer_size):
    # Sprawdź rozmiary obrazu
    height, width = image.shape

    # Podziel obraz na 9x9 fragmentów
    fragment_height = height // 9
    fragment_width = width // 9

    fragments = []

    # Iteruj przez fragmenty obrazu
    for i in range(9):
        for j in range(9):
            # Pobierz indeksy dla aktualnego fragmentu
            start_row = i * fragment_height
            end_row = (i + 1) * fragment_height
            start_col = j * fragment_width
            end_col = (j + 1) * fragment_width

            # Dodaj bufor wokół aktualnego fragmentu
            start_row_buffered = max(start_row - buffer_size, 0)
            end_row_buffered = min(end_row + buffer_size, height)
            start_col_buffered = max(start_col - buffer_size, 0)
            end_col_buffered = min(end_col + buffer_size, width)

            # Wyodrębnij fragment z buforem
            fragment_with_buffer = image[start_row_buffered:end_row_buffered, start_col_buffered:end_col_buffered]

            # Jeśli fragment jest na skraju obrazu, dodaj czarne piksele do bufora
            if start_row_buffered == 0:
                top_buffer = np.zeros((buffer_size, fragment_with_buffer.shape[1]), dtype=np.uint8)
                fragment_with_buffer = np.vstack((top_buffer, fragment_with_buffer))
            if end_row_buffered == height:
                bottom_buffer = np.zeros((buffer_size, fragment_with_buffer.shape[1]), dtype=np.uint8)
                fragment_with_buffer = np.vstack((fragment_with_buffer, bottom_buffer))
            if start_col_buffered == 0:
                left_buffer = np.zeros((fragment_with_buffer.shape[0], buffer_size), dtype=np.uint8)
                fragment_with_buffer = np.hstack((left_buffer, fragment_with_buffer))
            if end_col_buffered == width:
                right_buffer = np.zeros((fragment_with_buffer.shape[0], buffer_size), dtype=np.uint8)
                fragment_with_buffer = np.hstack((fragment_with_buffer, right_buffer))

            fragments.append(fragment_with_buffer)

    return fragments

def replace_chars(text):
    list_of_numbers = re.findall(r'\d+', text)
    result_number = ''.join(list_of_numbers)
    return result_number

def extract_from_cell(thresh, debug = False,  min_contour_area_percent = 0.04):
    # img = cv2.resize(image, (50, 50), interpolation = cv2.INTER_LINEAR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if debug: lista = [thresh]
    thresh = clear_border(thresh)
    if debug: lista.append(thresh)
    # thresh = remove_frame_pixels(thresh, 5)
    w,h = thresh.shape

    thresh = remove_noise_and_keep_large_contours(thresh,min_contour_area_percent)
    if debug: lista.append(thresh)

    thresh = keep_nearest_blob_only(thresh,(w * h)*min_contour_area_percent)
    if debug: lista.append(thresh)

    percentFilled = cv2.countNonZero(thresh) / float(w * h)
    text = ""
    if debug:
            debug_photo = np.concatenate(lista, axis=1)
            # print("Predicted: " + text)
            cv2.imshow("debug", debug_photo)
            cv2.waitKey(0)

    if percentFilled > min_contour_area_percent:
        
        input_to_model = cv2.resize(thresh, (28,28), interpolation = cv2.INTER_LINEAR)
        text += pytesseract.image_to_string(input_to_model, lang='eng',config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789')
        input_to_model = cv2.resize(thresh, (40,40), interpolation = cv2.INTER_LINEAR)
        text += pytesseract.image_to_string(input_to_model, lang='eng',config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789')
        input_to_model = cv2.resize(thresh, (50,50), interpolation = cv2.INTER_LINEAR)
        text += pytesseract.image_to_string(input_to_model, lang='eng',config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789')
        input_to_model = cv2.resize(thresh, (70,70), interpolation = cv2.INTER_LINEAR)
        text += pytesseract.image_to_string(input_to_model, lang='eng',config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789')
        text = replace_chars(text)
        
        if text!= "":
            text_count = Counter(text)
            text = text_count.most_common(1)[0][0]

        if debug:
            print("Predicted: " + text)
        
    return text

def get_divided_image(original, corners = None):
    sudoku = get_square_box_from_image(original, corners)

    img = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (540, 540), interpolation = cv2.INTER_LINEAR)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 27, 10)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel)

    return divide_image_with_buffer(thresh, 10)

# def detect_sudoku(original, corners = None, signal = None, debug = False):
    
#     sudoku = get_square_box_from_image(original, corners)

#     img = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY)
#     img = cv2.resize(img, (540, 540), interpolation = cv2.INTER_LINEAR)
#     thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 27, 10)

#     kernel = np.ones((3, 3), np.uint8) 

#     # thresh = cv2.erode(thresh, kernel)
#     thresh = cv2.dilate(thresh, kernel)
#     # thresh = clear_border(thresh)

#     # thresh = cv2.erode(thresh, kernel)

#     # cv2.imshow("Cell Thresh", thresh)
#     # cv2.waitKey(0)

#     sudoku_board = []
#     for i, cell in enumerate(divide_image_with_buffer(thresh, 10)):
#         if i%9 == 0:
#             sudoku_board.append([])
#         if not signal is None:
#             signal.emit(i/81)
        
#         # cv2.imshow("Cell Thresh", cell)
#         # cv2.waitKey(0)
#         num = extract_from_cell(cell, debug)
#         if(num == ""):
#             num = 0
#         else:
#             num = int(num)
        
#         sudoku_board[-1].append(num)

#     return sudoku_board
