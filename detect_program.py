import sys
import os
import json
import pyperclip
import numpy as np
from ultralytics import YOLO
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QAction,
    QFileDialog,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QLineEdit,
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt


def validate_railway_car_number(number):
    """
    Checks the correctness of the number of a railway car in Russia.

    Args:
        number: The railway car number as a string.

    Returns:
        True if the number is valid, False otherwise.
    """

    number = number.strip()

    # Check length
    if len(number) != 8:
        return False

    # Check if the first digit is valid
    first_digit = int(number[0])  # Corrected index to 0
    if not 0 <= first_digit <= 9:
        return False

    # Check if the remaining digits are numeric
    for digit in number[1:7]:
        if not digit.isdigit():
            return False

    # Check the control digit
    control_digit = calculate_control_digit(number[:7])
    if control_digit != int(number[7]):  # Corrected index to 7
        return False

    return True


def calculate_control_digit(number):
    """
    Calculates the control digit for a given 7-digit railway car number.

    Args:
        number: The 7-digit railway car number as a string.

    Returns:
        The control digit as an integer.
    """

    weights = [2, 1, 2, 1, 2, 1, 2]
    sum = 0
    for i in range(7):
        digit = int(number[i])
        product = digit * weights[i]
        if product > 9:
            product -= 9
        sum += product

    remainder = sum % 10
    if remainder == 0:
        return 0
    else:
        return 10 - remainder


class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Train Car Number Detection with YOLOv8")
        self.setGeometry(100, 100, 1400, 800)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.text_field = QLineEdit(self)
        self.text_field.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.text_field)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.model_file = None
        self.last_image_file = None
        self.model = None

        self.load_settings()
        self.init_menu()

        if not self.model_file or not os.path.exists(self.model_file):
            self.model_file = self.select_model_file()
            if not self.model_file:
                QMessageBox.critical(self, "Error", "Model file not selected. Exiting.")
                sys.exit()

        self.load_model()
        if self.last_image_file:
            self.detect_objects(self.last_image_file)

    def init_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Image", self)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        select_model_action = QAction("Select Model", self)
        select_model_action.triggered.connect(self.select_model_file_from_menu)
        file_menu.addAction(select_model_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        self.setWindowTitle("Object Detection with YOLOv8")
        self.show()

    def load_settings(self):
        if os.path.exists("settings.json"):
            with open("settings.json", "r") as f:
                settings = json.load(f)
                self.model_file = settings.get("model_file")
                self.last_image_file = settings.get("last_image_file")

    def save_settings(self):
        settings = {
            "model_file": self.model_file,
            "last_image_file": self.last_image_file,
        }
        with open("settings.json", "w") as f:
            json.dump(settings, f)

    def select_model_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Model Files (*.pt);;All Files (*)",
            options=options,
        )
        return file_name

    def select_model_file_from_menu(self):
        file_name = self.select_model_file()
        if file_name:
            self.model_file = file_name
            self.load_model()
            if self.last_image_file:
                self.detect_objects(self.last_image_file)
            self.save_settings()

    def load_model(self):
        if self.model_file:
            self.model = YOLO(self.model_file)

    def open_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
            options=options,
        )
        if file_name:
            self.last_image_file = file_name
            self.detect_objects(file_name)
            self.save_settings()

    def detect_objects(self, image_path):
        results = self.model(image_path)
        self.display_results(results)

    def display_results(self, results):
        if results:
            for result in results:
                # Get the image with detected boxes
                im_bgr = result.plot(
                    conf=True, line_width=1, font_size=10
                )  # BGR-order numpy array
                im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
                im_rgb.save("annotated_image.jpg")  # Save annotated image temporarily

                # Display the image
                self.display_image("annotated_image.jpg")

                # Extract detected digits and try to compose train car numbers
                detected_digits = []
                for box in result.boxes.data.tolist():
                    class_id = int(box[5])
                    confidence = box[4]
                    x1, y1, x2, y2 = map(int, box[:4])
                    digit = class_id
                    detected_digits.append(
                        {
                            "digit": digit,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "confidence": confidence,
                        }
                    )

                # Sort digits by Y and then X coordinates
                # detected_digits.sort(key=lambda x: (x["y1"], x["x1"]))

                # Sort digits by Y + X coordinates
                # detected_digits.sort(key=lambda x: (x["y1"] ** 2 + x["x1"] ** 2))
                detected_digits.sort(key=lambda x: (x["y1"] + x["x1"]))

                # Call the delete_intersected_digits function
                detected_digits = self.delete_intersected_digits(detected_digits)

                # Find and validate train car numbers
                train_car_numbers = self.find_train_car_numbers(detected_digits)
                self.text_field.setText(", ".join(train_car_numbers))

    def find_train_car_numbers(self, detected_digits):
        train_car_numbers = []
        used_digits = set()

        for i, digit_data in enumerate(detected_digits):
            if i in used_digits:
                continue

            potential_number = str(digit_data["digit"])
            last_digit_data = digit_data
            digits_in_number = [i]

            for j in range(i + 1, len(detected_digits)):
                if j in used_digits:
                    continue

                next_digit_data = detected_digits[j]

                # Check if the next digit meets the criteria
                if self.is_valid_next_digit(last_digit_data, next_digit_data):
                    potential_number += str(next_digit_data["digit"])
                    last_digit_data = next_digit_data
                    digits_in_number.append(j)

                    if len(potential_number) == 8:
                        if validate_railway_car_number(potential_number):
                            train_car_numbers.append(potential_number)
                            used_digits.update(digits_in_number)
                        break
                # else:
                # Stop searching for the current number if criteria not met
                # break

        return train_car_numbers

    def is_valid_next_digit(self, last_digit_data, next_digit_data):
        height_ratio = (next_digit_data["y2"] - next_digit_data["y1"]) / (
            last_digit_data["y2"] - last_digit_data["y1"]
        )
        center_y_diff = abs(
            (next_digit_data["y1"] + next_digit_data["y2"]) / 2
            - (last_digit_data["y1"] + last_digit_data["y2"]) / 2
        )
        distance_to_previous = (next_digit_data["x1"] + next_digit_data["x2"]) * 0.5 - (
            last_digit_data["x1"] + last_digit_data["x2"]
        ) * 0.5

        return (
            0.8 <= height_ratio <= 1.2
            and center_y_diff <= 0.2 * (last_digit_data["y2"] - last_digit_data["y1"])
            and distance_to_previous
            <= 3 * (last_digit_data["x2"] - last_digit_data["x1"])
        )

    def display_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def delete_intersected_digits(self, detected_digits):
        # Convert the list of detected digits to a list of tuples for easier removal
        detected_digits = [(i, data) for i, data in enumerate(detected_digits)]

        # Create a list to store the indices of the digits to be removed
        indices_to_remove = set()

        # Iterate over each pair of digits to check for intersection and confidence criteria
        for i, (index_i, digit_data_i) in enumerate(detected_digits):
            for j, digit_data_j in detected_digits[i + 1 :]:
                if self.is_intersected(digit_data_i, digit_data_j):
                    # Check if the current digit's confidence is less than the other digit's confidence
                    if digit_data_i["confidence"] < digit_data_j["confidence"]:
                        # Add the index of the current digit to the list of indices to be removed
                        indices_to_remove.add(index_i)
                    else:
                        indices_to_remove.add(j)

        # Remove the digits that are marked for removal
        detected_digits = [
            data for index, data in detected_digits if index not in indices_to_remove
        ]

        return detected_digits

    def is_intersected(self, digit_data_i, digit_data_j):
        # Calculate the centers of the two digits
        center_i_x = (digit_data_i["x1"] + digit_data_i["x2"]) / 2
        center_i_y = (digit_data_i["y1"] + digit_data_i["y2"]) / 2
        center_j_x = (digit_data_j["x1"] + digit_data_j["x2"]) / 2
        center_j_y = (digit_data_j["y1"] + digit_data_j["y2"]) / 2

        # Calculate the distance between the centers of the two digits
        distance_centers = (
            (center_i_x - center_j_x) ** 2 + (center_i_y - center_j_y) ** 2
        ) ** 0.5

        # Calculate the width of the current digit
        width_i = digit_data_i["x2"] - digit_data_i["x1"]

        # Check if the distance between the centers is less than half the width of the current digit
        return distance_centers < width_i / 2

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_V and event.modifiers() == Qt.ControlModifier:
            self.paste_image_from_clipboard()

    def paste_image_from_clipboard(self):
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()

        if mime_data.hasImage():
            image = clipboard.image()
            image.save("temp_clipboard_image.jpg")
            self.last_image_file = "temp_clipboard_image.jpg"
            self.detect_objects(self.last_image_file)
            self.save_settings()
        else:
            QMessageBox.warning(self, "Warning", "No valid image found in clipboard.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    sys.exit(app.exec_())
