from pathlib import Path
import zarr
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QScrollArea
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint
import sys


class ScrollViewer(QMainWindow):
    def __init__(self, zarr_path, chunk_size=16):
        super().__init__()
        self.zarr_path = zarr_path
        self.chunk_size = chunk_size
        self.current_index = 0
        self.last_clicked_pos = QPoint(0, 0)  # Store last clicked position

        # Open zarr array
        self.zarr_array = zarr.open(zarr_path, mode="r")
        self.total_slices = self.zarr_array.shape[0]  # z dimension
        self.height = self.zarr_array.shape[1]  # y dimension
        self.width = self.zarr_array.shape[2]  # x dimension

        # Initialize the current chunk buffer
        self.load_current_chunk()
        self.init_ui()
        self.update_image()

    def load_current_chunk(self):
        chunk_start = (self.current_index // self.chunk_size) * self.chunk_size
        chunk_end = min(chunk_start + self.chunk_size, self.total_slices)
        self.current_chunk = self.zarr_array[chunk_start:chunk_end]
        self.chunk_start_idx = chunk_start

    def get_current_slice(self):
        relative_idx = self.current_index - self.chunk_start_idx
        return self.current_chunk[relative_idx]

    def init_ui(self):
        self.setWindowTitle('Coordinate Tracker')
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMouseTracking(True)
        self.scroll_area.setWidget(self.image_label)

        layout.addWidget(self.scroll_area)
        self.image_label.installEventFilter(self)

    def eventFilter(self, source, event):
        if source == self.image_label:
            if event.type() == event.MouseButtonPress:
                self.last_clicked_pos = event.pos()
            elif event.type() == event.MouseMove:
                self.last_clicked_pos = event.pos()
        return super().eventFilter(source, event)

    def get_image_coordinates(self):
        # Get scroll positions
        scroll_x = self.scroll_area.horizontalScrollBar().value()
        scroll_y = self.scroll_area.verticalScrollBar().value()

        # Add scroll position to last clicked position
        total_x = self.last_clicked_pos.x() + scroll_x
        total_y = self.last_clicked_pos.y() + scroll_y

        # Get label and image sizes
        label_size = self.image_label.size()
        pixmap = self.image_label.pixmap()
        if pixmap:
            image_size = pixmap.size()

            # Calculate scaling factors
            scale_x = self.width / image_size.width()
            scale_y = self.height / image_size.height()

            # Scale coordinates
            image_x = int(total_x * scale_x)
            image_y = int(total_y * scale_y)

            # Clamp to image boundaries
            image_x = max(0, min(image_x, self.width - 1))
            image_y = max(0, min(image_y, self.height - 1))

            return image_x, image_y
        return 0, 0

    def update_image(self):
        img_data = self.get_current_slice()
        img_normalized = ((img_data - img_data.min()) * (255.0 / (img_data.max() - img_data.min()))).astype(np.uint8)

        height, width = img_normalized.shape
        bytes_per_line = width
        q_img = QImage(img_normalized.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

        # Print coordinates after updating image
        x, y = self.get_image_coordinates()
        print(f"Frame: {self.current_index}, X: {x}, Y: {y}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_W:
            self.current_index = (self.current_index + 1) % self.total_slices
            if self.current_index >= self.chunk_start_idx + self.chunk_size:
                self.load_current_chunk()
            self.update_image()
        elif event.key() == Qt.Key_S:
            self.current_index = (self.current_index - 1) % self.total_slices
            if self.current_index < self.chunk_start_idx:
                self.load_current_chunk()
            self.update_image()
        event.accept()


if __name__ == "__main__":
    SCROLL_PATH = Path("/Volumes/vesuvius/dl.ash2txt.org/data/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1B.zarr/0")

    app = QApplication(sys.argv)
    viewer = ScrollViewer(SCROLL_PATH, chunk_size=16)
    viewer.show()
    sys.exit(app.exec_())