import skimage.exposure
import vtk
import matplotlib.cm as cm
import zarr
import numpy as np
import pipeline
import snic
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QSlider
from PyQt5.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import time


class VolumeViewer(QMainWindow):
    def __init__(self, centroids, values):
        super().__init__()
        self.centroids = np.array(centroids)
        self.values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize base values to [0,1]
        v_min, v_max = 0,255#self.values.min(), self.values.max()
        if v_min == v_max:
            self.normalized_base = np.zeros_like(self.values, dtype=np.float32)
        else:
            self.normalized_base = ((self.values - v_min) / (v_max - v_min)).astype(np.float32)

        # Apply separate mappings for radius and color
        self.radius_boost = 0.5  # Controls how much small values are boosted for radius
        self.color_boost = 0.0  # Controls how much small values are boosted for color

        self.update_mappings()
        self.setup_ui()
        self.create_visualization()

    def update_mappings(self):
        # Map normalized values to radius (boosted)
        self.radius_values = self.normalized_base + (1 - self.normalized_base) * self.radius_boost

        # Map normalized values to color (potentially different boost)
        self.color_values = self.normalized_base + (1 - self.normalized_base) * self.color_boost

    def create_visualization(self):
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.01, 0.01, 0.01)

        # Create points and scalars
        points = vtk.vtkPoints()
        radius_scalars = vtk.vtkFloatArray()
        color_scalars = vtk.vtkFloatArray()

        # Scale points if needed
        max_coord = np.max(np.abs(self.centroids))
        if max_coord == 0:
            scaled_points = self.centroids
        else:
            scaled_points = self.centroids / max_coord * 99

        for point, radius_val, color_val in zip(scaled_points, self.radius_values, self.color_values):
            points.InsertNextPoint(point)
            radius_scalars.InsertNextValue(radius_val)
            color_scalars.InsertNextValue(color_val)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        # Use radius values for scaling
        polydata.GetPointData().SetScalars(radius_scalars)
        # Add color values as a separate array
        polydata.GetPointData().AddArray(color_scalars)

        # Create sphere glyph
        sphere = vtk.vtkSphereSource()
        sphere.SetPhiResolution(1)
        sphere.SetThetaResolution(1)
        sphere.SetRadius(1.0)

        self.glyph3D = vtk.vtkGlyph3D()
        self.glyph3D.SetSourceConnection(sphere.GetOutputPort())
        self.glyph3D.SetInputData(polydata)
        self.glyph3D.SetScaleModeToScaleByScalar()
        self.glyph3D.SetScaleFactor(0.5)
        self.glyph3D.SetColorModeToColorByScalar()

        # Create lookup table
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        for i in range(256):
            color = cm.inferno(i / 255.0)
            lut.SetTableValue(i, color[0], color[1], color[2], 1.0)
        lut.Build()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.glyph3D.GetOutputPort())
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(0, 1)
        # Use the color values array for coloring
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(1)  # Index 1 is our color_scalars array

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(20)

        self.renderer.AddActor(actor)
        self.polydata = polydata

        # Add lighting
        light1 = vtk.vtkLight()
        light1.SetFocalPoint(0, 0, 0)
        light1.SetPosition(1, 1, 1)
        light1.SetIntensity(1.2)

        light2 = vtk.vtkLight()
        light2.SetFocalPoint(0, 0, 0)
        light2.SetPosition(-1, -1, -1)
        light2.SetIntensity(1.2)

        self.renderer.AddLight(light1)
        self.renderer.AddLight(light2)

        self.renderer.ResetCamera()
        self.vtk_widget.Initialize()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.vtk_widget = QVTKRenderWindowInteractor()
        layout.addWidget(self.vtk_widget)

        # Value threshold slider
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(0)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        layout.addWidget(self.threshold_slider)

        # Radius boost slider
        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setMinimum(0)
        self.radius_slider.setMaximum(100)
        self.radius_slider.setValue(int(self.radius_boost * 100))
        self.radius_slider.valueChanged.connect(self.update_radius_boost)
        layout.addWidget(self.radius_slider)

        # Color boost slider
        self.color_slider = QSlider(Qt.Horizontal)
        self.color_slider.setMinimum(0)
        self.color_slider.setMaximum(100)
        self.color_slider.setValue(int(self.color_boost * 100))
        self.color_slider.valueChanged.connect(self.update_color_boost)
        layout.addWidget(self.color_slider)

        self.resize(800, 600)

    def update_radius_boost(self):
        self.radius_boost = self.radius_slider.value() / 100.0
        self.update_mappings()
        self.create_visualization()

    def update_color_boost(self):
        self.color_boost = self.color_slider.value() / 100.0
        self.update_mappings()
        self.create_visualization()

    def update_threshold(self):
        threshold = self.threshold_slider.value() / 100.0

        threshold_filter = vtk.vtkThresholdPoints()
        threshold_filter.SetInputData(self.polydata)
        threshold_filter.ThresholdByUpper(threshold)
        self.glyph3D.SetInputConnection(threshold_filter.GetOutputPort())

        self.vtk_widget.GetRenderWindow().Render()


def visualize_volume(centroids, values):
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = VolumeViewer(centroids, values)
    viewer.show()
    app.exec_()


def process_chunk(chunk_path, output_path=None):
    print("Reading chunk...")
    scroll1 = zarr.open(chunk_path, mode="r")

    # Read and process fixed size chunk (256³)
    chunk = scroll1[4096:4096 + 256, 2048:2048 + 256, 2048:2048 + 256]

    print("Processing volume...")
    # Convert to uint8 and normalize
    processed = np.ascontiguousarray(chunk, dtype=np.float32)
    processed = (processed - processed.min()) / (processed.max() - processed.min())
    #processed[processed < 0.1] = 0
    #processed[processed > 0.9] = 1
    #processed = (processed - processed.min()) / (processed.max() - processed.min())


    # Convert back to uint8 for SNIC
    processed = (processed * 255).astype(np.uint8)

    # Apply GLCAE
    processed = pipeline.apply_glcae_3d(processed)

    print("Running SNIC...")
    start_time = time.time()

    # Run SNIC with fixed parameters for 256³ volume
    labels, superpixels, num_superpixels = snic.run_snic(
        processed,
        d_seed=2,
        iso_threshold= 32
    )

    print(f"SNIC completed in {time.time() - start_time:.2f} seconds")
    print(f"Created {num_superpixels} superpixels above threshold")

    # Extract features from valid superpixels (1 to num_superpixels)
    centroids = [(sp.x, sp.y, sp.z) for sp in superpixels[1:num_superpixels + 1]]
    values = np.array([sp.c for sp in superpixels[1:num_superpixels + 1]])

    # Optionally save results
    if output_path:
        np.savez(output_path,
                 labels=labels,
                 centroids=centroids,
                 values=values,
                 num_superpixels=num_superpixels)

    return centroids, values


if __name__ == "__main__":
    # Compile SNIC
    print("Compiling SNIC...")
    snic.compile_snic()

    # Process scroll chunk
    SCROLL_PATH = "/Volumes/vesuvius/dl.ash2txt.org/data/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/0"

    centroids, values = process_chunk(SCROLL_PATH)

    # Visualize results
    print("Visualizing results...")
    visualize_volume(centroids, values)