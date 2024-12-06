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
import snic

class VolumeViewer(QMainWindow):
    def __init__(self, centroids, values):
        super().__init__()
        self.centroids = np.array(centroids)
        self.values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize values
        v_min, v_max = self.values.min(), self.values.max()
        if v_min == v_max:
            self.normalized_values = np.zeros_like(self.values, dtype=np.float32)
        else:
            self.normalized_values = ((self.values - v_min) / (v_max - v_min)).astype(np.float32)

        self.setup_ui()
        self.create_visualization()

    def create_visualization(self):
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.01, 0.01, 0.01)

        # Create points and scalars
        points = vtk.vtkPoints()
        scalars = vtk.vtkFloatArray()

        # Scale points if needed
        max_coord = np.max(np.abs(self.centroids))
        if max_coord == 0:
            scaled_points = self.centroids
        else:
            scaled_points = self.centroids / max_coord * 99

        for point, value in zip(scaled_points, self.normalized_values):
            points.InsertNextPoint(point)
            scalars.InsertNextValue(value)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().SetScalars(scalars)

        # Create sphere glyph
        sphere = vtk.vtkSphereSource()
        sphere.SetPhiResolution(4)
        sphere.SetThetaResolution(4)
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
        light1.SetIntensity(1.0)

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

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_threshold)
        layout.addWidget(self.slider)

        self.resize(800, 600)

    def update_threshold(self):
        threshold = self.slider.value() / 100.0

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
    processed = (processed * 255.0 / processed.max()).astype(np.uint8)

    # Apply GLCAE
    processed = pipeline.apply_glcae_3d(processed)

    # Apply CLAHE
    processed = skimage.exposure.equalize_adapthist(
        processed.astype(np.float32),
        nbins=256,
        clip_limit=0.05
    )

    # Convert back to uint8 for SNIC
    processed = (processed * 255).astype(np.uint8)

    print("Running SNIC...")
    start_time = time.time()

    # Run SNIC with fixed parameters for 256³ volume
    # Run SNIC
    labels, superpixels = snic.run_snic(
        processed,
        d_seed=2,  # Controls number of superpixels
        compactness=1.0,  # Spatial regularization
    )

    print(f"SNIC completed in {time.time() - start_time:.2f} seconds")

    # Extract features
    centroids = [(sp.x, sp.y, sp.z) for sp in superpixels[1:]]
    values = np.array([sp.c for sp in superpixels[1:]])

    # Optionally save results
    if output_path:
        np.savez(output_path,
                 labels=labels,
                 centroids=centroids,
                 values=values)

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