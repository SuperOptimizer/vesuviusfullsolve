import skimage.exposure
import vtk
import matplotlib.cm as cm
import zarr
import numpy as np
import pipeline
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QSlider
from PyQt5.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import time
import supervoxeler
from pathlib import Path


class VolumeViewer(QMainWindow):
    def __init__(self, centroids, values):
        super().__init__()
        self.centroids = np.array(centroids)
        self.values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize base values to [0,1]
        v_min, v_max = 0, 255
        self.normalized_base = ((self.values - v_min) / (v_max - v_min)).astype(np.float32)

        # Initialize modifiers
        self.radius_modifier = 0.0
        self.color_modifier = 0.0
        self.threshold = 0.0

        self.setup_ui()
        self.setup_vtk_objects()
        self.setup_visualization()
        self.update_mappings()
        self.update_visualization()

    def setup_vtk_objects(self):
        # Initialize VTK pipeline objects
        self.points = vtk.vtkPoints()
        self.radius_scalars = vtk.vtkFloatArray()
        self.color_scalars = vtk.vtkFloatArray()

        # Set initial points and scalars
        max_coord = np.max(np.abs(self.centroids))
        scaled_points = self.centroids / max_coord * 99

        for point, value in zip(scaled_points, self.normalized_base):
            self.points.InsertNextPoint(point)
            self.radius_scalars.InsertNextValue(value)
            self.color_scalars.InsertNextValue(value)

        self.polydata = vtk.vtkPolyData()
        self.polydata.SetPoints(self.points)
        self.polydata.GetPointData().SetScalars(self.radius_scalars)
        self.polydata.GetPointData().AddArray(self.color_scalars)

        self.sphere = vtk.vtkSphereSource()
        self.sphere.SetPhiResolution(1)
        self.sphere.SetThetaResolution(1)
        self.sphere.SetRadius(1.0)

        self.glyph3D = vtk.vtkGlyph3D()
        self.glyph3D.SetSourceConnection(self.sphere.GetOutputPort())
        self.glyph3D.SetInputData(self.polydata)
        self.glyph3D.SetScaleModeToScaleByScalar()
        self.glyph3D.SetScaleFactor(.125)
        self.glyph3D.SetColorModeToColorByScalar()

    def setup_visualization(self):
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.01, 0.01, 0.01)

        # Create lookup table
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        for i in range(256):
            color = cm.inferno(i / 255.0)
            lut.SetTableValue(i, color[0], color[1], color[2], 1.0)
        lut.Build()

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.glyph3D.GetOutputPort())
        self.mapper.SetLookupTable(lut)
        self.mapper.SetScalarRange(0, 1)
        self.mapper.SetScalarModeToUsePointFieldData()
        self.mapper.SelectColorArray(1)

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetSpecular(0.3)
        self.actor.GetProperty().SetSpecularPower(20)

        self.renderer.AddActor(self.actor)

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

    def update_mappings(self):
        if self.radius_modifier >= 0:
            self.radius_values = self.normalized_base * (1 - self.radius_modifier) + self.radius_modifier
        else:
            self.radius_values = self.normalized_base * (1 + self.radius_modifier)

        if self.color_modifier >= 0:
            self.color_values = self.normalized_base * (1 - self.color_modifier) + self.color_modifier
        else:
            self.color_values = self.normalized_base * (1 + self.color_modifier)

    def update_visualization(self):
        valid_indices = self.normalized_base >= self.threshold
        valid_centroids = self.centroids[valid_indices]
        valid_radius = self.radius_values[valid_indices]
        valid_colors = self.color_values[valid_indices]

        self.points.Reset()
        self.radius_scalars.Reset()
        self.color_scalars.Reset()

        if len(valid_centroids) > 0:
            max_coord = np.max(np.abs(valid_centroids))
            scaled_points = valid_centroids / max_coord * 99

            for point, radius_val, color_val in zip(scaled_points, valid_radius, valid_colors):
                self.points.InsertNextPoint(point)
                self.radius_scalars.InsertNextValue(radius_val)
                self.color_scalars.InsertNextValue(color_val)

        self.polydata.Modified()
        self.vtk_widget.GetRenderWindow().Render()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.vtk_widget = QVTKRenderWindowInteractor()
        layout.addWidget(self.vtk_widget)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(0)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        layout.addWidget(self.threshold_slider)

        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setMinimum(-100)
        self.radius_slider.setMaximum(100)
        self.radius_slider.setValue(0)
        self.radius_slider.valueChanged.connect(self.update_radius_modifier)
        layout.addWidget(self.radius_slider)

        self.color_slider = QSlider(Qt.Horizontal)
        self.color_slider.setMinimum(-100)
        self.color_slider.setMaximum(100)
        self.color_slider.setValue(0)
        self.color_slider.valueChanged.connect(self.update_color_modifier)
        layout.addWidget(self.color_slider)

        self.resize(800, 600)

    def update_radius_modifier(self):
        self.radius_modifier = self.radius_slider.value() / 100.0
        self.update_mappings()
        self.update_visualization()

    def update_color_modifier(self):
        self.color_modifier = self.color_slider.value() / 100.0
        self.update_mappings()
        self.update_visualization()

    def update_threshold(self):
        self.threshold = self.threshold_slider.value() / 100.0
        self.update_visualization()


def visualize_volume(centroids, values):
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = VolumeViewer(centroids, values)
    viewer.show()
    app.exec_()



def process_chunk(chunk_path, output_path=None, chunk_coords=(4096, 4096, 4096), chunk_size=256):

    print("Reading chunk...")
    scroll = zarr.open(chunk_path, mode="r")

    # Extract chunk coordinates
    z_start, y_start, x_start = chunk_coords

    # Read and process fixed size chunk
    chunk = scroll[
            z_start:z_start + chunk_size,
            y_start:y_start + chunk_size,
            x_start:x_start + chunk_size
            ]

    print("Processing volume...")

    processed = np.ascontiguousarray(chunk)

    processed = pipeline.apply_glcae_3d(processed)
    processed[processed < 96] = 0
    #processed[processed > 128+64] = 255
    print("Running SNIC...")
    start_time = time.time()

    # Run SNIC with fixed parameters for chunk
    supervoxels, num_supervoxels = supervoxeler.run_supervoxeler(processed)

    print(f"SNIC completed in {time.time() - start_time:.2f} seconds")

    # Extract features and adjust for chunk coordinates
    centroids = [
        (sp.z + z_start, sp.y + y_start, sp.x + x_start)  # Note: changed order to match chunk coords
        for sp in supervoxels[1:]
        if sp.n > 0  # Filter out empty superpixels
    ]
    values = np.array([sp.c for sp in supervoxels[1:] if sp.n > 0])

    # Calculate and print statistics
    print(f"Generated {len(centroids)} superpixels")
    print(f"Average intensity: {values.mean():.2f}")
    print(f"Intensity std: {values.std():.2f}")
    return centroids, values



if __name__ == "__main__":
    # Set up paths
    #SCROLL_PATH = Path("/Volumes/vesuvius/dl.ash2txt.org/data/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/0")
    SCROLL_PATH = Path("/Volumes/vesuvius/dl.ash2txt.org/data/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll5.zarr/0")

    OUTPUT_DIR = Path("output")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Compile SNIC
    print("Compiling SNIC...")
    try:
        supervoxeler.compile_supervoxeler()
    except Exception as e:
        print(f"Error compiling SNIC: {e}")
        raise

    try:
        # Process scroll chunk
        centroids, values = process_chunk(
            SCROLL_PATH,
            output_path=OUTPUT_DIR / "scroll1_chunk_results.npz"
        )

        # Visualize results if visualization function is available
        if 'visualize_volume' in globals():
            print("Visualizing results...")
            visualize_volume(centroids, values)

    except Exception as e:
        print(f"Error in main processing: {e}")
        raise