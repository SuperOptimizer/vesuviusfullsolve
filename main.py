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

import snic
import supervoxeler
from pathlib import Path
import metrics
import matplotlib.pyplot as plt
import sys
from PyQt5.QtWidgets import QApplication
import zarr
import time
import pipeline
import supervoxeler
import snic
from pathlib import Path

class VolumeViewer(QMainWindow):
    def __init__(self, centroids, values, d_seed):
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
        self.d_seed = d_seed

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
        self.sphere.SetPhiResolution(2)
        self.sphere.SetThetaResolution(2)
        self.sphere.SetRadius(1.0)

        self.glyph3D = vtk.vtkGlyph3D()
        self.glyph3D.SetSourceConnection(self.sphere.GetOutputPort())
        self.glyph3D.SetInputData(self.polydata)
        self.glyph3D.SetScaleModeToScaleByScalar()
        self.glyph3D.SetScaleFactor(0.5 * self.d_seed)
        self.glyph3D.SetColorModeToColorByScalar()

    def setup_visualization(self):
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.01, 0.01, 0.01)

        # Enable shadows
        self.renderer.SetUseShadows(1)
        self.renderer.SetTwoSidedLighting(True)

        # Create lookup table
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        for i in range(256):
            color = cm.viridis(i / 255.0)
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
        self.actor.GetProperty().SetSpecular(0.5)
        self.actor.GetProperty().SetSpecularPower(20)

        # Enable shadow casting for the actor
        self.actor.GetProperty().SetAmbient(0.7)
        self.actor.GetProperty().SetDiffuse(0.7)

        # Add bright point light above the mesh
        point_light = vtk.vtkLight()
        #point_light.SetLightTypeToPositional()  # Make it a point light
        point_light.SetPosition(127.5, 127.5, 300)  # Position above the center of the 0-255 volume
        point_light.SetFocalPoint(127.5, 127.5, 127.5)  # Point at center of volume
        point_light.SetIntensity(1.5)  # Make it bright
        point_light.SetColor(1, 1, 1)  # White light
        point_light.SetConeAngle(90)
        point_light.SetPositional(True)

        # Add ambient light for base illumination
        ambient_light = vtk.vtkLight()
        ambient_light.SetLightTypeToHeadlight()
        ambient_light.SetIntensity(0.7)  # Reduced ambient intensity to make point light more prominent

        # Add all lights to renderer
        self.renderer.AddLight(point_light)
        self.renderer.AddLight(ambient_light)

        self.renderer.AddActor(self.actor)
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

def extract_supervoxels_from_grid(grid_result):
    """Extract valid supervoxels from the grid structure."""
    centroids = []
    values = []

    # Process supervoxels
    for z in range(supervoxeler.GRID_DIM):
        for y in range(supervoxeler.GRID_DIM):
            for x in range(supervoxeler.GRID_DIM):
                sv = grid_result[z, y, x]
                if sv['n'] > 0 and sv['c'] > 0:
                    centroids.append((sv['z'], sv['y'], sv['x']))
                    values.append(sv['c'])

    return centroids, np.array(values)

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
    #processed = skimage.filters.gaussian(processed, sigma=1.0)

    processed = pipeline.segment_and_clean(processed,32,192)

    processed = (processed - processed.min()) / (processed.max() - processed.min())
    processed = skimage.exposure.equalize_adapthist(processed,nbins=256,kernel_size=256)

    # Apply unsharp masking
    # First create blurred version for unsharp mask
    #blurred = skimage.filters.gaussian(processed, sigma=1.0)
    # Create the unsharp mask by subtracting blurred from original
    #mask = processed - blurred
    # Add mask back to original with a strength factor
    #strength = 1.0  # Adjust this to control sharpening intensity
    #processed = processed + strength * mask


    # Scale back to 0-255 uint8
    processed = (processed - processed.min()) / (processed.max() - processed.min()) * 255
    processed = processed.astype(np.uint8)
    processed = pipeline.apply_glcae_3d(processed)
    plt.figure(figsize=(10, 6))
    plt.hist(processed.flatten(), bins=256, edgecolor='black')
    plt.title('Histogram of Processed Data after GLCAE')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Also print some basic statistics
    print(f"Processed data statistics:")
    print(f"Min: {processed.min():.2f}")
    print(f"Max: {processed.max():.2f}")
    print(f"Mean: {processed.mean():.2f}")
    print(f"Std: {processed.std():.2f}")

    #metrics.analyze_superpixel_movement(processed,16,16.0 * 16.0 * 16.0)

    print("Supervoxeling ...")
    start_time = time.time()
    # Run supervoxeler with grid-based structure
    grid_result = supervoxeler.run_supervoxeler(processed)
    print(f"Supervoxeling completed in {time.time() - start_time:.2f} seconds")

    # Extract valid supervoxels
    supervoxel_centroids, supervoxel_values = extract_supervoxels_from_grid(grid_result)

    # Calculate and print statistics
    print(f"Generated {len(supervoxel_centroids)} supervoxels")
    if len(supervoxel_values) > 0:
        print(f"Average intensity: {supervoxel_values.mean():.2f}")
        print(f"Intensity std: {supervoxel_values.std():.2f}")

    print("Superclustering ...")
    start_time = time.time()
    labels, superclusters = snic.run_snic(processed, 8, 8.0)
    print(f"Superclustering completed in {time.time() - start_time:.2f} seconds")
    print(f"got {len(superclusters)} superclusters")

    supercluster_centroids = [(sp.z, sp.y, sp.x) for sp in superclusters]
    supercluster_values = np.array([sp.c for sp in superclusters])

    # Calculate and print statistics
    print(f"Generated {len(supercluster_centroids)} superclusters")
    if len(supercluster_values) > 0:
        print(f"Average intensity: {supercluster_values.mean():.2f}")
        print(f"Intensity std: {supercluster_values.std():.2f}")

    #return [supercluster_centroids],[supercluster_values]
    return [supervoxel_centroids, supercluster_centroids], [supervoxel_values, supercluster_values]

if __name__ == "__main__":


    def visualize_volume(centroids, values, d_seed):
        app = QApplication.instance() or QApplication(sys.argv)
        viewer = VolumeViewer(centroids, values, d_seed)
        viewer.show()
        app.exec_()

    # Set up paths
    # keep if commented out
    #SCROLL_PATH = Path("/Users/forrest/dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll5.zarr/0")
    #SCROLL_PATH = Path("/Users/forrest/dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll5.zarr/")
    SCROLL_PATH = Path("/Volumes/vesuvius/dl.ash2txt.org/data/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll5.zarr/0")
    #SCROLL_PATH = Path("/Volumes/vesuvius/dl.ash2txt.org/data/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/2")

    OUTPUT_DIR = Path("output")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Compiling Code...")
    supervoxeler.compile_supervoxeler()
    snic.compile_snic('./c/snic.c','./libsnic.so')

    # Process scroll chunk
    centroids, values = process_chunk(SCROLL_PATH,chunk_coords=[4096,4096,4096])

    # Visualize results if visualization function is available
    print("Visualizing results...")
    visualize_volume(centroids[0], values[0], 2)
    #visualize_volume(centroids[1], values[1], 2)