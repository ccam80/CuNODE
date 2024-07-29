# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:37:30 2024

@author: cca79
"""

from os import environ
environ["QT_API"] = "pyside6"

from pyvistaqt import QtInteractor
import pyvista as pv #example only
from qtpy.QtWidgets import QWidget, QVBoxLayout
import numpy as np

class pyVistaView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Create PyVista interactor
        self.pv_widget = QtInteractor(self)
        layout.addWidget(self.pv_widget)

        # Set the layout to the widget
        self.setLayout(layout)

        # Load and display an example mesh
        self.load_example_mesh()

    def load_example_mesh(self):
        example_mesh = pv.Sphere()  # Example mesh, you can choose any other mesh
        self.pv_widget.add_mesh(example_mesh)

    def generate_param_grid(self):
        self.pv_widget.clear()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.pv_widget.resize(self.size())

    # def setup_axes(self, labels, transforms):

    def plot_3d_from_mesh(self, X, Y, Z):
        grid = pv.StructuredGrid(X, Y, Z)
        grid.plot(cmap='viridis')

    def plot_3d_from_vectors(self, x, y, z, gridshape):
        mesh = pv.StructuredGrid()
        mesh.points = np.columnstack(x,y,z)
        mesh.dimensions = [gridshape[0], gridshape[1], 1]
        mesh.plot(cmap='viridis')

    def plot_phase3d(self, x, y, z):
        points = np.column_stack((x, y, z))
        line = self.polyline_from_points(points)
        line["scalars"] = np.arange(polyline.n_points) # This is for shading i think

        tube = line.tube(radius=0.1)
        tube.plot()

    def polyline_from_points(points):
        """Given an array of points, make a single polyline.
        Taken entirely from docs.pyvista.org - creating a spline.
        Comments for my understanding"""
        poly = pv.PolyData() # The pyvista polygon data class - "surface geometry"
        poly.points = points # polydata is defined in vertices, lines, polygons - these are vertices
        the_cell = np.arange(0, len(points), dtype=np.int_) # one big cell, differs from small cells in that it's 0d (scalar) I think.
        the_cell = np.insert(the_cell, 0, len(points))
        poly.lines = cells
        return poly
