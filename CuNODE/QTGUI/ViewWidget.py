# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:37:30 2024

@author: cca79
"""

from pyvistaqt import QtInteractor
import pyvista as pv #example only
from PySide6 import QWidget, QVBoxLayout


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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.pv_widget.resize(self.size())
