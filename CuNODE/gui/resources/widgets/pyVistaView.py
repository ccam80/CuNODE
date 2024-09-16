# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:37:30 2024

@author: cca79
"""

from os import environ
environ["QT_API"] = "pyside6"

from pyvistaqt import QtInteractor#, BackgroundPlotter
import pyvista as pv #example only
from qtpy.QtWidgets import QWidget, QVBoxLayout
import numpy as np
from gui.resources.qt_animation import Animation

class pyVistaView(QWidget):
    def __init__(self, parent=None, messaging_service=None):
        super().__init__(parent)
        self.initUI()
        self.animation = None
        if messaging_service:
            self.register_messaging_service(messaging_service)
        else:
            self.messenger = None

    def register_messaging_service(self, messaging_service):
        """Connect a message passing service to communicate between widgets.
        Subscribe all setter functions that don't generate their own data or get
        it from the user interface.

        Args:
            messaging_service(class): A class with publish, subscribe, unsubscribe methods"""

        messaging_service.subscribe("update_plot", self.update_plot)
        messaging_service.subscribe("pause_animation", self.pause_animation)
        messaging_service.subscribe("resume_animation", self.resume_animation)
        self.messenger = messaging_service

    def publish(self, topic, data):
        self.messenger.publish(topic, data)


    def initUI(self):
        layout = QVBoxLayout(self)

        # Create PyVista interactor
        self.pv_widget = QtInteractor()
        layout.addWidget(self.pv_widget)
        self.setLayout(layout)
        self.load_example_mesh()

    def load_example_mesh(self):
        example_mesh = pv.Sphere()  # Example mesh, you can choose any other mesh
        self.pv_widget.add_mesh(example_mesh)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.pv_widget.resize(self.size())

    def update_plot(self, plot_data):
        data = plot_data['data']
        scales = plot_data['scales']
        style = plot_data['style']
        axis_labels = plot_data['axis_labels']
        animate = plot_data['animate']

        axis_ranges = {}

        self.pv_widget.clear()

        for axis in ['x', 'y', 'z']:
            axis_ranges[axis] = [np.min(data[axis]), np.max(data[axis])]
            if scales[axis] == 'Logarithmic':
                data[axis] = self.transform_data(scales[axis], data[axis])
                axis_ranges[axis] = [np.log10(np.min(data[axis])), np.log10(np.max(data[axis]))]
                axis_labels[axis] = "log_10(" + axis_labels[axis] + ")"

        self.set_scale(data)
        if style == 'surface':
            self.plot_surface(data, axis_labels, animate, axis_ranges)

    def transform_data(self, scale, data):
        if scale == 'Logarithmic':
            data =  np.log10(data)
        return data

    def set_scale(self, data_vectors, xweight=1, yweight=1, zweight=1):
        """ Scale widget such that each axis is of the same visual length, regardless of
        data scaling. xweight, yweight, zweight allow you to tweak the scale, and sets
        the relative size of each axis. """
        zscale = None
        xscale = 1 * xweight
        x_ptp = np.ptp(data_vectors['x'])
        y_ptp = np.ptp(data_vectors['y'])
        yscale = x_ptp / y_ptp * yweight
        if 'z' in data_vectors.keys():
            z_ptp = np.ptp(data_vectors['z'])
            zscale = x_ptp / z_ptp * zweight

        self.pv_widget.set_scale(xscale=xscale, yscale=yscale, zscale=zscale)

    def plot_surface(self, data, labels, animate, axis_ranges):
        X = data['x']
        Y = data['y']
        Z = data['z']
        axis_ranges_list = [axis_ranges['x'][0], axis_ranges['x'][1],
                            axis_ranges['y'][0], axis_ranges['y'][1],
                            axis_ranges['z'][0], axis_ranges['z'][1]]

        if type(Z) == list:
            Z = Z[0]

        grid = pv.StructuredGrid(X, Y, Z)

        self.actor = self.pv_widget.add_mesh(grid,
                                             scalars=grid.points[:, -1],
                                             show_edges=True,
                                             scalar_bar_args={'vertical': True})

        self.pv_widget.show_grid(xtitle=labels['x'],
                                 ytitle=labels['y'],
                                 ztitle=labels['z'],
                                 axes_ranges=axis_ranges_list)
        self.pv_widget.show()

        if animate:
            def step_animation(step):
                grid.points[:,-1] = data['z'][step].ravel(order='F')
                grid.scalars = grid.points[:,-1]
                scalars_name=grid.active_scalars_name
                self.actor.mapper.set_scalars(grid.scalars, scalars_name)
                self.publish('animation_step',step)

            num_frames = len(data['z'])
            #Dynamically set duration from speed setting
            self.animation = Animation(max_steps=num_frames, interval=100, callback=step_animation)

    def pause_animation(self):
        if self.animation:
            self.animation.pause()

    def resume_animation(self):
        if self.animation:
            self.animation.pause()

    def plot_phase3d(self, x, y, z):
        points = np.column_stack((x, y, z))
        line = self.polyline_from_points(points)
        line["scalars"] = np.arange(line.n_points) # This is for shading i think

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
        # poly.lines = cells
        return poly