# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 21:40:56 2024

@author: cca79
"""


from qtpy.QtWidgets import QFrame, QMenu
from qtpy.QtCore import Signal, Slot, Property
from qtpy.QtGui import QAction
from gui.resources.widgets.qtdesigner.slider_entry_animate_ui import Ui_slider_entry_animate
from numpy import zeros, argmin

class slider_entry_animate(QFrame, Ui_slider_entry_animate):
    """Custom widget containing a label, slider, entry, and animate button."""

    index_changed = Signal(int)
    animate = Signal()
    animation_speed_change = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self._vector = zeros(0)

    def populate_animation_speed_menus(self):
        speed_options = ['0.5x', '1.0x', '1.5x', '2.0x', '2.5x']

        def create_action(speed_text):
            action = QAction(speed_text, self)
            action.triggered.connect(lambda: self.set_animation_speed(speed_text))
            return action

        menu = QMenu(self.animate_button)
        for speed in speed_options:
            menu.addAction(create_action(speed))
        self.animate_button.setMenu(menu)

    def set_animation_speed(self, speed):
        speed = float(speed.rstrip('x'))

    @Slot(int)
    def on_slider_value_changed(self, index):
        self.index_changed.emit(index)
        float_val = self._vector[index]

    @Slot(int)
    def set_slider_index(self, index):
        self.slider.setValue(index)
        try:
            self.set_entry_value(self._vector[index])
        except IndexError:
            pass

    @Slot(float)
    def set_entry_value(self, value):
        self.entry.setText(f"{value:.2f}")

    @Slot()
    def on_entry_edit_finished(self):
        try:
            entry_value = float(self.entry.text())
        except ValueError:
            value = 0.0
        index = argmin(abs(self._vector - entry_value))
        self.on_slider_value_changed(index)
        return value

    @Slot()
    def on_animate_button_clicked(self):
        self.animate.emit()

    def set_label_text(self, text):
        self.label.setText(text)

    def get_label_text(self):
        return self.label.text()


    labelText = Property(str, get_label_text, set_label_text)

    def set_entry_default(self, value):
        self.entry.setText(f"{value:.2f}")

    def get_entry_default(self):
        return self.entry.text()

    entryDefault = Property(str, get_entry_default, set_entry_default)

    def set_vector(self, vector):
        self._vector = vector

    def get_vector(self):
        return self._vector

    vector = Property(list, get_vector, set_vector)