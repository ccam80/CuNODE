# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 20:31:01 2024

@author: cca79
"""

from qtpy.QtCore import QTimer, Slot

class Animation():
    #Modified from pyvistaqt and pyvista's timer callback functions to e QT compatible
    # and counted in steps
    def __init__(self, max_steps, callback, interval=100, parent=None):
        self.timer = QTimer(parent=parent)
        self.timer.timeout.connect(self.execute_callback)
        self.step = 0
        self.max_steps = max_steps
        self.callback = callback

        self.running = True
        self.timer.start(interval)

    def execute_callback(self):
        if self.running:
            if self.step < self.max_steps:
                self.callback(self.step)
                self.step += 1
            else:
                self.timer.stop()
                self.running = False

    @Slot()
    def pause(self):
        self.running = False

    @Slot()
    def play(self):
        self.running = True