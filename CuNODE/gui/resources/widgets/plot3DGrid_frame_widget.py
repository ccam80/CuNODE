# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 21:00:07 2024

@author: cca79
"""
from qtpy.QtWidgets import QFrame, QAbstractButton
from qtpy.QtCore import Slot, Signal
from gui.resources.widgets.qtdesigner.plot_controller import Ui_plotController
import logging

class plot_controller_widget(QFrame, Ui_plotController):

    def __init__(self, parent=None):
        super(plot_controller_widget, self).__init__(parent)
        self.setupUi(self)
