# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:58:11 2024

@author: cca79
"""
from qtpy.QtWidgets import QFrame, QAbstractButton
from qtpy.QtCore import Slot, Signal
from QT_designer_source.custom_widgets.variable_from_to_scale import Ui_variable_from_to_scale
import logging

class variable_from_to_scale_widget(QFrame, Ui_variable_from_to_scale):
    variable_changed = Signal(str)
    from_changed = Signal(str)
    to_changed = Signal(str)
    scale_changed = Signal(str)

    def __init__(self, parent=None):
        super(variable_from_to_scale_widget, self).__init__(parent)
        self.setupUi(self)


    @Slot(str)
    def on_var_change(self, new_value):
        var = self.Var_dd.currentText()
        self.variable_changed.emit(var)
        logging.info(f"Variable changed to: {var}")

    @Slot()
    def on_from_change(self):
        _from = self.from_entry.toPlainText()
        self.variable_changed.emit(_from)
        logging.info(f"From changed to: {_from}")

    @Slot()
    def on_to_change(self):
        _to = self.from_entry.toPlainText()
        self.variable_changed.emit(_to)
        logging.info(f"To changed to: {_to}")

    @Slot(QAbstractButton)
    def on_scale_change(self, button):
        scale = button.text()
        self.variable_changed.emit(scale)
        logging.info(f"scale changed to: {scale}")

    @Slot(str)
    def append_to_variable_list(self, item):
        self.Var_dd.addItems(item)


    @Slot()
    def clear_variable_list(self):
        self.Var_dd.clear()

    @Slot(str)
    def edit_from(self, text):
        self.from_entry.setPlainText(text)
        pass

    @Slot(str)
    def edit_to(self, text):
        self.to_entry.setPlainText(text)
        pass
