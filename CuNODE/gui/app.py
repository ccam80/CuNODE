# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:22:03 2024

@author: cca79
"""

from gui.GUI import ODE_GUI
from qtpy.QtWidgets import QApplication
import sys

def run():
    app = QApplication(sys.argv)
    window = ODE_GUI()
    window.show()
    sys.exit(app.exec())