# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'QT_simGUI.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from qtpy.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from qtpy.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from qtpy.QtWidgets import (QApplication, QFrame, QGridLayout, QHBoxLayout,
    QLayout, QMainWindow, QMenu, QMenuBar,
    QPushButton, QSizePolicy, QStatusBar, QVBoxLayout,
    QWidget)

from gui.resources.widgets.plot_controller_widget import plot_controller_widget
from gui.resources.widgets.pyVistaView import pyVistaView
from gui.resources.widgets.sim_controller_widget import sim_controller_widget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1718, 1364)
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName(u"actionExit")
        self.actionLoad_System_File = QAction(MainWindow)
        self.actionLoad_System_File.setObjectName(u"actionLoad_System_File")
        self.action64_bit_2 = QAction(MainWindow)
        self.action64_bit_2.setObjectName(u"action64_bit_2")
        self.action32_bit_2 = QAction(MainWindow)
        self.action32_bit_2.setObjectName(u"action32_bit_2")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.gridLayout_2 = QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.plotFrame = QFrame(self.centralwidget)
        self.plotFrame.setObjectName(u"plotFrame")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(2)
        sizePolicy1.setVerticalStretch(1)
        sizePolicy1.setHeightForWidth(self.plotFrame.sizePolicy().hasHeightForWidth())
        self.plotFrame.setSizePolicy(sizePolicy1)
        self.plotFrame.setFrameShape(QFrame.Shape.StyledPanel)
        self.plotFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout = QGridLayout(self.plotFrame)
        self.gridLayout.setObjectName(u"gridLayout")
        self.Plotwidget = pyVistaView(self.plotFrame)
        self.Plotwidget.setObjectName(u"Plotwidget")
        sizePolicy1.setHeightForWidth(self.Plotwidget.sizePolicy().hasHeightForWidth())
        self.Plotwidget.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.Plotwidget, 0, 0, 1, 1)


        self.gridLayout_2.addWidget(self.plotFrame, 0, 0, 1, 1)

        self.controlFrame = QFrame(self.centralwidget)
        self.controlFrame.setObjectName(u"controlFrame")
        sizePolicy.setHeightForWidth(self.controlFrame.sizePolicy().hasHeightForWidth())
        self.controlFrame.setSizePolicy(sizePolicy)
        self.controlFrame.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.controlFrame.setFrameShape(QFrame.Shape.StyledPanel)
        self.controlFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.controlFrame.setLineWidth(1)
        self.verticalLayout_2 = QVBoxLayout(self.controlFrame)
        self.verticalLayout_2.setSpacing(1)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setSizeConstraint(QLayout.SizeConstraint.SetMaximumSize)
        self.verticalLayout_2.setContentsMargins(-1, 0, -1, 0)
        self.plotController = plot_controller_widget(self.controlFrame)
        self.plotController.setObjectName(u"plotController")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy2.setHorizontalStretch(1)
        sizePolicy2.setVerticalStretch(5)
        sizePolicy2.setHeightForWidth(self.plotController.sizePolicy().hasHeightForWidth())
        self.plotController.setSizePolicy(sizePolicy2)
        self.plotController.setFrameShape(QFrame.Shape.StyledPanel)
        self.plotController.setFrameShadow(QFrame.Shadow.Raised)
        self.plotController.setLineWidth(1)
        self.plotController.setMidLineWidth(0)

        self.verticalLayout_2.addWidget(self.plotController)

        self.simController = sim_controller_widget(self.controlFrame)
        self.simController.setObjectName(u"simController")
        sizePolicy2.setHeightForWidth(self.simController.sizePolicy().hasHeightForWidth())
        self.simController.setSizePolicy(sizePolicy2)
        self.simController.setFrameShape(QFrame.Shape.StyledPanel)
        self.simController.setFrameShadow(QFrame.Shadow.Raised)

        self.verticalLayout_2.addWidget(self.simController)

        self.saveOrSolve_f = QFrame(self.controlFrame)
        self.saveOrSolve_f.setObjectName(u"saveOrSolve_f")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy3.setHorizontalStretch(1)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.saveOrSolve_f.sizePolicy().hasHeightForWidth())
        self.saveOrSolve_f.setSizePolicy(sizePolicy3)
        self.saveOrSolve_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.saveOrSolve_f.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout = QHBoxLayout(self.saveOrSolve_f)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.save_button = QPushButton(self.saveOrSolve_f)
        self.save_button.setObjectName(u"save_button")
        sizePolicy.setHeightForWidth(self.save_button.sizePolicy().hasHeightForWidth())
        self.save_button.setSizePolicy(sizePolicy)
        font = QFont()
        font.setPointSize(24)
        self.save_button.setFont(font)

        self.horizontalLayout.addWidget(self.save_button)

        self.solve_button = QPushButton(self.saveOrSolve_f)
        self.solve_button.setObjectName(u"solve_button")
        sizePolicy.setHeightForWidth(self.solve_button.sizePolicy().hasHeightForWidth())
        self.solve_button.setSizePolicy(sizePolicy)
        self.solve_button.setFont(font)

        self.horizontalLayout.addWidget(self.solve_button)

        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 1)

        self.verticalLayout_2.addWidget(self.saveOrSolve_f)

        self.verticalLayout_2.setStretch(0, 5)
        self.verticalLayout_2.setStretch(1, 5)
        self.verticalLayout_2.setStretch(2, 1)

        self.gridLayout_2.addWidget(self.controlFrame, 0, 1, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1718, 22))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuPrecision = QMenu(self.menuFile)
        self.menuPrecision.setObjectName(u"menuPrecision")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menuFile.addAction(self.actionLoad_System_File)
        self.menuFile.addAction(self.menuPrecision.menuAction())
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menuPrecision.addAction(self.action64_bit_2)
        self.menuPrecision.addAction(self.action32_bit_2)

        self.retranslateUi(MainWindow)
        self.save_button.clicked.connect(MainWindow.save_results)
        self.solve_button.released.connect(MainWindow.solve_ODE)
        self.actionLoad_System_File.triggered.connect(MainWindow.load_system_from_filedialog)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionExit.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.actionLoad_System_File.setText(QCoreApplication.translate("MainWindow", u"Load System File", None))
        self.action64_bit_2.setText(QCoreApplication.translate("MainWindow", u"64-bit", None))
        self.action32_bit_2.setText(QCoreApplication.translate("MainWindow", u"32-bit", None))
        self.save_button.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.solve_button.setText(QCoreApplication.translate("MainWindow", u"Solve", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuPrecision.setTitle(QCoreApplication.translate("MainWindow", u"Precision", None))
    # retranslateUi

