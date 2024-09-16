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
from qtpy.QtWidgets import (QApplication, QFrame, QGridLayout, QLayout,
    QMainWindow, QMenu, QMenuBar, QSizePolicy,
    QStatusBar, QToolBox, QVBoxLayout, QWidget)

from gui.resources.widgets.plot_controller_widget import plot_controller_widget
from gui.resources.widgets.pyVistaView import pyVistaView
from gui.resources.widgets.sim_controller_widget import sim_controller_widget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1726, 872)
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
        self.controlToolBox = QToolBox(self.controlFrame)
        self.controlToolBox.setObjectName(u"controlToolBox")
        font = QFont()
        font.setPointSize(16)
        self.controlToolBox.setFont(font)
        self.plotController_page = QWidget()
        self.plotController_page.setObjectName(u"plotController_page")
        self.plotController_page.setGeometry(QRect(0, 0, 547, 725))
        sizePolicy.setHeightForWidth(self.plotController_page.sizePolicy().hasHeightForWidth())
        self.plotController_page.setSizePolicy(sizePolicy)
        self.verticalLayout_3 = QVBoxLayout(self.plotController_page)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.plotController = plot_controller_widget(self.plotController_page)
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

        self.verticalLayout_3.addWidget(self.plotController)

        self.controlToolBox.addItem(self.plotController_page, u"Plot Settings")
        self.simController_page = QWidget()
        self.simController_page.setObjectName(u"simController_page")
        self.simController_page.setGeometry(QRect(0, 0, 547, 725))
        sizePolicy.setHeightForWidth(self.simController_page.sizePolicy().hasHeightForWidth())
        self.simController_page.setSizePolicy(sizePolicy)
        self.verticalLayout = QVBoxLayout(self.simController_page)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.simController = sim_controller_widget(self.simController_page)
        self.simController.setObjectName(u"simController")
        sizePolicy2.setHeightForWidth(self.simController.sizePolicy().hasHeightForWidth())
        self.simController.setSizePolicy(sizePolicy2)
        self.simController.setFrameShape(QFrame.Shape.StyledPanel)
        self.simController.setFrameShadow(QFrame.Shadow.Raised)

        self.verticalLayout.addWidget(self.simController)

        self.controlToolBox.addItem(self.simController_page, u"Simulation Settings")

        self.verticalLayout_2.addWidget(self.controlToolBox)

        self.verticalLayout_2.setStretch(0, 9)

        self.gridLayout_2.addWidget(self.controlFrame, 0, 1, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1726, 22))
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
        self.actionLoad_System_File.triggered.connect(MainWindow.load_system_from_filedialog)
        self.simController.solve_request.connect(MainWindow.on_solve_request)
        self.plotController.updatePlot.connect(MainWindow.update_plot)

        self.controlToolBox.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionExit.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.actionLoad_System_File.setText(QCoreApplication.translate("MainWindow", u"Load System File", None))
        self.action64_bit_2.setText(QCoreApplication.translate("MainWindow", u"64-bit", None))
        self.action32_bit_2.setText(QCoreApplication.translate("MainWindow", u"32-bit", None))
        self.controlToolBox.setItemText(self.controlToolBox.indexOf(self.plotController_page), QCoreApplication.translate("MainWindow", u"Plot Settings", None))
        self.controlToolBox.setItemText(self.controlToolBox.indexOf(self.simController_page), QCoreApplication.translate("MainWindow", u"Simulation Settings", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuPrecision.setTitle(QCoreApplication.translate("MainWindow", u"Precision", None))
    # retranslateUi

