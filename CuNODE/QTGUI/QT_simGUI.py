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
from qtpy.QtWidgets import (QAbstractScrollArea, QApplication, QButtonGroup, QComboBox,
    QFormLayout, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLCDNumber, QLabel, QLayout,
    QMainWindow, QMenu, QMenuBar, QPlainTextEdit,
    QPushButton, QRadioButton, QSizePolicy, QSlider,
    QSpacerItem, QSplitter, QStatusBar, QTabWidget,
    QTextEdit, QToolBox, QVBoxLayout, QWidget)

from pyVistaView import pyVistaView

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1600, 1364)
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName(u"actionExit")
        self.actionLoad_System_File = QAction(MainWindow)
        self.actionLoad_System_File.setObjectName(u"actionLoad_System_File")
        self.action64_bit = QAction(MainWindow)
        self.action64_bit.setObjectName(u"action64_bit")
        self.action64_bit.setCheckable(True)
        self.action64_bit.setChecked(True)
        self.action32_bit = QAction(MainWindow)
        self.action32_bit.setObjectName(u"action32_bit")
        self.action32_bit.setCheckable(True)
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
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.plotFrame.sizePolicy().hasHeightForWidth())
        self.plotFrame.setSizePolicy(sizePolicy1)
        self.plotFrame.setFrameShape(QFrame.Shape.StyledPanel)
        self.plotFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout = QGridLayout(self.plotFrame)
        self.gridLayout.setObjectName(u"gridLayout")
        self.Plotwidget = pyVistaView(self.plotFrame)
        self.Plotwidget.setObjectName(u"Plotwidget")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.Plotwidget.sizePolicy().hasHeightForWidth())
        self.Plotwidget.setSizePolicy(sizePolicy2)

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
        self.plotSettingsTabs = QTabWidget(self.controlFrame)
        self.plotSettingsTabs.setObjectName(u"plotSettingsTabs")
        sizePolicy.setHeightForWidth(self.plotSettingsTabs.sizePolicy().hasHeightForWidth())
        self.plotSettingsTabs.setSizePolicy(sizePolicy)
        font = QFont()
        font.setPointSize(14)
        font.setStrikeOut(False)
        font.setKerning(False)
        self.plotSettingsTabs.setFont(font)
        self.plotSettingsTabs.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.plotSettingsTabs.setUsesScrollButtons(False)
        self.plot3d_tab = QWidget()
        self.plot3d_tab.setObjectName(u"plot3d_tab")
        sizePolicy.setHeightForWidth(self.plot3d_tab.sizePolicy().hasHeightForWidth())
        self.plot3d_tab.setSizePolicy(sizePolicy)
        self.verticalLayout_5 = QVBoxLayout(self.plot3d_tab)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.xAxis_l = QLabel(self.plot3d_tab)
        self.xAxis_l.setObjectName(u"xAxis_l")
        font1 = QFont()
        font1.setPointSize(14)
        font1.setBold(True)
        font1.setStrikeOut(False)
        font1.setKerning(True)
        self.xAxis_l.setFont(font1)

        self.verticalLayout_5.addWidget(self.xAxis_l)

        self.xAxis_f = QFrame(self.plot3d_tab)
        self.xAxis_f.setObjectName(u"xAxis_f")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        sizePolicy3.setHorizontalStretch(1)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.xAxis_f.sizePolicy().hasHeightForWidth())
        self.xAxis_f.setSizePolicy(sizePolicy3)
        self.xAxis_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.xAxis_f.setFrameShadow(QFrame.Shadow.Raised)
        self.xAxis_f.setLineWidth(5)
        self.gridLayout_3 = QGridLayout(self.xAxis_f)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setSizeConstraint(QLayout.SizeConstraint.SetMaximumSize)
        self.gridLayout_3.setVerticalSpacing(1)
        self.gridLayout_3.setContentsMargins(-1, 1, -1, 1)
        self.xAxisVar_dd = QComboBox(self.xAxis_f)
        self.xAxisVar_dd.addItem("")
        self.xAxisVar_dd.addItem("")
        self.xAxisVar_dd.addItem("")
        self.xAxisVar_dd.addItem("")
        self.xAxisVar_dd.setObjectName(u"xAxisVar_dd")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy4.setHorizontalStretch(1)
        sizePolicy4.setVerticalStretch(1)
        sizePolicy4.setHeightForWidth(self.xAxisVar_dd.sizePolicy().hasHeightForWidth())
        self.xAxisVar_dd.setSizePolicy(sizePolicy4)
        self.xAxisVar_dd.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)

        self.gridLayout_3.addWidget(self.xAxisVar_dd, 1, 0, 1, 1)

        self.xSliceLower_entry = QTextEdit(self.xAxis_f)
        self.xSliceLower_entry.setObjectName(u"xSliceLower_entry")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy5.setHorizontalStretch(1)
        sizePolicy5.setVerticalStretch(1)
        sizePolicy5.setHeightForWidth(self.xSliceLower_entry.sizePolicy().hasHeightForWidth())
        self.xSliceLower_entry.setSizePolicy(sizePolicy5)
        self.xSliceLower_entry.setMaximumSize(QSize(16777215, 40))
        font2 = QFont()
        font2.setPointSize(10)
        font2.setStrikeOut(False)
        font2.setKerning(True)
        self.xSliceLower_entry.setFont(font2)

        self.gridLayout_3.addWidget(self.xSliceLower_entry, 1, 1, 1, 1)

        self.xSliceUpper_entry = QTextEdit(self.xAxis_f)
        self.xSliceUpper_entry.setObjectName(u"xSliceUpper_entry")
        sizePolicy5.setHeightForWidth(self.xSliceUpper_entry.sizePolicy().hasHeightForWidth())
        self.xSliceUpper_entry.setSizePolicy(sizePolicy5)
        self.xSliceUpper_entry.setMaximumSize(QSize(16777215, 40))
        self.xSliceUpper_entry.setFont(font2)
        self.xSliceUpper_entry.setLineWrapMode(QTextEdit.LineWrapMode.FixedColumnWidth)
        self.xSliceUpper_entry.setAcceptRichText(False)

        self.gridLayout_3.addWidget(self.xSliceUpper_entry, 1, 2, 1, 1)

        self.xAxisVar_l = QLabel(self.xAxis_f)
        self.xAxisVar_l.setObjectName(u"xAxisVar_l")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.xAxisVar_l.sizePolicy().hasHeightForWidth())
        self.xAxisVar_l.setSizePolicy(sizePolicy6)
        self.xAxisVar_l.setMaximumSize(QSize(16777215, 40))
        font3 = QFont()
        font3.setPointSize(12)
        font3.setStrikeOut(False)
        font3.setKerning(True)
        self.xAxisVar_l.setFont(font3)

        self.gridLayout_3.addWidget(self.xAxisVar_l, 0, 0, 1, 1)

        self.xTo_l = QLabel(self.xAxis_f)
        self.xTo_l.setObjectName(u"xTo_l")
        sizePolicy6.setHeightForWidth(self.xTo_l.sizePolicy().hasHeightForWidth())
        self.xTo_l.setSizePolicy(sizePolicy6)
        self.xTo_l.setMaximumSize(QSize(16777215, 40))
        self.xTo_l.setFont(font3)

        self.gridLayout_3.addWidget(self.xTo_l, 0, 2, 1, 1)

        self.xFrom_l = QLabel(self.xAxis_f)
        self.xFrom_l.setObjectName(u"xFrom_l")
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        sizePolicy7.setHorizontalStretch(1)
        sizePolicy7.setVerticalStretch(1)
        sizePolicy7.setHeightForWidth(self.xFrom_l.sizePolicy().hasHeightForWidth())
        self.xFrom_l.setSizePolicy(sizePolicy7)
        self.xFrom_l.setMaximumSize(QSize(16777215, 40))
        self.xFrom_l.setFont(font3)

        self.gridLayout_3.addWidget(self.xFrom_l, 0, 1, 1, 1)

        self.xScale_box = QGroupBox(self.xAxis_f)
        self.xScale_box.setObjectName(u"xScale_box")
        self.xScale_box.setMaximumSize(QSize(16777208, 16777215))
        self.xScale_box.setFont(font3)
        self.verticalLayout = QVBoxLayout(self.xScale_box)
        self.verticalLayout.setSpacing(1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(-1, 1, -1, 1)
        self.xLin_button = QRadioButton(self.xScale_box)
        self.xScale_buttons = QButtonGroup(MainWindow)
        self.xScale_buttons.setObjectName(u"xScale_buttons")
        self.xScale_buttons.addButton(self.xLin_button)
        self.xLin_button.setObjectName(u"xLin_button")
        self.xLin_button.setMaximumSize(QSize(16777215, 24))
        self.xLin_button.setChecked(True)

        self.verticalLayout.addWidget(self.xLin_button)

        self.xLog_button = QRadioButton(self.xScale_box)
        self.xScale_buttons.addButton(self.xLog_button)
        self.xLog_button.setObjectName(u"xLog_button")
        self.xLog_button.setMaximumSize(QSize(16777215, 24))

        self.verticalLayout.addWidget(self.xLog_button)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 1)

        self.gridLayout_3.addWidget(self.xScale_box, 0, 3, 2, 1)

        self.gridLayout_3.setRowStretch(0, 1)
        self.gridLayout_3.setRowStretch(1, 1)
        self.gridLayout_3.setColumnStretch(0, 1)
        self.gridLayout_3.setColumnStretch(1, 1)
        self.gridLayout_3.setColumnStretch(2, 1)
        self.gridLayout_3.setColumnStretch(3, 1)

        self.verticalLayout_5.addWidget(self.xAxis_f)

        self.yAxis_l = QLabel(self.plot3d_tab)
        self.yAxis_l.setObjectName(u"yAxis_l")
        sizePolicy8 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy8.setHorizontalStretch(1)
        sizePolicy8.setVerticalStretch(0)
        sizePolicy8.setHeightForWidth(self.yAxis_l.sizePolicy().hasHeightForWidth())
        self.yAxis_l.setSizePolicy(sizePolicy8)
        self.yAxis_l.setFont(font1)

        self.verticalLayout_5.addWidget(self.yAxis_l)

        self.yAxis_f = QFrame(self.plot3d_tab)
        self.yAxis_f.setObjectName(u"yAxis_f")
        sizePolicy9 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy9.setHorizontalStretch(1)
        sizePolicy9.setVerticalStretch(5)
        sizePolicy9.setHeightForWidth(self.yAxis_f.sizePolicy().hasHeightForWidth())
        self.yAxis_f.setSizePolicy(sizePolicy9)
        self.yAxis_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.yAxis_f.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_4 = QGridLayout(self.yAxis_f)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_4.setSizeConstraint(QLayout.SizeConstraint.SetMaximumSize)
        self.gridLayout_4.setVerticalSpacing(0)
        self.gridLayout_4.setContentsMargins(-1, 0, -1, 0)
        self.yAxisVar_l = QLabel(self.yAxis_f)
        self.yAxisVar_l.setObjectName(u"yAxisVar_l")
        self.yAxisVar_l.setFont(font3)

        self.gridLayout_4.addWidget(self.yAxisVar_l, 0, 0, 1, 1)

        self.yFrom_l = QLabel(self.yAxis_f)
        self.yFrom_l.setObjectName(u"yFrom_l")
        self.yFrom_l.setFont(font3)

        self.gridLayout_4.addWidget(self.yFrom_l, 0, 1, 1, 1)

        self.yAxisVar_dd = QComboBox(self.yAxis_f)
        self.yAxisVar_dd.addItem("")
        self.yAxisVar_dd.addItem("")
        self.yAxisVar_dd.setObjectName(u"yAxisVar_dd")
        sizePolicy7.setHeightForWidth(self.yAxisVar_dd.sizePolicy().hasHeightForWidth())
        self.yAxisVar_dd.setSizePolicy(sizePolicy7)
        self.yAxisVar_dd.setFont(font3)

        self.gridLayout_4.addWidget(self.yAxisVar_dd, 1, 0, 1, 1)

        self.ySliceLower_entry = QTextEdit(self.yAxis_f)
        self.ySliceLower_entry.setObjectName(u"ySliceLower_entry")
        self.ySliceLower_entry.setEnabled(True)
        sizePolicy5.setHeightForWidth(self.ySliceLower_entry.sizePolicy().hasHeightForWidth())
        self.ySliceLower_entry.setSizePolicy(sizePolicy5)
        self.ySliceLower_entry.setMaximumSize(QSize(16777215, 40))
        self.ySliceLower_entry.setFont(font2)
        self.ySliceLower_entry.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)

        self.gridLayout_4.addWidget(self.ySliceLower_entry, 1, 1, 1, 1)

        self.ySliceUpper_entry = QTextEdit(self.yAxis_f)
        self.ySliceUpper_entry.setObjectName(u"ySliceUpper_entry")
        sizePolicy5.setHeightForWidth(self.ySliceUpper_entry.sizePolicy().hasHeightForWidth())
        self.ySliceUpper_entry.setSizePolicy(sizePolicy5)
        self.ySliceUpper_entry.setMaximumSize(QSize(16777215, 40))
        self.ySliceUpper_entry.setFont(font2)
        self.ySliceUpper_entry.setAcceptRichText(False)

        self.gridLayout_4.addWidget(self.ySliceUpper_entry, 1, 2, 1, 1)

        self.yTo_l = QLabel(self.yAxis_f)
        self.yTo_l.setObjectName(u"yTo_l")
        self.yTo_l.setFont(font3)

        self.gridLayout_4.addWidget(self.yTo_l, 0, 2, 1, 1)

        self.yScale_box = QGroupBox(self.yAxis_f)
        self.yScale_box.setObjectName(u"yScale_box")
        sizePolicy10 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy10.setHorizontalStretch(1)
        sizePolicy10.setVerticalStretch(0)
        sizePolicy10.setHeightForWidth(self.yScale_box.sizePolicy().hasHeightForWidth())
        self.yScale_box.setSizePolicy(sizePolicy10)
        self.yScale_box.setFont(font3)
        self.verticalLayout_3 = QVBoxLayout(self.yScale_box)
        self.verticalLayout_3.setSpacing(1)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(-1, 1, -1, 1)
        self.yLin_button = QRadioButton(self.yScale_box)
        self.yScale_buttons = QButtonGroup(MainWindow)
        self.yScale_buttons.setObjectName(u"yScale_buttons")
        self.yScale_buttons.addButton(self.yLin_button)
        self.yLin_button.setObjectName(u"yLin_button")
        sizePolicy11 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy11.setHorizontalStretch(1)
        sizePolicy11.setVerticalStretch(0)
        sizePolicy11.setHeightForWidth(self.yLin_button.sizePolicy().hasHeightForWidth())
        self.yLin_button.setSizePolicy(sizePolicy11)
        self.yLin_button.setMaximumSize(QSize(16777215, 24))
        self.yLin_button.setChecked(True)

        self.verticalLayout_3.addWidget(self.yLin_button)

        self.yLog_button = QRadioButton(self.yScale_box)
        self.yScale_buttons.addButton(self.yLog_button)
        self.yLog_button.setObjectName(u"yLog_button")
        sizePolicy11.setHeightForWidth(self.yLog_button.sizePolicy().hasHeightForWidth())
        self.yLog_button.setSizePolicy(sizePolicy11)
        self.yLog_button.setMaximumSize(QSize(16777215, 24))

        self.verticalLayout_3.addWidget(self.yLog_button)

        self.verticalLayout_3.setStretch(0, 2)
        self.verticalLayout_3.setStretch(1, 1)

        self.gridLayout_4.addWidget(self.yScale_box, 0, 3, 2, 1)

        self.gridLayout_4.setRowStretch(0, 1)
        self.gridLayout_4.setColumnStretch(0, 1)
        self.gridLayout_4.setColumnStretch(1, 1)
        self.gridLayout_4.setColumnStretch(2, 1)
        self.gridLayout_4.setColumnStretch(3, 1)

        self.verticalLayout_5.addWidget(self.yAxis_f)

        self.zAxis_l = QLabel(self.plot3d_tab)
        self.zAxis_l.setObjectName(u"zAxis_l")
        self.zAxis_l.setFont(font1)

        self.verticalLayout_5.addWidget(self.zAxis_l)

        self.zAxis_f = QFrame(self.plot3d_tab)
        self.zAxis_f.setObjectName(u"zAxis_f")
        sizePolicy8.setHeightForWidth(self.zAxis_f.sizePolicy().hasHeightForWidth())
        self.zAxis_f.setSizePolicy(sizePolicy8)
        self.zAxis_f.setMaximumSize(QSize(16777215, 238))
        self.zAxis_f.setFont(font3)
        self.zAxis_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.zAxis_f.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_5 = QGridLayout(self.zAxis_f)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setVerticalSpacing(1)
        self.gridLayout_5.setContentsMargins(-1, 1, -1, 1)
        self.zSliceUpper_entry = QTextEdit(self.zAxis_f)
        self.zSliceUpper_entry.setObjectName(u"zSliceUpper_entry")
        sizePolicy5.setHeightForWidth(self.zSliceUpper_entry.sizePolicy().hasHeightForWidth())
        self.zSliceUpper_entry.setSizePolicy(sizePolicy5)
        self.zSliceUpper_entry.setMaximumSize(QSize(16777215, 40))
        self.zSliceUpper_entry.setFont(font2)
        self.zSliceUpper_entry.setAcceptRichText(False)

        self.gridLayout_5.addWidget(self.zSliceUpper_entry, 1, 2, 1, 1)

        self.zFrom_l = QLabel(self.zAxis_f)
        self.zFrom_l.setObjectName(u"zFrom_l")
        sizePolicy12 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy12.setHorizontalStretch(0)
        sizePolicy12.setVerticalStretch(1)
        sizePolicy12.setHeightForWidth(self.zFrom_l.sizePolicy().hasHeightForWidth())
        self.zFrom_l.setSizePolicy(sizePolicy12)
        self.zFrom_l.setFont(font3)

        self.gridLayout_5.addWidget(self.zFrom_l, 0, 1, 1, 1)

        self.zTo_l = QLabel(self.zAxis_f)
        self.zTo_l.setObjectName(u"zTo_l")
        sizePolicy13 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy13.setHorizontalStretch(1)
        sizePolicy13.setVerticalStretch(1)
        sizePolicy13.setHeightForWidth(self.zTo_l.sizePolicy().hasHeightForWidth())
        self.zTo_l.setSizePolicy(sizePolicy13)
        self.zTo_l.setMaximumSize(QSize(16777215, 40))
        self.zTo_l.setFont(font3)

        self.gridLayout_5.addWidget(self.zTo_l, 0, 2, 1, 1)

        self.zAxisVar_l = QLabel(self.zAxis_f)
        self.zAxisVar_l.setObjectName(u"zAxisVar_l")
        self.zAxisVar_l.setFont(font3)

        self.gridLayout_5.addWidget(self.zAxisVar_l, 0, 0, 1, 1)

        self.zScale_box = QGroupBox(self.zAxis_f)
        self.zScale_box.setObjectName(u"zScale_box")
        self.zScale_box.setFont(font3)
        self.verticalLayout_4 = QVBoxLayout(self.zScale_box)
        self.verticalLayout_4.setSpacing(1)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(8, 1, -1, 1)
        self.zLin_button = QRadioButton(self.zScale_box)
        self.zScale_buttons = QButtonGroup(MainWindow)
        self.zScale_buttons.setObjectName(u"zScale_buttons")
        self.zScale_buttons.addButton(self.zLin_button)
        self.zLin_button.setObjectName(u"zLin_button")
        self.zLin_button.setMaximumSize(QSize(16777215, 24))
        self.zLin_button.setChecked(True)

        self.verticalLayout_4.addWidget(self.zLin_button)

        self.zLog_button = QRadioButton(self.zScale_box)
        self.zScale_buttons.addButton(self.zLog_button)
        self.zLog_button.setObjectName(u"zLog_button")
        self.zLog_button.setMaximumSize(QSize(16777215, 24))

        self.verticalLayout_4.addWidget(self.zLog_button)

        self.verticalLayout_4.setStretch(0, 2)
        self.verticalLayout_4.setStretch(1, 1)

        self.gridLayout_5.addWidget(self.zScale_box, 0, 3, 2, 1)

        self.zAxisVar_dd = QComboBox(self.zAxis_f)
        self.zAxisVar_dd.addItem("")
        self.zAxisVar_dd.addItem("")
        self.zAxisVar_dd.addItem("")
        self.zAxisVar_dd.setObjectName(u"zAxisVar_dd")
        sizePolicy4.setHeightForWidth(self.zAxisVar_dd.sizePolicy().hasHeightForWidth())
        self.zAxisVar_dd.setSizePolicy(sizePolicy4)

        self.gridLayout_5.addWidget(self.zAxisVar_dd, 1, 0, 1, 1)

        self.zSliceLower_entry = QTextEdit(self.zAxis_f)
        self.zSliceLower_entry.setObjectName(u"zSliceLower_entry")
        sizePolicy5.setHeightForWidth(self.zSliceLower_entry.sizePolicy().hasHeightForWidth())
        self.zSliceLower_entry.setSizePolicy(sizePolicy5)
        self.zSliceLower_entry.setMaximumSize(QSize(16777215, 40))
        self.zSliceLower_entry.setFont(font2)

        self.gridLayout_5.addWidget(self.zSliceLower_entry, 1, 1, 1, 1)

        self.gridLayout_5.setRowStretch(0, 1)
        self.gridLayout_5.setColumnStretch(0, 1)

        self.verticalLayout_5.addWidget(self.zAxis_f)

        self.freqSelect_f = QFrame(self.plot3d_tab)
        self.freqSelect_f.setObjectName(u"freqSelect_f")
        self.freqSelect_f.setFont(font3)
        self.freqSelect_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.freqSelect_f.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.freqSelect_f)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(-1, 1, -1, 1)
        self.frequency_l = QLabel(self.freqSelect_f)
        self.frequency_l.setObjectName(u"frequency_l")
        self.frequency_l.setFont(font3)

        self.horizontalLayout_2.addWidget(self.frequency_l)

        self.frequency_slider = QSlider(self.freqSelect_f)
        self.frequency_slider.setObjectName(u"frequency_slider")
        self.frequency_slider.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_2.addWidget(self.frequency_slider)

        self.currentFrequency_l = QLCDNumber(self.freqSelect_f)
        self.currentFrequency_l.setObjectName(u"currentFrequency_l")

        self.horizontalLayout_2.addWidget(self.currentFrequency_l)


        self.verticalLayout_5.addWidget(self.freqSelect_f)

        self.plotCommands_f = QFrame(self.plot3d_tab)
        self.plotCommands_f.setObjectName(u"plotCommands_f")
        self.plotCommands_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.plotCommands_f.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.plotCommands_f)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(-1, 2, -1, 2)
        self.animatePlot_button = QPushButton(self.plotCommands_f)
        self.animatePlot_button.setObjectName(u"animatePlot_button")
        self.animatePlot_button.setFont(font3)

        self.horizontalLayout_3.addWidget(self.animatePlot_button)

        self.updatePlot_button = QPushButton(self.plotCommands_f)
        self.updatePlot_button.setObjectName(u"updatePlot_button")
        self.updatePlot_button.setFont(font3)

        self.horizontalLayout_3.addWidget(self.updatePlot_button)


        self.verticalLayout_5.addWidget(self.plotCommands_f)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_5.addItem(self.verticalSpacer_3)

        self.verticalLayout_5.setStretch(0, 1)
        self.verticalLayout_5.setStretch(1, 1)
        self.verticalLayout_5.setStretch(2, 1)
        self.verticalLayout_5.setStretch(3, 1)
        self.verticalLayout_5.setStretch(4, 1)
        self.verticalLayout_5.setStretch(5, 1)
        self.verticalLayout_5.setStretch(6, 1)
        self.plotSettingsTabs.addTab(self.plot3d_tab, "")
        self.plotSingle_tab = QWidget()
        self.plotSingle_tab.setObjectName(u"plotSingle_tab")
        sizePolicy2.setHeightForWidth(self.plotSingle_tab.sizePolicy().hasHeightForWidth())
        self.plotSingle_tab.setSizePolicy(sizePolicy2)
        self.verticalLayout_6 = QVBoxLayout(self.plotSingle_tab)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.splitter = QSplitter(self.plotSingle_tab)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Orientation.Vertical)
        self.singlePlotStyle_box = QGroupBox(self.splitter)
        self.singlePlotStyle_box.setObjectName(u"singlePlotStyle_box")
        sizePolicy14 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        sizePolicy14.setHorizontalStretch(0)
        sizePolicy14.setVerticalStretch(0)
        sizePolicy14.setHeightForWidth(self.singlePlotStyle_box.sizePolicy().hasHeightForWidth())
        self.singlePlotStyle_box.setSizePolicy(sizePolicy14)
        self.singlePlotStyle_box.setFont(font3)
        self.gridLayout_6 = QGridLayout(self.singlePlotStyle_box)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.singleSpectrogram_button = QRadioButton(self.singlePlotStyle_box)
        self.singlePlotStyle_buttons = QButtonGroup(MainWindow)
        self.singlePlotStyle_buttons.setObjectName(u"singlePlotStyle_buttons")
        self.singlePlotStyle_buttons.addButton(self.singleSpectrogram_button)
        self.singleSpectrogram_button.setObjectName(u"singleSpectrogram_button")
        self.singleSpectrogram_button.setFont(font3)

        self.gridLayout_6.addWidget(self.singleSpectrogram_button, 0, 2, 1, 1)

        self.singleTimeDomain_button = QRadioButton(self.singlePlotStyle_box)
        self.singlePlotStyle_buttons.addButton(self.singleTimeDomain_button)
        self.singleTimeDomain_button.setObjectName(u"singleTimeDomain_button")
        self.singleTimeDomain_button.setFont(font3)

        self.gridLayout_6.addWidget(self.singleTimeDomain_button, 0, 4, 1, 1)

        self.phase2d_button = QRadioButton(self.singlePlotStyle_box)
        self.singlePlotStyle_buttons.addButton(self.phase2d_button)
        self.phase2d_button.setObjectName(u"phase2d_button")

        self.gridLayout_6.addWidget(self.phase2d_button, 1, 2, 1, 1)

        self.phase3d_button = QRadioButton(self.singlePlotStyle_box)
        self.singlePlotStyle_buttons.addButton(self.phase3d_button)
        self.phase3d_button.setObjectName(u"phase3d_button")
        self.phase3d_button.setChecked(True)

        self.gridLayout_6.addWidget(self.phase3d_button, 1, 4, 1, 1)

        self.splitter.addWidget(self.singlePlotStyle_box)
        self.paramSelect_f = QFrame(self.splitter)
        self.paramSelect_f.setObjectName(u"paramSelect_f")
        sizePolicy14.setHeightForWidth(self.paramSelect_f.sizePolicy().hasHeightForWidth())
        self.paramSelect_f.setSizePolicy(sizePolicy14)
        self.paramSelect_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.paramSelect_f.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_7 = QGridLayout(self.paramSelect_f)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.param2ValSelect_dd = QComboBox(self.paramSelect_f)
        self.param2ValSelect_dd.setObjectName(u"param2ValSelect_dd")
        self.param2ValSelect_dd.setFont(font3)

        self.gridLayout_7.addWidget(self.param2ValSelect_dd, 1, 1, 1, 1)

        self.param1Select_l = QLabel(self.paramSelect_f)
        self.param1Select_l.setObjectName(u"param1Select_l")
        self.param1Select_l.setFont(font3)

        self.gridLayout_7.addWidget(self.param1Select_l, 0, 0, 1, 1)

        self.param1ValSelect_dd = QComboBox(self.paramSelect_f)
        self.param1ValSelect_dd.setObjectName(u"param1ValSelect_dd")
        self.param1ValSelect_dd.setFont(font3)

        self.gridLayout_7.addWidget(self.param1ValSelect_dd, 0, 1, 1, 1)

        self.Param2Select_l = QLabel(self.paramSelect_f)
        self.Param2Select_l.setObjectName(u"Param2Select_l")
        self.Param2Select_l.setFont(font3)

        self.gridLayout_7.addWidget(self.Param2Select_l, 1, 0, 1, 1)

        self.splitter.addWidget(self.paramSelect_f)

        self.verticalLayout_6.addWidget(self.splitter)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_6.addItem(self.verticalSpacer)

        self.plotSettingsTabs.addTab(self.plotSingle_tab, "")

        self.verticalLayout_2.addWidget(self.plotSettingsTabs, 0, Qt.AlignmentFlag.AlignTop)

        self.simSettingsTabs = QToolBox(self.controlFrame)
        self.simSettingsTabs.setObjectName(u"simSettingsTabs")
        sizePolicy.setHeightForWidth(self.simSettingsTabs.sizePolicy().hasHeightForWidth())
        self.simSettingsTabs.setSizePolicy(sizePolicy)
        font4 = QFont()
        font4.setPointSize(16)
        self.simSettingsTabs.setFont(font4)
        self.simSettingsTabs.setFrameShape(QFrame.Shape.StyledPanel)
        self.simSettings_tab = QWidget()
        self.simSettings_tab.setObjectName(u"simSettings_tab")
        self.simSettings_tab.setGeometry(QRect(0, 0, 503, 505))
        self.gridLayout_11 = QGridLayout(self.simSettings_tab)
        self.gridLayout_11.setObjectName(u"gridLayout_11")
        self.frame_2 = QFrame(self.simSettings_tab)
        self.frame_2.setObjectName(u"frame_2")
        sizePolicy13.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy13)
        self.frame_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_8 = QVBoxLayout(self.frame_2)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.splitter_5 = QSplitter(self.frame_2)
        self.splitter_5.setObjectName(u"splitter_5")
        self.splitter_5.setOrientation(Qt.Orientation.Vertical)
        self.splitter_2 = QSplitter(self.splitter_5)
        self.splitter_2.setObjectName(u"splitter_2")
        sizePolicy7.setHeightForWidth(self.splitter_2.sizePolicy().hasHeightForWidth())
        self.splitter_2.setSizePolicy(sizePolicy7)
        self.splitter_2.setMaximumSize(QSize(16777214, 337))
        self.splitter_2.setOrientation(Qt.Orientation.Vertical)
        self.p1SweepConfig_f = QFrame(self.splitter_2)
        self.p1SweepConfig_f.setObjectName(u"p1SweepConfig_f")
        sizePolicy5.setHeightForWidth(self.p1SweepConfig_f.sizePolicy().hasHeightForWidth())
        self.p1SweepConfig_f.setSizePolicy(sizePolicy5)
        self.p1SweepConfig_f.setMaximumSize(QSize(16777214, 100))
        self.p1SweepConfig_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.p1SweepConfig_f.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_8 = QGridLayout(self.p1SweepConfig_f)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.p1From_l = QLabel(self.p1SweepConfig_f)
        self.p1From_l.setObjectName(u"p1From_l")
        sizePolicy13.setHeightForWidth(self.p1From_l.sizePolicy().hasHeightForWidth())
        self.p1From_l.setSizePolicy(sizePolicy13)
        self.p1From_l.setMaximumSize(QSize(16777215, 40))
        font5 = QFont()
        font5.setPointSize(12)
        self.p1From_l.setFont(font5)

        self.gridLayout_8.addWidget(self.p1From_l, 0, 1, 1, 1)

        self.p1To_l = QLabel(self.p1SweepConfig_f)
        self.p1To_l.setObjectName(u"p1To_l")
        sizePolicy12.setHeightForWidth(self.p1To_l.sizePolicy().hasHeightForWidth())
        self.p1To_l.setSizePolicy(sizePolicy12)
        self.p1To_l.setFont(font5)

        self.gridLayout_8.addWidget(self.p1To_l, 0, 4, 1, 1)

        self.p1Var_l = QLabel(self.p1SweepConfig_f)
        self.p1Var_l.setObjectName(u"p1Var_l")
        sizePolicy13.setHeightForWidth(self.p1Var_l.sizePolicy().hasHeightForWidth())
        self.p1Var_l.setSizePolicy(sizePolicy13)
        self.p1Var_l.setFont(font5)

        self.gridLayout_8.addWidget(self.p1Var_l, 0, 0, 1, 1)

        self.p1Select_dd = QComboBox(self.p1SweepConfig_f)
        self.p1Select_dd.setObjectName(u"p1Select_dd")
        sizePolicy7.setHeightForWidth(self.p1Select_dd.sizePolicy().hasHeightForWidth())
        self.p1Select_dd.setSizePolicy(sizePolicy7)
        self.p1Select_dd.setFont(font5)

        self.gridLayout_8.addWidget(self.p1Select_dd, 1, 0, 1, 1)

        self.p1SliceLower_entry = QTextEdit(self.p1SweepConfig_f)
        self.p1SliceLower_entry.setObjectName(u"p1SliceLower_entry")
        sizePolicy5.setHeightForWidth(self.p1SliceLower_entry.sizePolicy().hasHeightForWidth())
        self.p1SliceLower_entry.setSizePolicy(sizePolicy5)
        self.p1SliceLower_entry.setMaximumSize(QSize(16777215, 40))
        font6 = QFont()
        font6.setPointSize(10)
        self.p1SliceLower_entry.setFont(font6)

        self.gridLayout_8.addWidget(self.p1SliceLower_entry, 1, 1, 1, 1)

        self.p1SliceUpper_entry = QTextEdit(self.p1SweepConfig_f)
        self.p1SliceUpper_entry.setObjectName(u"p1SliceUpper_entry")
        sizePolicy5.setHeightForWidth(self.p1SliceUpper_entry.sizePolicy().hasHeightForWidth())
        self.p1SliceUpper_entry.setSizePolicy(sizePolicy5)
        self.p1SliceUpper_entry.setMaximumSize(QSize(16777215, 40))
        self.p1SliceUpper_entry.setFont(font6)
        self.p1SliceUpper_entry.setAcceptRichText(False)

        self.gridLayout_8.addWidget(self.p1SliceUpper_entry, 1, 4, 1, 1)

        self.p1SweepScale_box = QGroupBox(self.p1SweepConfig_f)
        self.p1SweepScale_box.setObjectName(u"p1SweepScale_box")
        self.p1SweepScale_box.setFont(font5)
        self.verticalLayout_7 = QVBoxLayout(self.p1SweepScale_box)
        self.verticalLayout_7.setSpacing(1)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(-1, 1, -1, 1)
        self.p1Lin_button = QRadioButton(self.p1SweepScale_box)
        self.param1SweepScale_buttons = QButtonGroup(MainWindow)
        self.param1SweepScale_buttons.setObjectName(u"param1SweepScale_buttons")
        self.param1SweepScale_buttons.addButton(self.p1Lin_button)
        self.p1Lin_button.setObjectName(u"p1Lin_button")
        self.p1Lin_button.setFont(font5)
        self.p1Lin_button.setChecked(True)

        self.verticalLayout_7.addWidget(self.p1Lin_button)

        self.p1Log_button = QRadioButton(self.p1SweepScale_box)
        self.param1SweepScale_buttons.addButton(self.p1Log_button)
        self.p1Log_button.setObjectName(u"p1Log_button")
        self.p1Log_button.setFont(font5)

        self.verticalLayout_7.addWidget(self.p1Log_button)

        self.verticalLayout_7.setStretch(0, 2)
        self.verticalLayout_7.setStretch(1, 1)

        self.gridLayout_8.addWidget(self.p1SweepScale_box, 0, 5, 2, 1)

        self.gridLayout_8.setRowStretch(0, 1)
        self.gridLayout_8.setRowStretch(1, 1)
        self.gridLayout_8.setColumnStretch(0, 1)
        self.gridLayout_8.setColumnStretch(1, 1)
        self.gridLayout_8.setColumnStretch(4, 1)
        self.gridLayout_8.setColumnStretch(5, 1)
        self.gridLayout_8.setRowMinimumHeight(0, 10)
        self.gridLayout_8.setRowMinimumHeight(1, 10)
        self.splitter_2.addWidget(self.p1SweepConfig_f)
        self.p2SweepConfig_f = QFrame(self.splitter_2)
        self.p2SweepConfig_f.setObjectName(u"p2SweepConfig_f")
        sizePolicy15 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        sizePolicy15.setHorizontalStretch(1)
        sizePolicy15.setVerticalStretch(1)
        sizePolicy15.setHeightForWidth(self.p2SweepConfig_f.sizePolicy().hasHeightForWidth())
        self.p2SweepConfig_f.setSizePolicy(sizePolicy15)
        self.p2SweepConfig_f.setMaximumSize(QSize(16777215, 232))
        self.p2SweepConfig_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.p2SweepConfig_f.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_10 = QGridLayout(self.p2SweepConfig_f)
        self.gridLayout_10.setObjectName(u"gridLayout_10")
        self.p2SliceUpper_entry_4 = QTextEdit(self.p2SweepConfig_f)
        self.p2SliceUpper_entry_4.setObjectName(u"p2SliceUpper_entry_4")
        sizePolicy5.setHeightForWidth(self.p2SliceUpper_entry_4.sizePolicy().hasHeightForWidth())
        self.p2SliceUpper_entry_4.setSizePolicy(sizePolicy5)
        self.p2SliceUpper_entry_4.setMaximumSize(QSize(16777215, 40))
        self.p2SliceUpper_entry_4.setFont(font6)
        self.p2SliceUpper_entry_4.setAcceptRichText(False)

        self.gridLayout_10.addWidget(self.p2SliceUpper_entry_4, 1, 4, 1, 1)

        self.p2SliceLower_entry_4 = QTextEdit(self.p2SweepConfig_f)
        self.p2SliceLower_entry_4.setObjectName(u"p2SliceLower_entry_4")
        sizePolicy5.setHeightForWidth(self.p2SliceLower_entry_4.sizePolicy().hasHeightForWidth())
        self.p2SliceLower_entry_4.setSizePolicy(sizePolicy5)
        self.p2SliceLower_entry_4.setMaximumSize(QSize(16777215, 40))
        self.p2SliceLower_entry_4.setFont(font6)

        self.gridLayout_10.addWidget(self.p2SliceLower_entry_4, 1, 1, 1, 1)

        self.p2From_l = QLabel(self.p2SweepConfig_f)
        self.p2From_l.setObjectName(u"p2From_l")
        sizePolicy13.setHeightForWidth(self.p2From_l.sizePolicy().hasHeightForWidth())
        self.p2From_l.setSizePolicy(sizePolicy13)
        self.p2From_l.setMaximumSize(QSize(16777215, 40))
        self.p2From_l.setFont(font5)

        self.gridLayout_10.addWidget(self.p2From_l, 0, 1, 1, 1)

        self.p2Var_l = QLabel(self.p2SweepConfig_f)
        self.p2Var_l.setObjectName(u"p2Var_l")
        self.p2Var_l.setMaximumSize(QSize(16777215, 40))
        self.p2Var_l.setFont(font5)

        self.gridLayout_10.addWidget(self.p2Var_l, 0, 0, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_10.addItem(self.verticalSpacer_2, 3, 1, 1, 1)

        self.p2Select_dd = QComboBox(self.p2SweepConfig_f)
        self.p2Select_dd.setObjectName(u"p2Select_dd")
        sizePolicy7.setHeightForWidth(self.p2Select_dd.sizePolicy().hasHeightForWidth())
        self.p2Select_dd.setSizePolicy(sizePolicy7)
        self.p2Select_dd.setFont(font5)

        self.gridLayout_10.addWidget(self.p2Select_dd, 1, 0, 1, 1)

        self.p2To_l_4 = QLabel(self.p2SweepConfig_f)
        self.p2To_l_4.setObjectName(u"p2To_l_4")
        sizePolicy12.setHeightForWidth(self.p2To_l_4.sizePolicy().hasHeightForWidth())
        self.p2To_l_4.setSizePolicy(sizePolicy12)
        self.p2To_l_4.setFont(font5)

        self.gridLayout_10.addWidget(self.p2To_l_4, 0, 4, 1, 1)

        self.p2SweepScale_box = QGroupBox(self.p2SweepConfig_f)
        self.p2SweepScale_box.setObjectName(u"p2SweepScale_box")
        sizePolicy6.setHeightForWidth(self.p2SweepScale_box.sizePolicy().hasHeightForWidth())
        self.p2SweepScale_box.setSizePolicy(sizePolicy6)
        self.p2SweepScale_box.setFont(font5)
        self.verticalLayout_9 = QVBoxLayout(self.p2SweepScale_box)
        self.verticalLayout_9.setSpacing(1)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(-1, 1, -1, 1)
        self.p2Lin_button = QRadioButton(self.p2SweepScale_box)
        self.param2SweepScale_buttons = QButtonGroup(MainWindow)
        self.param2SweepScale_buttons.setObjectName(u"param2SweepScale_buttons")
        self.param2SweepScale_buttons.addButton(self.p2Lin_button)
        self.p2Lin_button.setObjectName(u"p2Lin_button")
        self.p2Lin_button.setFont(font5)
        self.p2Lin_button.setChecked(True)

        self.verticalLayout_9.addWidget(self.p2Lin_button)

        self.p2Log_button = QRadioButton(self.p2SweepScale_box)
        self.param2SweepScale_buttons.addButton(self.p2Log_button)
        self.p2Log_button.setObjectName(u"p2Log_button")
        self.p2Log_button.setFont(font5)

        self.verticalLayout_9.addWidget(self.p2Log_button)

        self.verticalLayout_9.setStretch(0, 2)
        self.verticalLayout_9.setStretch(1, 1)

        self.gridLayout_10.addWidget(self.p2SweepScale_box, 0, 5, 2, 1)

        self.gridLayout_10.setColumnStretch(0, 1)
        self.splitter_2.addWidget(self.p2SweepConfig_f)
        self.splitter_5.addWidget(self.splitter_2)
        self.splitter_4 = QSplitter(self.splitter_5)
        self.splitter_4.setObjectName(u"splitter_4")
        self.splitter_4.setOrientation(Qt.Orientation.Horizontal)
        self.simTime_f = QFrame(self.splitter_4)
        self.simTime_f.setObjectName(u"simTime_f")
        sizePolicy6.setHeightForWidth(self.simTime_f.sizePolicy().hasHeightForWidth())
        self.simTime_f.setSizePolicy(sizePolicy6)
        self.simTime_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.simTime_f.setFrameShadow(QFrame.Shadow.Raised)
        self.formLayout = QFormLayout(self.simTime_f)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setSizeConstraint(QLayout.SizeConstraint.SetMaximumSize)
        self.formLayout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.formLayout.setVerticalSpacing(2)
        self.formLayout.setContentsMargins(-1, -1, -1, 3)
        self.label = QLabel(self.simTime_f)
        self.label.setObjectName(u"label")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label)

        self.label_2 = QLabel(self.simTime_f)
        self.label_2.setObjectName(u"label_2")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_2)

        self.plainTextEdit_3 = QPlainTextEdit(self.simTime_f)
        self.plainTextEdit_3.setObjectName(u"plainTextEdit_3")
        sizePolicy15.setHeightForWidth(self.plainTextEdit_3.sizePolicy().hasHeightForWidth())
        self.plainTextEdit_3.setSizePolicy(sizePolicy15)
        self.plainTextEdit_3.setMaximumSize(QSize(16777215, 40))
        self.plainTextEdit_3.setFont(font6)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.plainTextEdit_3)

        self.label_3 = QLabel(self.simTime_f)
        self.label_3.setObjectName(u"label_3")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.label_3)

        self.plainTextEdit_2 = QPlainTextEdit(self.simTime_f)
        self.plainTextEdit_2.setObjectName(u"plainTextEdit_2")
        sizePolicy16 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy16.setHorizontalStretch(0)
        sizePolicy16.setVerticalStretch(0)
        sizePolicy16.setHeightForWidth(self.plainTextEdit_2.sizePolicy().hasHeightForWidth())
        self.plainTextEdit_2.setSizePolicy(sizePolicy16)
        self.plainTextEdit_2.setMaximumSize(QSize(16777215, 40))
        self.plainTextEdit_2.setFont(font6)

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.plainTextEdit_2)

        self.label_4 = QLabel(self.simTime_f)
        self.label_4.setObjectName(u"label_4")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.label_4)

        self.plainTextEdit_4 = QPlainTextEdit(self.simTime_f)
        self.plainTextEdit_4.setObjectName(u"plainTextEdit_4")
        sizePolicy5.setHeightForWidth(self.plainTextEdit_4.sizePolicy().hasHeightForWidth())
        self.plainTextEdit_4.setSizePolicy(sizePolicy5)
        self.plainTextEdit_4.setMaximumSize(QSize(16777215, 40))
        self.plainTextEdit_4.setFont(font6)

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.plainTextEdit_4)

        self.plainTextEdit = QPlainTextEdit(self.simTime_f)
        self.plainTextEdit.setObjectName(u"plainTextEdit")
        sizePolicy5.setHeightForWidth(self.plainTextEdit.sizePolicy().hasHeightForWidth())
        self.plainTextEdit.setSizePolicy(sizePolicy5)
        self.plainTextEdit.setMaximumSize(QSize(16777215, 40))
        self.plainTextEdit.setSizeIncrement(QSize(10, 10))
        self.plainTextEdit.setBaseSize(QSize(0, 60))
        font7 = QFont()
        font7.setFamilies([u"Segoe UI"])
        font7.setPointSize(10)
        font7.setStyleStrategy(QFont.NoAntialias)
        self.plainTextEdit.setFont(font7)

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.plainTextEdit)

        self.splitter_4.addWidget(self.simTime_f)
        self.inits_box = QGroupBox(self.splitter_4)
        self.inits_box.setObjectName(u"inits_box")
        self.gridLayout_9 = QGridLayout(self.inits_box)
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.splitter_3 = QSplitter(self.inits_box)
        self.splitter_3.setObjectName(u"splitter_3")
        self.splitter_3.setOrientation(Qt.Orientation.Horizontal)
        self.init_0 = QLabel(self.splitter_3)
        self.init_0.setObjectName(u"init_0")
        self.splitter_3.addWidget(self.init_0)
        self.init0_e = QTextEdit(self.splitter_3)
        self.init0_e.setObjectName(u"init0_e")
        self.splitter_3.addWidget(self.init0_e)

        self.gridLayout_9.addWidget(self.splitter_3, 0, 0, 1, 1)

        self.splitter_4.addWidget(self.inits_box)
        self.splitter_5.addWidget(self.splitter_4)

        self.verticalLayout_8.addWidget(self.splitter_5)


        self.gridLayout_11.addWidget(self.frame_2, 0, 0, 1, 2)

        self.simSettingsTabs.addItem(self.simSettings_tab, u"Simulation settings")
        self.sysParams_tab = QWidget()
        self.sysParams_tab.setObjectName(u"sysParams_tab")
        self.sysParams_tab.setGeometry(QRect(0, 0, 503, 505))
        sizePolicy13.setHeightForWidth(self.sysParams_tab.sizePolicy().hasHeightForWidth())
        self.sysParams_tab.setSizePolicy(sizePolicy13)
        self.verticalLayout_11 = QVBoxLayout(self.sysParams_tab)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.label_5 = QLabel(self.sysParams_tab)
        self.label_5.setObjectName(u"label_5")
        sizePolicy17 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy17.setHorizontalStretch(0)
        sizePolicy17.setVerticalStretch(0)
        sizePolicy17.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy17)
        font8 = QFont()
        font8.setPointSize(16)
        font8.setItalic(True)
        self.label_5.setFont(font8)

        self.verticalLayout_11.addWidget(self.label_5)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_11.addItem(self.verticalSpacer_4)

        self.simSettingsTabs.addItem(self.sysParams_tab, u"System Parameters")

        self.verticalLayout_2.addWidget(self.simSettingsTabs)

        self.saveOrSolve_f = QFrame(self.controlFrame)
        self.saveOrSolve_f.setObjectName(u"saveOrSolve_f")
        self.saveOrSolve_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.saveOrSolve_f.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout = QHBoxLayout(self.saveOrSolve_f)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.save_button = QPushButton(self.saveOrSolve_f)
        self.save_button.setObjectName(u"save_button")
        sizePolicy.setHeightForWidth(self.save_button.sizePolicy().hasHeightForWidth())
        self.save_button.setSizePolicy(sizePolicy)
        font9 = QFont()
        font9.setPointSize(24)
        self.save_button.setFont(font9)

        self.horizontalLayout.addWidget(self.save_button)

        self.solve_button = QPushButton(self.saveOrSolve_f)
        self.solve_button.setObjectName(u"solve_button")
        sizePolicy.setHeightForWidth(self.solve_button.sizePolicy().hasHeightForWidth())
        self.solve_button.setSizePolicy(sizePolicy)
        self.solve_button.setFont(font9)

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
        self.menubar.setGeometry(QRect(0, 0, 1600, 22))
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
        self.updatePlot_button.released.connect(MainWindow.update_plot)
        self.zAxisVar_dd.currentTextChanged.connect(MainWindow.update_z_var)
        self.yAxisVar_dd.currentTextChanged.connect(MainWindow.update_y_var)
        self.xAxisVar_dd.currentTextChanged.connect(MainWindow.update_x_var)
        self.xSliceLower_entry.textChanged.connect(MainWindow.update_x_slice)
        self.xSliceUpper_entry.textChanged.connect(MainWindow.update_x_slice)
        self.ySliceLower_entry.textChanged.connect(MainWindow.update_y_slice)
        self.ySliceUpper_entry.textChanged.connect(MainWindow.update_y_slice)
        self.zSliceLower_entry.textChanged.connect(MainWindow.update_z_slice)
        self.zSliceUpper_entry.textChanged.connect(MainWindow.update_z_slice)
        self.animatePlot_button.objectNameChanged.connect(MainWindow.animate_3D)
        self.frequency_slider.valueChanged.connect(self.currentFrequency_l.display)
        self.frequency_slider.valueChanged.connect(MainWindow.set_current_fft_freq)
        self.xScale_buttons.idClicked.connect(MainWindow.set_singlePlot_style)
        self.zScale_buttons.idClicked.connect(MainWindow.set_zScale)
        self.yScale_buttons.idClicked.connect(MainWindow.set_yScale)
        self.xScale_buttons.idClicked.connect(MainWindow.set_xScale)
        self.p1Select_dd.currentTextChanged.connect(MainWindow.set_param1_var)
        self.p2Select_dd.currentTextChanged.connect(MainWindow.set_param2_var)
        self.p1SliceLower_entry.textChanged.connect(MainWindow.set_param1Sweep_bounds)
        self.p1SliceUpper_entry.textChanged.connect(MainWindow.set_param1Sweep_bounds)
        self.p2SliceLower_entry_4.textChanged.connect(MainWindow.set_param2Sweep_bounds)
        self.p2SliceUpper_entry_4.textChanged.connect(MainWindow.set_param2Sweep_bounds)
        self.param1SweepScale_buttons.idClicked.connect(MainWindow.set_param1Sweep_scale)
        self.param2SweepScale_buttons.idClicked.connect(MainWindow.set_param2Sweep_scale)
        self.plainTextEdit.textChanged.connect(MainWindow.set_duration)
        self.plainTextEdit_3.textChanged.connect(MainWindow.set_warmup)
        self.plainTextEdit_2.textChanged.connect(MainWindow.set_fs)
        self.plainTextEdit_4.textChanged.connect(MainWindow.set_dt)
        self.init0_e.textChanged.connect(MainWindow.set_y0)
        self.save_button.clicked.connect(MainWindow.save_results)
        self.solve_button.released.connect(MainWindow.solve_ODE)
        self.param1ValSelect_dd.currentTextChanged.connect(MainWindow.select_param1)
        self.param2ValSelect_dd.currentTextChanged.connect(MainWindow.select_param2)
        self.actionLoad_System_File.triggered.connect(MainWindow.load_system_from_filedialog)

        self.plotSettingsTabs.setCurrentIndex(0)
        self.simSettingsTabs.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionExit.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.actionLoad_System_File.setText(QCoreApplication.translate("MainWindow", u"Load System File", None))
        self.action64_bit.setText(QCoreApplication.translate("MainWindow", u"64-bit", None))
        self.action64_bit.setIconText(QCoreApplication.translate("MainWindow", u"64-bit", None))
#if QT_CONFIG(tooltip)
        self.action64_bit.setToolTip(QCoreApplication.translate("MainWindow", u"64-bit floating-point values", None))
#endif // QT_CONFIG(tooltip)
        self.action32_bit.setText(QCoreApplication.translate("MainWindow", u"32-bit", None))
        self.action64_bit_2.setText(QCoreApplication.translate("MainWindow", u"64-bit", None))
        self.action32_bit_2.setText(QCoreApplication.translate("MainWindow", u"32-bit", None))
        self.xAxis_l.setText(QCoreApplication.translate("MainWindow", u"x axis", None))
        self.xAxisVar_dd.setItemText(0, QCoreApplication.translate("MainWindow", u"Param 1", None))
        self.xAxisVar_dd.setItemText(1, QCoreApplication.translate("MainWindow", u"Param 2", None))
        self.xAxisVar_dd.setItemText(2, QCoreApplication.translate("MainWindow", u"Time", None))
        self.xAxisVar_dd.setItemText(3, QCoreApplication.translate("MainWindow", u"Frequency", None))

        self.xAxisVar_l.setText(QCoreApplication.translate("MainWindow", u"Variable", None))
        self.xTo_l.setText(QCoreApplication.translate("MainWindow", u"to", None))
        self.xFrom_l.setText(QCoreApplication.translate("MainWindow", u"from", None))
        self.xScale_box.setTitle(QCoreApplication.translate("MainWindow", u"Scale", None))
        self.xLin_button.setText(QCoreApplication.translate("MainWindow", u"Linear", None))
        self.xLog_button.setText(QCoreApplication.translate("MainWindow", u"Logarithmic", None))
        self.yAxis_l.setText(QCoreApplication.translate("MainWindow", u"y axis", None))
        self.yAxisVar_l.setText(QCoreApplication.translate("MainWindow", u"Variable", None))
        self.yFrom_l.setText(QCoreApplication.translate("MainWindow", u"from", None))
        self.yAxisVar_dd.setItemText(0, QCoreApplication.translate("MainWindow", u"Param 2", None))
        self.yAxisVar_dd.setItemText(1, QCoreApplication.translate("MainWindow", u"Param 1", None))

        self.yTo_l.setText(QCoreApplication.translate("MainWindow", u"to", None))
        self.yScale_box.setTitle(QCoreApplication.translate("MainWindow", u"Scale", None))
        self.yLin_button.setText(QCoreApplication.translate("MainWindow", u"Linear", None))
        self.yLog_button.setText(QCoreApplication.translate("MainWindow", u"Logarithmic", None))
        self.zAxis_l.setText(QCoreApplication.translate("MainWindow", u"z axis", None))
        self.zSliceUpper_entry.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0.0</p></body></html>", None))
        self.zFrom_l.setText(QCoreApplication.translate("MainWindow", u"from", None))
        self.zTo_l.setText(QCoreApplication.translate("MainWindow", u"to", None))
        self.zAxisVar_l.setText(QCoreApplication.translate("MainWindow", u"Variable", None))
        self.zScale_box.setTitle(QCoreApplication.translate("MainWindow", u"Scale", None))
        self.zLin_button.setText(QCoreApplication.translate("MainWindow", u"Linear", None))
        self.zLog_button.setText(QCoreApplication.translate("MainWindow", u"Logarithmic", None))
        self.zAxisVar_dd.setItemText(0, QCoreApplication.translate("MainWindow", u"PSD Magnitude @f", None))
        self.zAxisVar_dd.setItemText(1, QCoreApplication.translate("MainWindow", u"FFT Phase @f", None))
        self.zAxisVar_dd.setItemText(2, QCoreApplication.translate("MainWindow", u"Amplitude (RMS/current)", None))

        self.frequency_l.setText(QCoreApplication.translate("MainWindow", u"Frequency", None))
        self.animatePlot_button.setText(QCoreApplication.translate("MainWindow", u"Animate free variable", None))
        self.updatePlot_button.setText(QCoreApplication.translate("MainWindow", u"Update Plot", None))
        self.plotSettingsTabs.setTabText(self.plotSettingsTabs.indexOf(self.plot3d_tab), QCoreApplication.translate("MainWindow", u"3D Plot", None))
#if QT_CONFIG(tooltip)
        self.plotSettingsTabs.setTabToolTip(self.plotSettingsTabs.indexOf(self.plot3d_tab), QCoreApplication.translate("MainWindow", u"Plot a three-dimensional surface of selected results", None))
#endif // QT_CONFIG(tooltip)
        self.singlePlotStyle_box.setTitle(QCoreApplication.translate("MainWindow", u"Plot Style", None))
        self.singleSpectrogram_button.setText(QCoreApplication.translate("MainWindow", u"Spectrogram", None))
        self.singleTimeDomain_button.setText(QCoreApplication.translate("MainWindow", u"Time-domain", None))
        self.phase2d_button.setText(QCoreApplication.translate("MainWindow", u"2D Phase Diagram", None))
        self.phase3d_button.setText(QCoreApplication.translate("MainWindow", u"3D Phase Diagram", None))
        self.param1Select_l.setText(QCoreApplication.translate("MainWindow", u"Parameter 1:", None))
        self.Param2Select_l.setText(QCoreApplication.translate("MainWindow", u"Parameter 2:", None))
        self.plotSettingsTabs.setTabText(self.plotSettingsTabs.indexOf(self.plotSingle_tab), QCoreApplication.translate("MainWindow", u"Single Dataset", None))
#if QT_CONFIG(tooltip)
        self.plotSettingsTabs.setTabToolTip(self.plotSettingsTabs.indexOf(self.plotSingle_tab), QCoreApplication.translate("MainWindow", u"Plot a single dataset in two dimensions", None))
#endif // QT_CONFIG(tooltip)
        self.p1From_l.setText(QCoreApplication.translate("MainWindow", u"from", None))
        self.p1To_l.setText(QCoreApplication.translate("MainWindow", u"to", None))
        self.p1Var_l.setText(QCoreApplication.translate("MainWindow", u"Variable", None))
        self.p1SliceLower_entry.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0.0</p></body></html>", None))
        self.p1SliceUpper_entry.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1.0</p></body></html>", None))
        self.p1SweepScale_box.setTitle(QCoreApplication.translate("MainWindow", u"Scale", None))
        self.p1Lin_button.setText(QCoreApplication.translate("MainWindow", u"Linear", None))
        self.p1Log_button.setText(QCoreApplication.translate("MainWindow", u"Logarithmic", None))
        self.p2SliceUpper_entry_4.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1.0</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>", None))
        self.p2SliceLower_entry_4.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0.0</p></body></html>", None))
        self.p2From_l.setText(QCoreApplication.translate("MainWindow", u"from", None))
        self.p2Var_l.setText(QCoreApplication.translate("MainWindow", u"Variable", None))
        self.p2To_l_4.setText(QCoreApplication.translate("MainWindow", u"to", None))
        self.p2SweepScale_box.setTitle(QCoreApplication.translate("MainWindow", u"Scale", None))
        self.p2Lin_button.setText(QCoreApplication.translate("MainWindow", u"Linear", None))
        self.p2Log_button.setText(QCoreApplication.translate("MainWindow", u"Logarithmic", None))
#if QT_CONFIG(tooltip)
        self.label.setToolTip(QCoreApplication.translate("MainWindow", u"How long to record for", None))
#endif // QT_CONFIG(tooltip)
        self.label.setText(QCoreApplication.translate("MainWindow", u"Duration", None))
#if QT_CONFIG(tooltip)
        self.label_2.setToolTip(QCoreApplication.translate("MainWindow", u"How long to wait before you start recording (to reach steady state)", None))
#endif // QT_CONFIG(tooltip)
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Warmup", None))
        self.plainTextEdit_3.setPlainText(QCoreApplication.translate("MainWindow", u"100.0\n"
"", None))
#if QT_CONFIG(tooltip)
        self.label_3.setToolTip(QCoreApplication.translate("MainWindow", u"Sample rate - how frequently you need to record samples. Max response freq * 2.5 is pretty safe.", None))
#endif // QT_CONFIG(tooltip)
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"fs", None))
        self.plainTextEdit_2.setPlainText(QCoreApplication.translate("MainWindow", u"1.0", None))
#if QT_CONFIG(tooltip)
        self.label_4.setToolTip(QCoreApplication.translate("MainWindow", u"(for fixed step solvers) The integration step size", None))
#endif // QT_CONFIG(tooltip)
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Step Size", None))
        self.plainTextEdit_4.setPlainText(QCoreApplication.translate("MainWindow", u"0.001", None))
        self.plainTextEdit.setPlainText(QCoreApplication.translate("MainWindow", u"100.0\n"
"", None))
        self.inits_box.setTitle(QCoreApplication.translate("MainWindow", u"y0", None))
        self.init_0.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.simSettingsTabs.setItemText(self.simSettingsTabs.indexOf(self.simSettings_tab), QCoreApplication.translate("MainWindow", u"Simulation settings", None))
#if QT_CONFIG(tooltip)
        self.simSettingsTabs.setItemToolTip(self.simSettingsTabs.indexOf(self.simSettings_tab), QCoreApplication.translate("MainWindow", u"Modify per-simulation settings", None))
#endif // QT_CONFIG(tooltip)
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"No System Loaded", None))
        self.simSettingsTabs.setItemText(self.simSettingsTabs.indexOf(self.sysParams_tab), QCoreApplication.translate("MainWindow", u"System Parameters", None))
#if QT_CONFIG(tooltip)
        self.simSettingsTabs.setItemToolTip(self.simSettingsTabs.indexOf(self.sysParams_tab), QCoreApplication.translate("MainWindow", u"Modify system constants and settings", None))
#endif // QT_CONFIG(tooltip)
        self.save_button.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.solve_button.setText(QCoreApplication.translate("MainWindow", u"Solve", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuPrecision.setTitle(QCoreApplication.translate("MainWindow", u"Precision", None))
    # retranslateUi
