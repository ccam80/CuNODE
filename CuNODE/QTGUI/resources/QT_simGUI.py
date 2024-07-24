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
from qtpy.QtWidgets import (QApplication, QButtonGroup, QComboBox, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLCDNumber, QLabel, QLayout, QMainWindow,
    QMenu, QMenuBar, QPlainTextEdit, QPushButton,
    QRadioButton, QSizePolicy, QSlider, QSpacerItem,
    QSplitter, QStatusBar, QTabWidget, QTextEdit,
    QToolBox, QVBoxLayout, QWidget)

from QT_designer_source.custom_widgets.variable_from_to_scale_widget import variable_from_to_scale_widget
from QT_designer_source.pyVistaView import pyVistaView

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1592, 1345)
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
        self.verticalLayout = QVBoxLayout(self.plot3d_tab)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.xAxis_l = QLabel(self.plot3d_tab)
        self.xAxis_l.setObjectName(u"xAxis_l")
        font1 = QFont()
        font1.setPointSize(14)
        font1.setBold(True)
        font1.setStrikeOut(False)
        font1.setKerning(True)
        self.xAxis_l.setFont(font1)

        self.verticalLayout.addWidget(self.xAxis_l)

        self.xAxisGridPlotOtions = variable_from_to_scale_widget(self.plot3d_tab)
        self.xAxisGridPlotOtions.setObjectName(u"xAxisGridPlotOtions")
        self.xAxisGridPlotOtions.setFrameShape(QFrame.Shape.StyledPanel)
        self.xAxisGridPlotOtions.setFrameShadow(QFrame.Shadow.Raised)

        self.verticalLayout.addWidget(self.xAxisGridPlotOtions)

        self.yAxis_l = QLabel(self.plot3d_tab)
        self.yAxis_l.setObjectName(u"yAxis_l")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy3.setHorizontalStretch(1)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.yAxis_l.sizePolicy().hasHeightForWidth())
        self.yAxis_l.setSizePolicy(sizePolicy3)
        self.yAxis_l.setFont(font1)

        self.verticalLayout.addWidget(self.yAxis_l)

        self.xAxisGridPlotOtions_2 = variable_from_to_scale_widget(self.plot3d_tab)
        self.xAxisGridPlotOtions_2.setObjectName(u"xAxisGridPlotOtions_2")
        self.xAxisGridPlotOtions_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.xAxisGridPlotOtions_2.setFrameShadow(QFrame.Shadow.Raised)

        self.verticalLayout.addWidget(self.xAxisGridPlotOtions_2)

        self.zAxis_l = QLabel(self.plot3d_tab)
        self.zAxis_l.setObjectName(u"zAxis_l")
        self.zAxis_l.setFont(font1)

        self.verticalLayout.addWidget(self.zAxis_l)

        self.xAxisGridPlotOtions_3 = variable_from_to_scale_widget(self.plot3d_tab)
        self.xAxisGridPlotOtions_3.setObjectName(u"xAxisGridPlotOtions_3")
        self.xAxisGridPlotOtions_3.setFrameShape(QFrame.Shape.StyledPanel)
        self.xAxisGridPlotOtions_3.setFrameShadow(QFrame.Shadow.Raised)

        self.verticalLayout.addWidget(self.xAxisGridPlotOtions_3)

        self.freqSelect_f = QFrame(self.plot3d_tab)
        self.freqSelect_f.setObjectName(u"freqSelect_f")
        font2 = QFont()
        font2.setPointSize(12)
        font2.setStrikeOut(False)
        font2.setKerning(True)
        self.freqSelect_f.setFont(font2)
        self.freqSelect_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.freqSelect_f.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.freqSelect_f)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(-1, 1, -1, 1)
        self.frequency_l = QLabel(self.freqSelect_f)
        self.frequency_l.setObjectName(u"frequency_l")
        self.frequency_l.setFont(font2)

        self.horizontalLayout_2.addWidget(self.frequency_l)

        self.frequency_slider = QSlider(self.freqSelect_f)
        self.frequency_slider.setObjectName(u"frequency_slider")
        self.frequency_slider.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_2.addWidget(self.frequency_slider)

        self.currentFrequency_l = QLCDNumber(self.freqSelect_f)
        self.currentFrequency_l.setObjectName(u"currentFrequency_l")

        self.horizontalLayout_2.addWidget(self.currentFrequency_l)


        self.verticalLayout.addWidget(self.freqSelect_f)

        self.plotCommands_f = QFrame(self.plot3d_tab)
        self.plotCommands_f.setObjectName(u"plotCommands_f")
        self.plotCommands_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.plotCommands_f.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.plotCommands_f)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(-1, 2, -1, 2)
        self.animatePlot_button = QPushButton(self.plotCommands_f)
        self.animatePlot_button.setObjectName(u"animatePlot_button")
        self.animatePlot_button.setFont(font2)

        self.horizontalLayout_3.addWidget(self.animatePlot_button)

        self.updatePlot_button = QPushButton(self.plotCommands_f)
        self.updatePlot_button.setObjectName(u"updatePlot_button")
        self.updatePlot_button.setFont(font2)

        self.horizontalLayout_3.addWidget(self.updatePlot_button)


        self.verticalLayout.addWidget(self.plotCommands_f)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_3)

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
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.singlePlotStyle_box.sizePolicy().hasHeightForWidth())
        self.singlePlotStyle_box.setSizePolicy(sizePolicy4)
        self.singlePlotStyle_box.setFont(font2)
        self.gridLayout_6 = QGridLayout(self.singlePlotStyle_box)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.singleSpectrogram_button = QRadioButton(self.singlePlotStyle_box)
        self.singlePlotStyle_buttons = QButtonGroup(MainWindow)
        self.singlePlotStyle_buttons.setObjectName(u"singlePlotStyle_buttons")
        self.singlePlotStyle_buttons.addButton(self.singleSpectrogram_button)
        self.singleSpectrogram_button.setObjectName(u"singleSpectrogram_button")
        self.singleSpectrogram_button.setFont(font2)

        self.gridLayout_6.addWidget(self.singleSpectrogram_button, 0, 2, 1, 1)

        self.singleTimeDomain_button = QRadioButton(self.singlePlotStyle_box)
        self.singlePlotStyle_buttons.addButton(self.singleTimeDomain_button)
        self.singleTimeDomain_button.setObjectName(u"singleTimeDomain_button")
        self.singleTimeDomain_button.setFont(font2)

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
        sizePolicy4.setHeightForWidth(self.paramSelect_f.sizePolicy().hasHeightForWidth())
        self.paramSelect_f.setSizePolicy(sizePolicy4)
        self.paramSelect_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.paramSelect_f.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_7 = QGridLayout(self.paramSelect_f)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.param2ValSelect_dd = QComboBox(self.paramSelect_f)
        self.param2ValSelect_dd.setObjectName(u"param2ValSelect_dd")
        self.param2ValSelect_dd.setFont(font2)

        self.gridLayout_7.addWidget(self.param2ValSelect_dd, 1, 1, 1, 1)

        self.param1Select_l = QLabel(self.paramSelect_f)
        self.param1Select_l.setObjectName(u"param1Select_l")
        self.param1Select_l.setFont(font2)

        self.gridLayout_7.addWidget(self.param1Select_l, 0, 0, 1, 1)

        self.param1ValSelect_dd = QComboBox(self.paramSelect_f)
        self.param1ValSelect_dd.setObjectName(u"param1ValSelect_dd")
        self.param1ValSelect_dd.setFont(font2)

        self.gridLayout_7.addWidget(self.param1ValSelect_dd, 0, 1, 1, 1)

        self.Param2Select_l = QLabel(self.paramSelect_f)
        self.Param2Select_l.setObjectName(u"Param2Select_l")
        self.Param2Select_l.setFont(font2)

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
        font3 = QFont()
        font3.setPointSize(16)
        self.simSettingsTabs.setFont(font3)
        self.simSettingsTabs.setFrameShape(QFrame.Shape.StyledPanel)
        self.simSettings_tab = QWidget()
        self.simSettings_tab.setObjectName(u"simSettings_tab")
        self.simSettings_tab.setGeometry(QRect(0, 0, 501, 496))
        self.gridLayout_11 = QGridLayout(self.simSettings_tab)
        self.gridLayout_11.setObjectName(u"gridLayout_11")
        self.frame_2 = QFrame(self.simSettings_tab)
        self.frame_2.setObjectName(u"frame_2")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy5.setHorizontalStretch(1)
        sizePolicy5.setVerticalStretch(1)
        sizePolicy5.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy5)
        self.frame_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_8 = QVBoxLayout(self.frame_2)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.splitter_5 = QSplitter(self.frame_2)
        self.splitter_5.setObjectName(u"splitter_5")
        self.splitter_5.setOrientation(Qt.Orientation.Vertical)
        self.splitter_2 = QSplitter(self.splitter_5)
        self.splitter_2.setObjectName(u"splitter_2")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        sizePolicy6.setHorizontalStretch(1)
        sizePolicy6.setVerticalStretch(1)
        sizePolicy6.setHeightForWidth(self.splitter_2.sizePolicy().hasHeightForWidth())
        self.splitter_2.setSizePolicy(sizePolicy6)
        self.splitter_2.setMaximumSize(QSize(16777214, 337))
        self.splitter_2.setOrientation(Qt.Orientation.Vertical)
        self.p1SweepConfig_f = QFrame(self.splitter_2)
        self.p1SweepConfig_f.setObjectName(u"p1SweepConfig_f")
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy7.setHorizontalStretch(1)
        sizePolicy7.setVerticalStretch(1)
        sizePolicy7.setHeightForWidth(self.p1SweepConfig_f.sizePolicy().hasHeightForWidth())
        self.p1SweepConfig_f.setSizePolicy(sizePolicy7)
        self.p1SweepConfig_f.setMaximumSize(QSize(16777214, 100))
        self.p1SweepConfig_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.p1SweepConfig_f.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_8 = QGridLayout(self.p1SweepConfig_f)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.p1From_l = QLabel(self.p1SweepConfig_f)
        self.p1From_l.setObjectName(u"p1From_l")
        sizePolicy5.setHeightForWidth(self.p1From_l.sizePolicy().hasHeightForWidth())
        self.p1From_l.setSizePolicy(sizePolicy5)
        self.p1From_l.setMaximumSize(QSize(16777215, 40))
        font4 = QFont()
        font4.setPointSize(12)
        self.p1From_l.setFont(font4)

        self.gridLayout_8.addWidget(self.p1From_l, 0, 1, 1, 1)

        self.p1To_l = QLabel(self.p1SweepConfig_f)
        self.p1To_l.setObjectName(u"p1To_l")
        sizePolicy8 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy8.setHorizontalStretch(1)
        sizePolicy8.setVerticalStretch(1)
        sizePolicy8.setHeightForWidth(self.p1To_l.sizePolicy().hasHeightForWidth())
        self.p1To_l.setSizePolicy(sizePolicy8)
        self.p1To_l.setFont(font4)

        self.gridLayout_8.addWidget(self.p1To_l, 0, 4, 1, 1)

        self.p1Var_l = QLabel(self.p1SweepConfig_f)
        self.p1Var_l.setObjectName(u"p1Var_l")
        sizePolicy5.setHeightForWidth(self.p1Var_l.sizePolicy().hasHeightForWidth())
        self.p1Var_l.setSizePolicy(sizePolicy5)
        self.p1Var_l.setFont(font4)

        self.gridLayout_8.addWidget(self.p1Var_l, 0, 0, 1, 1)

        self.p1Select_dd = QComboBox(self.p1SweepConfig_f)
        self.p1Select_dd.setObjectName(u"p1Select_dd")
        sizePolicy6.setHeightForWidth(self.p1Select_dd.sizePolicy().hasHeightForWidth())
        self.p1Select_dd.setSizePolicy(sizePolicy6)
        self.p1Select_dd.setFont(font4)

        self.gridLayout_8.addWidget(self.p1Select_dd, 1, 0, 1, 1)

        self.p1SweepLower_entry = QTextEdit(self.p1SweepConfig_f)
        self.p1SweepLower_entry.setObjectName(u"p1SweepLower_entry")
        sizePolicy7.setHeightForWidth(self.p1SweepLower_entry.sizePolicy().hasHeightForWidth())
        self.p1SweepLower_entry.setSizePolicy(sizePolicy7)
        self.p1SweepLower_entry.setMaximumSize(QSize(16777215, 40))
        font5 = QFont()
        font5.setPointSize(10)
        self.p1SweepLower_entry.setFont(font5)

        self.gridLayout_8.addWidget(self.p1SweepLower_entry, 1, 1, 1, 1)

        self.p1SweepUpper_entry = QTextEdit(self.p1SweepConfig_f)
        self.p1SweepUpper_entry.setObjectName(u"p1SweepUpper_entry")
        sizePolicy7.setHeightForWidth(self.p1SweepUpper_entry.sizePolicy().hasHeightForWidth())
        self.p1SweepUpper_entry.setSizePolicy(sizePolicy7)
        self.p1SweepUpper_entry.setMaximumSize(QSize(16777215, 40))
        self.p1SweepUpper_entry.setFont(font5)
        self.p1SweepUpper_entry.setAcceptRichText(False)

        self.gridLayout_8.addWidget(self.p1SweepUpper_entry, 1, 4, 1, 1)

        self.p1SweepScale_box = QGroupBox(self.p1SweepConfig_f)
        self.p1SweepScale_box.setObjectName(u"p1SweepScale_box")
        sizePolicy5.setHeightForWidth(self.p1SweepScale_box.sizePolicy().hasHeightForWidth())
        self.p1SweepScale_box.setSizePolicy(sizePolicy5)
        self.p1SweepScale_box.setFont(font4)
        self.verticalLayout_7 = QVBoxLayout(self.p1SweepScale_box)
        self.verticalLayout_7.setSpacing(1)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(-1, 1, -1, 1)
        self.p1Lin_button = QRadioButton(self.p1SweepScale_box)
        self.param1SweepScale_buttons = QButtonGroup(MainWindow)
        self.param1SweepScale_buttons.setObjectName(u"param1SweepScale_buttons")
        self.param1SweepScale_buttons.addButton(self.p1Lin_button)
        self.p1Lin_button.setObjectName(u"p1Lin_button")
        sizePolicy9 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy9.setHorizontalStretch(1)
        sizePolicy9.setVerticalStretch(1)
        sizePolicy9.setHeightForWidth(self.p1Lin_button.sizePolicy().hasHeightForWidth())
        self.p1Lin_button.setSizePolicy(sizePolicy9)
        self.p1Lin_button.setFont(font4)
        self.p1Lin_button.setChecked(True)

        self.verticalLayout_7.addWidget(self.p1Lin_button)

        self.p1Log_button = QRadioButton(self.p1SweepScale_box)
        self.param1SweepScale_buttons.addButton(self.p1Log_button)
        self.p1Log_button.setObjectName(u"p1Log_button")
        sizePolicy9.setHeightForWidth(self.p1Log_button.sizePolicy().hasHeightForWidth())
        self.p1Log_button.setSizePolicy(sizePolicy9)
        self.p1Log_button.setFont(font4)

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
        sizePolicy7.setHeightForWidth(self.p2SweepConfig_f.sizePolicy().hasHeightForWidth())
        self.p2SweepConfig_f.setSizePolicy(sizePolicy7)
        self.p2SweepConfig_f.setMaximumSize(QSize(16777215, 232))
        self.p2SweepConfig_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.p2SweepConfig_f.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_10 = QGridLayout(self.p2SweepConfig_f)
        self.gridLayout_10.setObjectName(u"gridLayout_10")
        self.p2SweepUpper_entry_4 = QTextEdit(self.p2SweepConfig_f)
        self.p2SweepUpper_entry_4.setObjectName(u"p2SweepUpper_entry_4")
        sizePolicy7.setHeightForWidth(self.p2SweepUpper_entry_4.sizePolicy().hasHeightForWidth())
        self.p2SweepUpper_entry_4.setSizePolicy(sizePolicy7)
        self.p2SweepUpper_entry_4.setMaximumSize(QSize(16777215, 40))
        self.p2SweepUpper_entry_4.setFont(font5)
        self.p2SweepUpper_entry_4.setAcceptRichText(False)

        self.gridLayout_10.addWidget(self.p2SweepUpper_entry_4, 1, 4, 1, 1)

        self.p2SweepLower_entry_4 = QTextEdit(self.p2SweepConfig_f)
        self.p2SweepLower_entry_4.setObjectName(u"p2SweepLower_entry_4")
        sizePolicy7.setHeightForWidth(self.p2SweepLower_entry_4.sizePolicy().hasHeightForWidth())
        self.p2SweepLower_entry_4.setSizePolicy(sizePolicy7)
        self.p2SweepLower_entry_4.setMaximumSize(QSize(16777215, 40))
        self.p2SweepLower_entry_4.setFont(font5)

        self.gridLayout_10.addWidget(self.p2SweepLower_entry_4, 1, 1, 1, 1)

        self.p2From_l = QLabel(self.p2SweepConfig_f)
        self.p2From_l.setObjectName(u"p2From_l")
        sizePolicy5.setHeightForWidth(self.p2From_l.sizePolicy().hasHeightForWidth())
        self.p2From_l.setSizePolicy(sizePolicy5)
        self.p2From_l.setMaximumSize(QSize(16777215, 40))
        self.p2From_l.setFont(font4)

        self.gridLayout_10.addWidget(self.p2From_l, 0, 1, 1, 1)

        self.p2Var_l = QLabel(self.p2SweepConfig_f)
        self.p2Var_l.setObjectName(u"p2Var_l")
        sizePolicy5.setHeightForWidth(self.p2Var_l.sizePolicy().hasHeightForWidth())
        self.p2Var_l.setSizePolicy(sizePolicy5)
        self.p2Var_l.setMaximumSize(QSize(16777215, 40))
        self.p2Var_l.setFont(font4)

        self.gridLayout_10.addWidget(self.p2Var_l, 0, 0, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_10.addItem(self.verticalSpacer_2, 3, 1, 1, 1)

        self.p2Select_dd = QComboBox(self.p2SweepConfig_f)
        self.p2Select_dd.setObjectName(u"p2Select_dd")
        sizePolicy6.setHeightForWidth(self.p2Select_dd.sizePolicy().hasHeightForWidth())
        self.p2Select_dd.setSizePolicy(sizePolicy6)
        self.p2Select_dd.setFont(font4)

        self.gridLayout_10.addWidget(self.p2Select_dd, 1, 0, 1, 1)

        self.p2To_l_4 = QLabel(self.p2SweepConfig_f)
        self.p2To_l_4.setObjectName(u"p2To_l_4")
        sizePolicy8.setHeightForWidth(self.p2To_l_4.sizePolicy().hasHeightForWidth())
        self.p2To_l_4.setSizePolicy(sizePolicy8)
        self.p2To_l_4.setFont(font4)

        self.gridLayout_10.addWidget(self.p2To_l_4, 0, 4, 1, 1)

        self.p2SweepScale_box = QGroupBox(self.p2SweepConfig_f)
        self.p2SweepScale_box.setObjectName(u"p2SweepScale_box")
        sizePolicy5.setHeightForWidth(self.p2SweepScale_box.sizePolicy().hasHeightForWidth())
        self.p2SweepScale_box.setSizePolicy(sizePolicy5)
        self.p2SweepScale_box.setFont(font4)
        self.verticalLayout_9 = QVBoxLayout(self.p2SweepScale_box)
        self.verticalLayout_9.setSpacing(1)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(-1, 1, -1, 1)
        self.p2Lin_button = QRadioButton(self.p2SweepScale_box)
        self.param2SweepScale_buttons = QButtonGroup(MainWindow)
        self.param2SweepScale_buttons.setObjectName(u"param2SweepScale_buttons")
        self.param2SweepScale_buttons.addButton(self.p2Lin_button)
        self.p2Lin_button.setObjectName(u"p2Lin_button")
        sizePolicy9.setHeightForWidth(self.p2Lin_button.sizePolicy().hasHeightForWidth())
        self.p2Lin_button.setSizePolicy(sizePolicy9)
        self.p2Lin_button.setFont(font4)
        self.p2Lin_button.setChecked(True)

        self.verticalLayout_9.addWidget(self.p2Lin_button)

        self.p2Log_button = QRadioButton(self.p2SweepScale_box)
        self.param2SweepScale_buttons.addButton(self.p2Log_button)
        self.p2Log_button.setObjectName(u"p2Log_button")
        sizePolicy9.setHeightForWidth(self.p2Log_button.sizePolicy().hasHeightForWidth())
        self.p2Log_button.setSizePolicy(sizePolicy9)
        self.p2Log_button.setFont(font4)

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
        sizePolicy10 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        sizePolicy10.setHorizontalStretch(0)
        sizePolicy10.setVerticalStretch(0)
        sizePolicy10.setHeightForWidth(self.simTime_f.sizePolicy().hasHeightForWidth())
        self.simTime_f.setSizePolicy(sizePolicy10)
        self.simTime_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.simTime_f.setFrameShadow(QFrame.Shadow.Raised)
        self.formLayout = QFormLayout(self.simTime_f)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setSizeConstraint(QLayout.SizeConstraint.SetMaximumSize)
        self.formLayout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.formLayout.setVerticalSpacing(2)
        self.formLayout.setContentsMargins(-1, -1, -1, 3)
        self.duration_l = QLabel(self.simTime_f)
        self.duration_l.setObjectName(u"duration_l")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.duration_l)

        self.warmup_l = QLabel(self.simTime_f)
        self.warmup_l.setObjectName(u"warmup_l")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.warmup_l)

        self.warmup_e = QPlainTextEdit(self.simTime_f)
        self.warmup_e.setObjectName(u"warmup_e")
        sizePolicy11 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        sizePolicy11.setHorizontalStretch(1)
        sizePolicy11.setVerticalStretch(1)
        sizePolicy11.setHeightForWidth(self.warmup_e.sizePolicy().hasHeightForWidth())
        self.warmup_e.setSizePolicy(sizePolicy11)
        self.warmup_e.setMaximumSize(QSize(16777215, 40))
        self.warmup_e.setFont(font5)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.warmup_e)

        self.fs_l = QLabel(self.simTime_f)
        self.fs_l.setObjectName(u"fs_l")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.fs_l)

        self.fs_e = QPlainTextEdit(self.simTime_f)
        self.fs_e.setObjectName(u"fs_e")
        sizePolicy12 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy12.setHorizontalStretch(0)
        sizePolicy12.setVerticalStretch(0)
        sizePolicy12.setHeightForWidth(self.fs_e.sizePolicy().hasHeightForWidth())
        self.fs_e.setSizePolicy(sizePolicy12)
        self.fs_e.setMaximumSize(QSize(16777215, 40))
        self.fs_e.setFont(font5)

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.fs_e)

        self.stepsize_l = QLabel(self.simTime_f)
        self.stepsize_l.setObjectName(u"stepsize_l")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.stepsize_l)

        self.stepsize_e = QPlainTextEdit(self.simTime_f)
        self.stepsize_e.setObjectName(u"stepsize_e")
        sizePolicy7.setHeightForWidth(self.stepsize_e.sizePolicy().hasHeightForWidth())
        self.stepsize_e.setSizePolicy(sizePolicy7)
        self.stepsize_e.setMaximumSize(QSize(16777215, 40))
        self.stepsize_e.setFont(font5)

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.stepsize_e)

        self.duration_e = QPlainTextEdit(self.simTime_f)
        self.duration_e.setObjectName(u"duration_e")
        sizePolicy7.setHeightForWidth(self.duration_e.sizePolicy().hasHeightForWidth())
        self.duration_e.setSizePolicy(sizePolicy7)
        self.duration_e.setMaximumSize(QSize(16777215, 40))
        self.duration_e.setSizeIncrement(QSize(10, 10))
        self.duration_e.setBaseSize(QSize(0, 60))
        font6 = QFont()
        font6.setFamilies([u"Segoe UI"])
        font6.setPointSize(10)
        font6.setStyleStrategy(QFont.NoAntialias)
        self.duration_e.setFont(font6)

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.duration_e)

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
        self.sysParams_tab.setGeometry(QRect(0, 0, 501, 496))
        sizePolicy5.setHeightForWidth(self.sysParams_tab.sizePolicy().hasHeightForWidth())
        self.sysParams_tab.setSizePolicy(sizePolicy5)
        self.verticalLayout_11 = QVBoxLayout(self.sysParams_tab)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.label_5 = QLabel(self.sysParams_tab)
        self.label_5.setObjectName(u"label_5")
        sizePolicy13 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy13.setHorizontalStretch(0)
        sizePolicy13.setVerticalStretch(0)
        sizePolicy13.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy13)
        font7 = QFont()
        font7.setPointSize(16)
        font7.setItalic(True)
        self.label_5.setFont(font7)

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
        font8 = QFont()
        font8.setPointSize(24)
        self.save_button.setFont(font8)

        self.horizontalLayout.addWidget(self.save_button)

        self.solve_button = QPushButton(self.saveOrSolve_f)
        self.solve_button.setObjectName(u"solve_button")
        sizePolicy.setHeightForWidth(self.solve_button.sizePolicy().hasHeightForWidth())
        self.solve_button.setSizePolicy(sizePolicy)
        self.solve_button.setFont(font8)

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
        self.menubar.setGeometry(QRect(0, 0, 1592, 22))
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
        self.animatePlot_button.objectNameChanged.connect(MainWindow.animate_3D)
        self.frequency_slider.valueChanged.connect(self.currentFrequency_l.display)
        self.frequency_slider.valueChanged.connect(MainWindow.set_current_fft_freq)
        self.p1Select_dd.currentTextChanged.connect(MainWindow.set_param1_var)
        self.p2Select_dd.currentTextChanged.connect(MainWindow.set_param2_var)
        self.p1SweepLower_entry.textChanged.connect(MainWindow.set_param1Sweep_bounds)
        self.p1SweepUpper_entry.textChanged.connect(MainWindow.set_param1Sweep_bounds)
        self.p2SweepLower_entry_4.textChanged.connect(MainWindow.set_param2Sweep_bounds)
        self.p2SweepUpper_entry_4.textChanged.connect(MainWindow.set_param2Sweep_bounds)
        self.param1SweepScale_buttons.idClicked.connect(MainWindow.set_param1Sweep_scale)
        self.param2SweepScale_buttons.idClicked.connect(MainWindow.set_param2Sweep_scale)
        self.duration_e.textChanged.connect(MainWindow.set_duration)
        self.warmup_e.textChanged.connect(MainWindow.set_warmup)
        self.fs_e.textChanged.connect(MainWindow.set_fs)
        self.stepsize_e.textChanged.connect(MainWindow.set_dt)
        self.init0_e.textChanged.connect(MainWindow.set_y0)
        self.save_button.clicked.connect(MainWindow.save_results)
        self.solve_button.released.connect(MainWindow.solve_ODE)
        self.param1ValSelect_dd.currentTextChanged.connect(MainWindow.select_param1)
        self.param2ValSelect_dd.currentTextChanged.connect(MainWindow.select_param2)
        self.actionLoad_System_File.triggered.connect(MainWindow.load_system_from_filedialog)

        self.plotSettingsTabs.setCurrentIndex(0)
        self.simSettingsTabs.setCurrentIndex(0)


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
        self.yAxis_l.setText(QCoreApplication.translate("MainWindow", u"y axis", None))
        self.zAxis_l.setText(QCoreApplication.translate("MainWindow", u"z axis", None))
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
        self.p1SweepLower_entry.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0.0</p></body></html>", None))
        self.p1SweepUpper_entry.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
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
        self.p2SweepUpper_entry_4.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1.0</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>", None))
        self.p2SweepLower_entry_4.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
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
        self.duration_l.setToolTip(QCoreApplication.translate("MainWindow", u"How long to record for", None))
#endif // QT_CONFIG(tooltip)
        self.duration_l.setText(QCoreApplication.translate("MainWindow", u"Duration", None))
#if QT_CONFIG(tooltip)
        self.warmup_l.setToolTip(QCoreApplication.translate("MainWindow", u"How long to wait before you start recording (to reach steady state)", None))
#endif // QT_CONFIG(tooltip)
        self.warmup_l.setText(QCoreApplication.translate("MainWindow", u"Warmup", None))
        self.warmup_e.setPlainText(QCoreApplication.translate("MainWindow", u"100.0\n"
"", None))
#if QT_CONFIG(tooltip)
        self.fs_l.setToolTip(QCoreApplication.translate("MainWindow", u"Sample rate - how frequently you need to record samples. Max response freq * 2.5 is pretty safe.", None))
#endif // QT_CONFIG(tooltip)
        self.fs_l.setText(QCoreApplication.translate("MainWindow", u"fs", None))
        self.fs_e.setPlainText(QCoreApplication.translate("MainWindow", u"1.0", None))
#if QT_CONFIG(tooltip)
        self.stepsize_l.setToolTip(QCoreApplication.translate("MainWindow", u"(for fixed step solvers) The integration step size", None))
#endif // QT_CONFIG(tooltip)
        self.stepsize_l.setText(QCoreApplication.translate("MainWindow", u"Step Size", None))
        self.stepsize_e.setPlainText(QCoreApplication.translate("MainWindow", u"0.001", None))
        self.duration_e.setPlainText(QCoreApplication.translate("MainWindow", u"100.0\n"
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

