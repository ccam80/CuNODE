# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'plot3dGrid_frame.ui'
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
from qtpy.QtWidgets import (QApplication, QComboBox, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLCDNumber, QLabel,
    QPushButton, QRadioButton, QSizePolicy, QSlider,
    QSpacerItem, QSplitter, QTabWidget, QToolButton,
    QVBoxLayout, QWidget)

from gui.resources.widgets.variable_from_to_scale_widget import variable_from_to_scale_widget

class Ui_plotController(object):
    def setupUi(self, plotController):
        if not plotController.objectName():
            plotController.setObjectName(u"plotController")
        plotController.resize(445, 483)
        self.action1x = QAction(plotController)
        self.action1x.setObjectName(u"action1x")
        self.action1x.setMenuRole(QAction.MenuRole.NoRole)
        self.plotSettingsTabs = QTabWidget(plotController)
        self.plotSettingsTabs.setObjectName(u"plotSettingsTabs")
        self.plotSettingsTabs.setGeometry(QRect(9, 9, 441, 478))
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.plotSettingsTabs.sizePolicy().hasHeightForWidth())
        self.plotSettingsTabs.setSizePolicy(sizePolicy)
        font = QFont()
        font.setPointSize(12)
        font.setStrikeOut(False)
        font.setKerning(False)
        self.plotSettingsTabs.setFont(font)
        self.plotSettingsTabs.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.plotSettingsTabs.setUsesScrollButtons(False)
        self.plot3d_tab = QWidget()
        self.plot3d_tab.setObjectName(u"plot3d_tab")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(1)
        sizePolicy1.setVerticalStretch(1)
        sizePolicy1.setHeightForWidth(self.plot3d_tab.sizePolicy().hasHeightForWidth())
        self.plot3d_tab.setSizePolicy(sizePolicy1)
        self.verticalLayout = QVBoxLayout(self.plot3d_tab)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.xAxis_l = QLabel(self.plot3d_tab)
        self.xAxis_l.setObjectName(u"xAxis_l")
        font1 = QFont()
        font1.setPointSize(12)
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
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(1)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.yAxis_l.sizePolicy().hasHeightForWidth())
        self.yAxis_l.setSizePolicy(sizePolicy2)
        self.yAxis_l.setFont(font1)

        self.verticalLayout.addWidget(self.yAxis_l)

        self.yAxisGridPlotOtions = variable_from_to_scale_widget(self.plot3d_tab)
        self.yAxisGridPlotOtions.setObjectName(u"yAxisGridPlotOtions")
        self.yAxisGridPlotOtions.setFrameShape(QFrame.Shape.StyledPanel)
        self.yAxisGridPlotOtions.setFrameShadow(QFrame.Shadow.Raised)

        self.verticalLayout.addWidget(self.yAxisGridPlotOtions)

        self.zAxis_l = QLabel(self.plot3d_tab)
        self.zAxis_l.setObjectName(u"zAxis_l")
        self.zAxis_l.setFont(font1)

        self.verticalLayout.addWidget(self.zAxis_l)

        self.zAxisGridPlotOtions = variable_from_to_scale_widget(self.plot3d_tab)
        self.zAxisGridPlotOtions.setObjectName(u"zAxisGridPlotOtions")
        self.zAxisGridPlotOtions.setFrameShape(QFrame.Shape.StyledPanel)
        self.zAxisGridPlotOtions.setFrameShadow(QFrame.Shadow.Raised)

        self.verticalLayout.addWidget(self.zAxisGridPlotOtions)

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
        self.updatePlot_button = QPushButton(self.plotCommands_f)
        self.updatePlot_button.setObjectName(u"updatePlot_button")
        self.updatePlot_button.setFont(font2)

        self.horizontalLayout_3.addWidget(self.updatePlot_button)


        self.verticalLayout.addWidget(self.plotCommands_f)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_3)

        self.plotSettingsTabs.addTab(self.plot3d_tab, "")
        self.timeHistory3D_tab = QWidget()
        self.timeHistory3D_tab.setObjectName(u"timeHistory3D_tab")
        self.verticalLayout_3 = QVBoxLayout(self.timeHistory3D_tab)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.paramAxis_l = QLabel(self.timeHistory3D_tab)
        self.paramAxis_l.setObjectName(u"paramAxis_l")
        sizePolicy2.setHeightForWidth(self.paramAxis_l.sizePolicy().hasHeightForWidth())
        self.paramAxis_l.setSizePolicy(sizePolicy2)
        self.paramAxis_l.setFont(font1)

        self.verticalLayout_3.addWidget(self.paramAxis_l)

        self.progressionVarTime_options = variable_from_to_scale_widget(self.timeHistory3D_tab)
        self.progressionVarTime_options.setObjectName(u"progressionVarTime_options")
        self.progressionVarTime_options.setMinimumSize(QSize(0, 10))
        self.progressionVarTime_options.setFrameShape(QFrame.Shape.StyledPanel)
        self.progressionVarTime_options.setFrameShadow(QFrame.Shadow.Raised)

        self.verticalLayout_3.addWidget(self.progressionVarTime_options)

        self.timeAxis_l = QLabel(self.timeHistory3D_tab)
        self.timeAxis_l.setObjectName(u"timeAxis_l")
        sizePolicy2.setHeightForWidth(self.timeAxis_l.sizePolicy().hasHeightForWidth())
        self.timeAxis_l.setSizePolicy(sizePolicy2)
        self.timeAxis_l.setFont(font1)

        self.verticalLayout_3.addWidget(self.timeAxis_l)

        self.timeAxis3d_options = variable_from_to_scale_widget(self.timeHistory3D_tab)
        self.timeAxis3d_options.setObjectName(u"timeAxis3d_options")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.timeAxis3d_options.sizePolicy().hasHeightForWidth())
        self.timeAxis3d_options.setSizePolicy(sizePolicy3)
        self.timeAxis3d_options.setMinimumSize(QSize(0, 10))
        self.timeAxis3d_options.setFrameShape(QFrame.Shape.StyledPanel)
        self.timeAxis3d_options.setFrameShadow(QFrame.Shadow.Raised)

        self.verticalLayout_3.addWidget(self.timeAxis3d_options)

        self.amplAxis_l = QLabel(self.timeHistory3D_tab)
        self.amplAxis_l.setObjectName(u"amplAxis_l")
        self.amplAxis_l.setFont(font1)

        self.verticalLayout_3.addWidget(self.amplAxis_l)

        self.amplAxisTime3d_options = variable_from_to_scale_widget(self.timeHistory3D_tab)
        self.amplAxisTime3d_options.setObjectName(u"amplAxisTime3d_options")
        sizePolicy3.setHeightForWidth(self.amplAxisTime3d_options.sizePolicy().hasHeightForWidth())
        self.amplAxisTime3d_options.setSizePolicy(sizePolicy3)
        self.amplAxisTime3d_options.setMinimumSize(QSize(0, 10))
        self.amplAxisTime3d_options.setFrameShape(QFrame.Shape.StyledPanel)
        self.amplAxisTime3d_options.setFrameShadow(QFrame.Shadow.Raised)

        self.verticalLayout_3.addWidget(self.amplAxisTime3d_options)

        self.fixedParamSelectTime_f = QFrame(self.timeHistory3D_tab)
        self.fixedParamSelectTime_f.setObjectName(u"fixedParamSelectTime_f")
        self.fixedParamSelectTime_f.setFont(font2)
        self.fixedParamSelectTime_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.fixedParamSelectTime_f.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.fixedParamSelectTime_f)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(-1, 1, -1, 1)
        self.fixedParamTime_l = QLabel(self.fixedParamSelectTime_f)
        self.fixedParamTime_l.setObjectName(u"fixedParamTime_l")
        self.fixedParamTime_l.setFont(font2)

        self.horizontalLayout_4.addWidget(self.fixedParamTime_l)

        self.fixedParamTime_slider = QSlider(self.fixedParamSelectTime_f)
        self.fixedParamTime_slider.setObjectName(u"fixedParamTime_slider")
        self.fixedParamTime_slider.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_4.addWidget(self.fixedParamTime_slider)

        self.fixedParamTime_display = QLCDNumber(self.fixedParamSelectTime_f)
        self.fixedParamTime_display.setObjectName(u"fixedParamTime_display")

        self.horizontalLayout_4.addWidget(self.fixedParamTime_display)

        self.fixedParamTimeAnimate_button = QToolButton(self.fixedParamSelectTime_f)
        self.fixedParamTimeAnimate_button.setObjectName(u"fixedParamTimeAnimate_button")
        self.fixedParamTimeAnimate_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.fixedParamTimeAnimate_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.fixedParamTimeAnimate_button.setAutoRaise(False)
        self.fixedParamTimeAnimate_button.setArrowType(Qt.ArrowType.NoArrow)

        self.horizontalLayout_4.addWidget(self.fixedParamTimeAnimate_button)


        self.verticalLayout_3.addWidget(self.fixedParamSelectTime_f)

        self.updatePlot_button_2 = QPushButton(self.timeHistory3D_tab)
        self.updatePlot_button_2.setObjectName(u"updatePlot_button_2")
        self.updatePlot_button_2.setFont(font2)

        self.verticalLayout_3.addWidget(self.updatePlot_button_2)

        self.verticalSpacer_4 = QSpacerItem(20, 221, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer_4)

        self.plotSettingsTabs.addTab(self.timeHistory3D_tab, "")
        self.spectrogram3d_tab = QWidget()
        self.spectrogram3d_tab.setObjectName(u"spectrogram3d_tab")
        self.verticalLayout_2 = QVBoxLayout(self.spectrogram3d_tab)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.paramAxisSpec_l = QLabel(self.spectrogram3d_tab)
        self.paramAxisSpec_l.setObjectName(u"paramAxisSpec_l")
        sizePolicy2.setHeightForWidth(self.paramAxisSpec_l.sizePolicy().hasHeightForWidth())
        self.paramAxisSpec_l.setSizePolicy(sizePolicy2)
        self.paramAxisSpec_l.setFont(font1)

        self.verticalLayout_2.addWidget(self.paramAxisSpec_l)

        self.paramAxisSpec3d_options = variable_from_to_scale_widget(self.spectrogram3d_tab)
        self.paramAxisSpec3d_options.setObjectName(u"paramAxisSpec3d_options")
        self.paramAxisSpec3d_options.setMinimumSize(QSize(0, 10))
        self.paramAxisSpec3d_options.setFrameShape(QFrame.Shape.StyledPanel)
        self.paramAxisSpec3d_options.setFrameShadow(QFrame.Shadow.Raised)

        self.verticalLayout_2.addWidget(self.paramAxisSpec3d_options)

        self.freqAxisSpec3d_l = QLabel(self.spectrogram3d_tab)
        self.freqAxisSpec3d_l.setObjectName(u"freqAxisSpec3d_l")
        self.freqAxisSpec3d_l.setFont(font1)

        self.verticalLayout_2.addWidget(self.freqAxisSpec3d_l)

        self.freqAxisSpec3d_options = variable_from_to_scale_widget(self.spectrogram3d_tab)
        self.freqAxisSpec3d_options.setObjectName(u"freqAxisSpec3d_options")
        self.freqAxisSpec3d_options.setMinimumSize(QSize(0, 10))
        self.freqAxisSpec3d_options.setFrameShape(QFrame.Shape.StyledPanel)
        self.freqAxisSpec3d_options.setFrameShadow(QFrame.Shadow.Raised)

        self.verticalLayout_2.addWidget(self.freqAxisSpec3d_options)

        self.amplAxisSpec3d_l = QLabel(self.spectrogram3d_tab)
        self.amplAxisSpec3d_l.setObjectName(u"amplAxisSpec3d_l")
        sizePolicy2.setHeightForWidth(self.amplAxisSpec3d_l.sizePolicy().hasHeightForWidth())
        self.amplAxisSpec3d_l.setSizePolicy(sizePolicy2)
        self.amplAxisSpec3d_l.setFont(font1)

        self.verticalLayout_2.addWidget(self.amplAxisSpec3d_l)

        self.amplAxisSpec3d_options = variable_from_to_scale_widget(self.spectrogram3d_tab)
        self.amplAxisSpec3d_options.setObjectName(u"amplAxisSpec3d_options")
        self.amplAxisSpec3d_options.setMinimumSize(QSize(0, 10))
        self.amplAxisSpec3d_options.setFrameShape(QFrame.Shape.StyledPanel)
        self.amplAxisSpec3d_options.setFrameShadow(QFrame.Shadow.Raised)

        self.verticalLayout_2.addWidget(self.amplAxisSpec3d_options)

        self.fixedParamSelectTime_f_2 = QFrame(self.spectrogram3d_tab)
        self.fixedParamSelectTime_f_2.setObjectName(u"fixedParamSelectTime_f_2")
        self.fixedParamSelectTime_f_2.setFont(font2)
        self.fixedParamSelectTime_f_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.fixedParamSelectTime_f_2.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_5 = QHBoxLayout(self.fixedParamSelectTime_f_2)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(-1, 1, -1, 1)
        self.fixedParamTime_l_2 = QLabel(self.fixedParamSelectTime_f_2)
        self.fixedParamTime_l_2.setObjectName(u"fixedParamTime_l_2")
        self.fixedParamTime_l_2.setFont(font2)

        self.horizontalLayout_5.addWidget(self.fixedParamTime_l_2)

        self.fixedParamTime_slider_2 = QSlider(self.fixedParamSelectTime_f_2)
        self.fixedParamTime_slider_2.setObjectName(u"fixedParamTime_slider_2")
        self.fixedParamTime_slider_2.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_5.addWidget(self.fixedParamTime_slider_2)

        self.fixedParamTime_display_2 = QLCDNumber(self.fixedParamSelectTime_f_2)
        self.fixedParamTime_display_2.setObjectName(u"fixedParamTime_display_2")

        self.horizontalLayout_5.addWidget(self.fixedParamTime_display_2)

        self.fixedParamTimeAnimate_button_2 = QToolButton(self.fixedParamSelectTime_f_2)
        self.fixedParamTimeAnimate_button_2.setObjectName(u"fixedParamTimeAnimate_button_2")
        self.fixedParamTimeAnimate_button_2.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.fixedParamTimeAnimate_button_2.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.fixedParamTimeAnimate_button_2.setAutoRaise(False)
        self.fixedParamTimeAnimate_button_2.setArrowType(Qt.ArrowType.NoArrow)

        self.horizontalLayout_5.addWidget(self.fixedParamTimeAnimate_button_2)


        self.verticalLayout_2.addWidget(self.fixedParamSelectTime_f_2)

        self.updatePlot_button_3 = QPushButton(self.spectrogram3d_tab)
        self.updatePlot_button_3.setObjectName(u"updatePlot_button_3")
        self.updatePlot_button_3.setFont(font2)

        self.verticalLayout_2.addWidget(self.updatePlot_button_3)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer_2)

        self.plotSettingsTabs.addTab(self.spectrogram3d_tab, "")
        self.plotSingle_tab = QWidget()
        self.plotSingle_tab.setObjectName(u"plotSingle_tab")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.plotSingle_tab.sizePolicy().hasHeightForWidth())
        self.plotSingle_tab.setSizePolicy(sizePolicy4)
        self.verticalLayout_6 = QVBoxLayout(self.plotSingle_tab)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.splitter = QSplitter(self.plotSingle_tab)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Orientation.Vertical)
        self.singlePlotStyle_box = QGroupBox(self.splitter)
        self.singlePlotStyle_box.setObjectName(u"singlePlotStyle_box")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.singlePlotStyle_box.sizePolicy().hasHeightForWidth())
        self.singlePlotStyle_box.setSizePolicy(sizePolicy5)
        self.singlePlotStyle_box.setFont(font2)
        self.gridLayout_6 = QGridLayout(self.singlePlotStyle_box)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.singleSpectrogram_button = QRadioButton(self.singlePlotStyle_box)
        self.singleSpectrogram_button.setObjectName(u"singleSpectrogram_button")
        self.singleSpectrogram_button.setFont(font2)

        self.gridLayout_6.addWidget(self.singleSpectrogram_button, 0, 2, 1, 1)

        self.singleTimeDomain_button = QRadioButton(self.singlePlotStyle_box)
        self.singleTimeDomain_button.setObjectName(u"singleTimeDomain_button")
        self.singleTimeDomain_button.setFont(font2)

        self.gridLayout_6.addWidget(self.singleTimeDomain_button, 0, 4, 1, 1)

        self.phase2d_button = QRadioButton(self.singlePlotStyle_box)
        self.phase2d_button.setObjectName(u"phase2d_button")

        self.gridLayout_6.addWidget(self.phase2d_button, 1, 2, 1, 1)

        self.phase3d_button = QRadioButton(self.singlePlotStyle_box)
        self.phase3d_button.setObjectName(u"phase3d_button")
        self.phase3d_button.setChecked(True)

        self.gridLayout_6.addWidget(self.phase3d_button, 1, 4, 1, 1)

        self.splitter.addWidget(self.singlePlotStyle_box)
        self.paramSelect_f = QFrame(self.splitter)
        self.paramSelect_f.setObjectName(u"paramSelect_f")
        sizePolicy5.setHeightForWidth(self.paramSelect_f.sizePolicy().hasHeightForWidth())
        self.paramSelect_f.setSizePolicy(sizePolicy5)
        self.paramSelect_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.paramSelect_f.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_7 = QGridLayout(self.paramSelect_f)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.Param2Select_l = QLabel(self.paramSelect_f)
        self.Param2Select_l.setObjectName(u"Param2Select_l")
        self.Param2Select_l.setFont(font2)

        self.gridLayout_7.addWidget(self.Param2Select_l, 1, 0, 1, 1)

        self.param1ValSelect_dd = QComboBox(self.paramSelect_f)
        self.param1ValSelect_dd.setObjectName(u"param1ValSelect_dd")
        self.param1ValSelect_dd.setFont(font2)

        self.gridLayout_7.addWidget(self.param1ValSelect_dd, 0, 1, 1, 1)

        self.param2ValSelect_dd = QComboBox(self.paramSelect_f)
        self.param2ValSelect_dd.setObjectName(u"param2ValSelect_dd")
        self.param2ValSelect_dd.setFont(font2)

        self.gridLayout_7.addWidget(self.param2ValSelect_dd, 1, 1, 1, 1)

        self.param1Select_l = QLabel(self.paramSelect_f)
        self.param1Select_l.setObjectName(u"param1Select_l")
        self.param1Select_l.setFont(font2)

        self.gridLayout_7.addWidget(self.param1Select_l, 0, 0, 1, 1)

        self.fixedParamTimeAnimate_button_3 = QToolButton(self.paramSelect_f)
        self.fixedParamTimeAnimate_button_3.setObjectName(u"fixedParamTimeAnimate_button_3")
        self.fixedParamTimeAnimate_button_3.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.fixedParamTimeAnimate_button_3.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.fixedParamTimeAnimate_button_3.setAutoRaise(False)
        self.fixedParamTimeAnimate_button_3.setArrowType(Qt.ArrowType.NoArrow)

        self.gridLayout_7.addWidget(self.fixedParamTimeAnimate_button_3, 0, 2, 1, 1)

        self.fixedParamTimeAnimate_button_4 = QToolButton(self.paramSelect_f)
        self.fixedParamTimeAnimate_button_4.setObjectName(u"fixedParamTimeAnimate_button_4")
        self.fixedParamTimeAnimate_button_4.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.fixedParamTimeAnimate_button_4.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.fixedParamTimeAnimate_button_4.setAutoRaise(False)
        self.fixedParamTimeAnimate_button_4.setArrowType(Qt.ArrowType.NoArrow)

        self.gridLayout_7.addWidget(self.fixedParamTimeAnimate_button_4, 1, 2, 1, 1)

        self.splitter.addWidget(self.paramSelect_f)

        self.verticalLayout_6.addWidget(self.splitter)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_6.addItem(self.verticalSpacer)

        self.plotSettingsTabs.addTab(self.plotSingle_tab, "")

        self.retranslateUi(plotController)

        self.plotSettingsTabs.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(plotController)
    # setupUi

    def retranslateUi(self, plotController):
        plotController.setWindowTitle(QCoreApplication.translate("plotController", u"Form", None))
        self.action1x.setText(QCoreApplication.translate("plotController", u"1x", None))
        self.xAxis_l.setText(QCoreApplication.translate("plotController", u"x axis", None))
        self.yAxis_l.setText(QCoreApplication.translate("plotController", u"y axis", None))
        self.zAxis_l.setText(QCoreApplication.translate("plotController", u"z axis", None))
        self.frequency_l.setText(QCoreApplication.translate("plotController", u"Frequency", None))
        self.updatePlot_button.setText(QCoreApplication.translate("plotController", u"Update Plot", None))
        self.plotSettingsTabs.setTabText(self.plotSettingsTabs.indexOf(self.plot3d_tab), QCoreApplication.translate("plotController", u"3D Grid", None))
#if QT_CONFIG(tooltip)
        self.plotSettingsTabs.setTabToolTip(self.plotSettingsTabs.indexOf(self.plot3d_tab), QCoreApplication.translate("plotController", u"Plot a three-dimensional surface of selected results", None))
#endif // QT_CONFIG(tooltip)
        self.paramAxis_l.setText(QCoreApplication.translate("plotController", u"Parameter axis", None))
        self.timeAxis_l.setText(QCoreApplication.translate("plotController", u"Time axis", None))
        self.amplAxis_l.setText(QCoreApplication.translate("plotController", u"Amplitude axis", None))
        self.fixedParamTime_l.setText(QCoreApplication.translate("plotController", u"Fixed Parameter", None))
        self.fixedParamTimeAnimate_button.setText(QCoreApplication.translate("plotController", u"Animate", None))
        self.updatePlot_button_2.setText(QCoreApplication.translate("plotController", u"Update Plot", None))
        self.plotSettingsTabs.setTabText(self.plotSettingsTabs.indexOf(self.timeHistory3D_tab), QCoreApplication.translate("plotController", u"3D Time", None))
        self.paramAxisSpec_l.setText(QCoreApplication.translate("plotController", u"Parameter axis", None))
        self.freqAxisSpec3d_l.setText(QCoreApplication.translate("plotController", u"Frequency axis", None))
        self.amplAxisSpec3d_l.setText(QCoreApplication.translate("plotController", u"Amplitude axis", None))
        self.fixedParamTime_l_2.setText(QCoreApplication.translate("plotController", u"Fixed Parameter", None))
        self.fixedParamTimeAnimate_button_2.setText(QCoreApplication.translate("plotController", u"Animate", None))
        self.updatePlot_button_3.setText(QCoreApplication.translate("plotController", u"Update Plot", None))
        self.plotSettingsTabs.setTabText(self.plotSettingsTabs.indexOf(self.spectrogram3d_tab), QCoreApplication.translate("plotController", u"3D Spectrum", None))
        self.singlePlotStyle_box.setTitle(QCoreApplication.translate("plotController", u"Plot Style", None))
        self.singleSpectrogram_button.setText(QCoreApplication.translate("plotController", u"Spectrogram", None))
        self.singleTimeDomain_button.setText(QCoreApplication.translate("plotController", u"Time-domain", None))
        self.phase2d_button.setText(QCoreApplication.translate("plotController", u"2D Phase Diagram", None))
        self.phase3d_button.setText(QCoreApplication.translate("plotController", u"3D Phase Diagram", None))
        self.Param2Select_l.setText(QCoreApplication.translate("plotController", u"Parameter 2:", None))
        self.param1Select_l.setText(QCoreApplication.translate("plotController", u"Parameter 1:", None))
        self.fixedParamTimeAnimate_button_3.setText(QCoreApplication.translate("plotController", u"Animate", None))
        self.fixedParamTimeAnimate_button_4.setText(QCoreApplication.translate("plotController", u"Animate", None))
        self.plotSettingsTabs.setTabText(self.plotSettingsTabs.indexOf(self.plotSingle_tab), QCoreApplication.translate("plotController", u"Single Dataset", None))
#if QT_CONFIG(tooltip)
        self.plotSettingsTabs.setTabToolTip(self.plotSettingsTabs.indexOf(self.plotSingle_tab), QCoreApplication.translate("plotController", u"Plot a single dataset in two dimensions", None))
#endif // QT_CONFIG(tooltip)
    # retranslateUi

