# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'sim_controller.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from qtpy.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from qtpy.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from qtpy.QtWidgets import (QApplication, QFormLayout, QFrame, QGridLayout,
    QGroupBox, QLabel, QLayout, QPushButton,
    QSizePolicy, QSpacerItem, QSplitter, QTabWidget,
    QVBoxLayout, QWidget)

from gui.resources.widgets.floatLineEdit import floatLineEdit
from gui.resources.widgets.variable_from_to_n_scale_widget import variable_from_to_n_scale_widget

class Ui_simController(object):
    def setupUi(self, simController):
        if not simController.objectName():
            simController.setObjectName(u"simController")
        simController.resize(467, 634)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(simController.sizePolicy().hasHeightForWidth())
        simController.setSizePolicy(sizePolicy)
        self.verticalLayout_2 = QVBoxLayout(simController)
        self.verticalLayout_2.setSpacing(1)
        self.verticalLayout_2.setContentsMargins(9, 9, 9, 9)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.frame = QFrame(simController)
        self.frame.setObjectName(u"frame")
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout = QVBoxLayout(self.frame)
        self.verticalLayout.setSpacing(1)
        self.verticalLayout.setContentsMargins(9, 9, 9, 9)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.simSettingsTabs = QTabWidget(self.frame)
        self.simSettingsTabs.setObjectName(u"simSettingsTabs")
        sizePolicy.setHeightForWidth(self.simSettingsTabs.sizePolicy().hasHeightForWidth())
        self.simSettingsTabs.setSizePolicy(sizePolicy)
        font = QFont()
        font.setPointSize(16)
        self.simSettingsTabs.setFont(font)
        self.SimulationSettings_tab = QWidget()
        self.SimulationSettings_tab.setObjectName(u"SimulationSettings_tab")
        self.gridLayout_11 = QGridLayout(self.SimulationSettings_tab)
        self.gridLayout_11.setSpacing(1)
        self.gridLayout_11.setContentsMargins(9, 9, 9, 9)
        self.gridLayout_11.setObjectName(u"gridLayout_11")
        self.frame_2 = QFrame(self.SimulationSettings_tab)
        self.frame_2.setObjectName(u"frame_2")
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_8 = QVBoxLayout(self.frame_2)
        self.verticalLayout_8.setSpacing(1)
        self.verticalLayout_8.setContentsMargins(9, 9, 9, 9)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.splitter = QSplitter(self.frame_2)
        self.splitter.setObjectName(u"splitter")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy1)
        self.splitter.setOrientation(Qt.Orientation.Vertical)
        self.param1Sweep_options = variable_from_to_n_scale_widget(self.splitter)
        self.param1Sweep_options.setObjectName(u"param1Sweep_options")
        sizePolicy.setHeightForWidth(self.param1Sweep_options.sizePolicy().hasHeightForWidth())
        self.param1Sweep_options.setSizePolicy(sizePolicy)
        self.param1Sweep_options.setMinimumSize(QSize(0, 80))
        self.param1Sweep_options.setFrameShape(QFrame.Shape.StyledPanel)
        self.param1Sweep_options.setFrameShadow(QFrame.Shadow.Raised)
        self.param1Sweep_options.setProperty("variableMax", 10000000.000000000000000)
        self.param1Sweep_options.setProperty("variableMin", -10000000.000000000000000)
        self.param1Sweep_options.setProperty("variableDecimals", 10.000000000000000)
        self.param1Sweep_options.setProperty("fromEntryDefault", 0.000000000000000)
        self.param1Sweep_options.setProperty("toEntryDefault", 1.000000000000000)
        self.param1Sweep_options.setProperty("nEntryMinValue", 0.000000000000000)
        self.param1Sweep_options.setProperty("nEntryMaxValue", 10000000.000000000000000)
        self.param1Sweep_options.setProperty("nEntryDefault", 100.000000000000000)
        self.splitter.addWidget(self.param1Sweep_options)
        self.param2Sweep_options = variable_from_to_n_scale_widget(self.splitter)
        self.param2Sweep_options.setObjectName(u"param2Sweep_options")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy2.setHorizontalStretch(1)
        sizePolicy2.setVerticalStretch(1)
        sizePolicy2.setHeightForWidth(self.param2Sweep_options.sizePolicy().hasHeightForWidth())
        self.param2Sweep_options.setSizePolicy(sizePolicy2)
        self.param2Sweep_options.setMinimumSize(QSize(0, 80))
        self.param2Sweep_options.setFrameShape(QFrame.Shape.StyledPanel)
        self.param2Sweep_options.setFrameShadow(QFrame.Shadow.Raised)
        self.param2Sweep_options.setProperty("variableMax", 10000000.000000000000000)
        self.param2Sweep_options.setProperty("variableMin", -10000000.000000000000000)
        self.param2Sweep_options.setProperty("variableDecimals", 10.000000000000000)
        self.param2Sweep_options.setProperty("fromEntryDefault", 0.000000000000000)
        self.param2Sweep_options.setProperty("toEntryDefault", 1.000000000000000)
        self.param2Sweep_options.setProperty("nEntryMinValue", 0.000000000000000)
        self.param2Sweep_options.setProperty("nEntryMaxValue", 1000000.000000000000000)
        self.param2Sweep_options.setProperty("nEntryDefault", 100.000000000000000)
        self.splitter.addWidget(self.param2Sweep_options)

        self.verticalLayout_8.addWidget(self.splitter)

        self.splitter_2 = QSplitter(self.frame_2)
        self.splitter_2.setObjectName(u"splitter_2")
        self.splitter_2.setOrientation(Qt.Orientation.Vertical)
        self.simTime_f = QFrame(self.splitter_2)
        self.simTime_f.setObjectName(u"simTime_f")
        sizePolicy.setHeightForWidth(self.simTime_f.sizePolicy().hasHeightForWidth())
        self.simTime_f.setSizePolicy(sizePolicy)
        self.simTime_f.setFrameShape(QFrame.Shape.StyledPanel)
        self.simTime_f.setFrameShadow(QFrame.Shadow.Raised)
        self.formLayout = QFormLayout(self.simTime_f)
        self.formLayout.setSpacing(1)
        self.formLayout.setContentsMargins(9, 9, 9, 9)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setSizeConstraint(QLayout.SizeConstraint.SetMaximumSize)
        self.formLayout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.formLayout.setVerticalSpacing(2)
        self.formLayout.setContentsMargins(-1, -1, -1, 3)
        self.duration_l = QLabel(self.simTime_f)
        self.duration_l.setObjectName(u"duration_l")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy3.setHorizontalStretch(1)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.duration_l.sizePolicy().hasHeightForWidth())
        self.duration_l.setSizePolicy(sizePolicy3)
        font1 = QFont()
        font1.setPointSize(12)
        self.duration_l.setFont(font1)

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.duration_l)

        self.warmup_l = QLabel(self.simTime_f)
        self.warmup_l.setObjectName(u"warmup_l")
        sizePolicy3.setHeightForWidth(self.warmup_l.sizePolicy().hasHeightForWidth())
        self.warmup_l.setSizePolicy(sizePolicy3)
        self.warmup_l.setFont(font1)

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.warmup_l)

        self.fs_l = QLabel(self.simTime_f)
        self.fs_l.setObjectName(u"fs_l")
        sizePolicy3.setHeightForWidth(self.fs_l.sizePolicy().hasHeightForWidth())
        self.fs_l.setSizePolicy(sizePolicy3)
        self.fs_l.setFont(font1)

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.fs_l)

        self.stepsize_l = QLabel(self.simTime_f)
        self.stepsize_l.setObjectName(u"stepsize_l")
        sizePolicy3.setHeightForWidth(self.stepsize_l.sizePolicy().hasHeightForWidth())
        self.stepsize_l.setSizePolicy(sizePolicy3)
        self.stepsize_l.setFont(font1)

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.stepsize_l)

        self.stepsize_e = floatLineEdit(self.simTime_f)
        self.stepsize_e.setObjectName(u"stepsize_e")
        sizePolicy3.setHeightForWidth(self.stepsize_e.sizePolicy().hasHeightForWidth())
        self.stepsize_e.setSizePolicy(sizePolicy3)
        self.stepsize_e.setFont(font1)
        self.stepsize_e.setProperty("minValue", 0.000000000000000)
        self.stepsize_e.setProperty("maxValue", 1000000000.000000000000000)
        self.stepsize_e.setProperty("decimals", 10)

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.stepsize_e)

        self.fs_e = floatLineEdit(self.simTime_f)
        self.fs_e.setObjectName(u"fs_e")
        sizePolicy3.setHeightForWidth(self.fs_e.sizePolicy().hasHeightForWidth())
        self.fs_e.setSizePolicy(sizePolicy3)
        self.fs_e.setFont(font1)
        self.fs_e.setProperty("minValue", 0.000000000000000)
        self.fs_e.setProperty("maxValue", 1000000000.000000000000000)
        self.fs_e.setProperty("decimals", 10)

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.fs_e)

        self.warmup_e = floatLineEdit(self.simTime_f)
        self.warmup_e.setObjectName(u"warmup_e")
        sizePolicy3.setHeightForWidth(self.warmup_e.sizePolicy().hasHeightForWidth())
        self.warmup_e.setSizePolicy(sizePolicy3)
        self.warmup_e.setFont(font1)
        self.warmup_e.setProperty("minValue", 0.000000000000000)
        self.warmup_e.setProperty("maxValue", 1000000000.000000000000000)
        self.warmup_e.setProperty("decimals", 10)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.warmup_e)

        self.duration_e = floatLineEdit(self.simTime_f)
        self.duration_e.setObjectName(u"duration_e")
        sizePolicy3.setHeightForWidth(self.duration_e.sizePolicy().hasHeightForWidth())
        self.duration_e.setSizePolicy(sizePolicy3)
        self.duration_e.setFont(font1)
        self.duration_e.setProperty("minValue", 0.000000000000000)
        self.duration_e.setProperty("maxValue", 1000000000.000000000000000)
        self.duration_e.setProperty("decimals", 10)

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.duration_e)

        self.splitter_2.addWidget(self.simTime_f)
        self.inits_box = QGroupBox(self.splitter_2)
        self.inits_box.setObjectName(u"inits_box")
        sizePolicy.setHeightForWidth(self.inits_box.sizePolicy().hasHeightForWidth())
        self.inits_box.setSizePolicy(sizePolicy)
        self.inits_box.setFont(font1)
        self.gridLayout_9 = QGridLayout(self.inits_box)
        self.gridLayout_9.setSpacing(1)
        self.gridLayout_9.setContentsMargins(9, 9, 9, 9)
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.splitter_3 = QSplitter(self.inits_box)
        self.splitter_3.setObjectName(u"splitter_3")
        sizePolicy.setHeightForWidth(self.splitter_3.sizePolicy().hasHeightForWidth())
        self.splitter_3.setSizePolicy(sizePolicy)
        self.splitter_3.setOrientation(Qt.Orientation.Horizontal)
        self.init_0 = QLabel(self.splitter_3)
        self.init_0.setObjectName(u"init_0")
        sizePolicy1.setHeightForWidth(self.init_0.sizePolicy().hasHeightForWidth())
        self.init_0.setSizePolicy(sizePolicy1)
        self.init_0.setFont(font1)
        self.splitter_3.addWidget(self.init_0)
        self.init0_e = floatLineEdit(self.splitter_3)
        self.init0_e.setObjectName(u"init0_e")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        sizePolicy4.setHorizontalStretch(1)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.init0_e.sizePolicy().hasHeightForWidth())
        self.init0_e.setSizePolicy(sizePolicy4)
        self.init0_e.setFont(font1)
        self.init0_e.setProperty("minValue", 0.000000000000000)
        self.init0_e.setProperty("maxValue", 1000000000.000000000000000)
        self.init0_e.setProperty("decimals", 10)
        self.splitter_3.addWidget(self.init0_e)

        self.gridLayout_9.addWidget(self.splitter_3, 0, 0, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.gridLayout_9.addItem(self.verticalSpacer_2, 1, 0, 1, 1)

        self.splitter_2.addWidget(self.inits_box)

        self.verticalLayout_8.addWidget(self.splitter_2)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_8.addItem(self.verticalSpacer)


        self.gridLayout_11.addWidget(self.frame_2, 0, 0, 1, 2)

        self.simSettingsTabs.addTab(self.SimulationSettings_tab, "")
        self.SystemParameters_tab = QWidget()
        self.SystemParameters_tab.setObjectName(u"SystemParameters_tab")
        self.verticalLayout_11 = QVBoxLayout(self.SystemParameters_tab)
        self.verticalLayout_11.setSpacing(1)
        self.verticalLayout_11.setContentsMargins(9, 9, 9, 9)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.label_5 = QLabel(self.SystemParameters_tab)
        self.label_5.setObjectName(u"label_5")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        sizePolicy5.setHorizontalStretch(1)
        sizePolicy5.setVerticalStretch(1)
        sizePolicy5.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy5)
        font2 = QFont()
        font2.setPointSize(16)
        font2.setItalic(True)
        self.label_5.setFont(font2)

        self.verticalLayout_11.addWidget(self.label_5)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_11.addItem(self.verticalSpacer_4)

        self.simSettingsTabs.addTab(self.SystemParameters_tab, "")

        self.verticalLayout.addWidget(self.simSettingsTabs)

        self.solve_button = QPushButton(self.frame)
        self.solve_button.setObjectName(u"solve_button")
        sizePolicy5.setHeightForWidth(self.solve_button.sizePolicy().hasHeightForWidth())
        self.solve_button.setSizePolicy(sizePolicy5)
        font3 = QFont()
        font3.setPointSize(24)
        self.solve_button.setFont(font3)

        self.verticalLayout.addWidget(self.solve_button)


        self.verticalLayout_2.addWidget(self.frame)


        self.retranslateUi(simController)
        self.solve_button.released.connect(simController.solve)
        self.param1Sweep_options.from_changed.connect(simController.update_p1_from)
        self.param1Sweep_options.n_changed.connect(simController.update_p1_n)
        self.param1Sweep_options.scale_changed.connect(simController.update_p1_scale)
        self.param1Sweep_options.to_changed.connect(simController.update_p1_to)
        self.param1Sweep_options.variable_changed.connect(simController.update_p1_var)
        self.param2Sweep_options.from_changed.connect(simController.update_p2_from)
        self.param2Sweep_options.n_changed.connect(simController.update_p2_n)
        self.param2Sweep_options.scale_changed.connect(simController.update_p2_scale)
        self.param2Sweep_options.to_changed.connect(simController.update_p2_to)
        self.param2Sweep_options.variable_changed.connect(simController.update_p2_var)
        self.duration_e.textChanged.connect(simController.update_duration)
        self.warmup_e.textChanged.connect(simController.update_warmup)
        self.fs_e.textChanged.connect(simController.update_fs)
        self.stepsize_e.textChanged.connect(simController.update_dt)

        self.simSettingsTabs.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(simController)
    # setupUi

    def retranslateUi(self, simController):
        simController.setWindowTitle(QCoreApplication.translate("simController", u"Form", None))
        self.param1Sweep_options.setProperty("nEntryDecimals", QCoreApplication.translate("simController", u"0", None))
        self.param1Sweep_options.setProperty("varDdItems", [])
        self.param2Sweep_options.setProperty("nEntryDecimals", "")
        self.param2Sweep_options.setProperty("varDdItems", [])
#if QT_CONFIG(tooltip)
        self.duration_l.setToolTip(QCoreApplication.translate("simController", u"How long to record for", None))
#endif // QT_CONFIG(tooltip)
        self.duration_l.setText(QCoreApplication.translate("simController", u"Duration", None))
#if QT_CONFIG(tooltip)
        self.warmup_l.setToolTip(QCoreApplication.translate("simController", u"How long to wait before you start recording (to reach steady state)", None))
#endif // QT_CONFIG(tooltip)
        self.warmup_l.setText(QCoreApplication.translate("simController", u"Warmup", None))
#if QT_CONFIG(tooltip)
        self.fs_l.setToolTip(QCoreApplication.translate("simController", u"Sample rate - how frequently you need to record samples. Max response freq * 2.5 is pretty safe.", None))
#endif // QT_CONFIG(tooltip)
        self.fs_l.setText(QCoreApplication.translate("simController", u"fs", None))
#if QT_CONFIG(tooltip)
        self.stepsize_l.setToolTip(QCoreApplication.translate("simController", u"(for fixed step solvers) The integration step size", None))
#endif // QT_CONFIG(tooltip)
        self.stepsize_l.setText(QCoreApplication.translate("simController", u"Step Size", None))
        self.stepsize_e.setText(QCoreApplication.translate("simController", u"0.001", None))
        self.fs_e.setText(QCoreApplication.translate("simController", u"1", None))
        self.warmup_e.setText(QCoreApplication.translate("simController", u"100", None))
        self.duration_e.setText(QCoreApplication.translate("simController", u"1000", None))
        self.inits_box.setTitle(QCoreApplication.translate("simController", u"y0", None))
        self.init_0.setText(QCoreApplication.translate("simController", u"0", None))
        self.init0_e.setText(QCoreApplication.translate("simController", u"0.001", None))
        self.simSettingsTabs.setTabText(self.simSettingsTabs.indexOf(self.SimulationSettings_tab), QCoreApplication.translate("simController", u"Simulation Settings", None))
        self.label_5.setText(QCoreApplication.translate("simController", u"No System Loaded", None))
        self.simSettingsTabs.setTabText(self.simSettingsTabs.indexOf(self.SystemParameters_tab), QCoreApplication.translate("simController", u"System Parameters", None))
        self.solve_button.setText(QCoreApplication.translate("simController", u"Solve", None))
    # retranslateUi

