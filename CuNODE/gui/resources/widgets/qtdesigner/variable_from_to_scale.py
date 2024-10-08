# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'variable_from_to_scale.ui'
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
from qtpy.QtWidgets import (QApplication, QButtonGroup, QComboBox, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLayout, QRadioButton, QSizePolicy, QVBoxLayout,
    QWidget)

from gui.resources.widgets.floatLineEdit import floatLineEdit

class Ui_variable_from_to_scale(object):
    def setupUi(self, variable_from_to_scale):
        if not variable_from_to_scale.objectName():
            variable_from_to_scale.setObjectName(u"variable_from_to_scale")
        variable_from_to_scale.resize(510, 80)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(variable_from_to_scale.sizePolicy().hasHeightForWidth())
        variable_from_to_scale.setSizePolicy(sizePolicy)
        variable_from_to_scale.setMinimumSize(QSize(0, 80))
        variable_from_to_scale.setMaximumSize(QSize(16777215, 100))
        self.horizontalLayout = QHBoxLayout(variable_from_to_scale)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.variable_from_to_frame = QFrame(variable_from_to_scale)
        self.variable_from_to_frame.setObjectName(u"variable_from_to_frame")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        sizePolicy1.setHorizontalStretch(1)
        sizePolicy1.setVerticalStretch(5)
        sizePolicy1.setHeightForWidth(self.variable_from_to_frame.sizePolicy().hasHeightForWidth())
        self.variable_from_to_frame.setSizePolicy(sizePolicy1)
        self.variable_from_to_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.variable_from_to_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_4 = QGridLayout(self.variable_from_to_frame)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_4.setSizeConstraint(QLayout.SizeConstraint.SetMaximumSize)
        self.gridLayout_4.setVerticalSpacing(0)
        self.gridLayout_4.setContentsMargins(-1, 0, -1, 0)
        self.From_l = QLabel(self.variable_from_to_frame)
        self.From_l.setObjectName(u"From_l")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.From_l.sizePolicy().hasHeightForWidth())
        self.From_l.setSizePolicy(sizePolicy2)
        font = QFont()
        font.setPointSize(12)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.From_l.setFont(font)

        self.gridLayout_4.addWidget(self.From_l, 0, 1, 1, 1)

        self.Var_l = QLabel(self.variable_from_to_frame)
        self.Var_l.setObjectName(u"Var_l")
        sizePolicy2.setHeightForWidth(self.Var_l.sizePolicy().hasHeightForWidth())
        self.Var_l.setSizePolicy(sizePolicy2)
        self.Var_l.setFont(font)

        self.gridLayout_4.addWidget(self.Var_l, 0, 0, 1, 1)

        self.to_entry = floatLineEdit(self.variable_from_to_frame)
        self.to_entry.setObjectName(u"to_entry")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy3.setHorizontalStretch(1)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.to_entry.sizePolicy().hasHeightForWidth())
        self.to_entry.setSizePolicy(sizePolicy3)
        self.to_entry.setProperty("minValue", 0.000000000000000)
        self.to_entry.setProperty("maxValue", 1000000000.000000000000000)
        self.to_entry.setProperty("decimals", 10)

        self.gridLayout_4.addWidget(self.to_entry, 1, 2, 1, 1)

        self.Var_dd = QComboBox(self.variable_from_to_frame)
        self.Var_dd.addItem("")
        self.Var_dd.addItem("")
        self.Var_dd.setObjectName(u"Var_dd")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        sizePolicy4.setHorizontalStretch(1)
        sizePolicy4.setVerticalStretch(1)
        sizePolicy4.setHeightForWidth(self.Var_dd.sizePolicy().hasHeightForWidth())
        self.Var_dd.setSizePolicy(sizePolicy4)
        self.Var_dd.setFont(font)

        self.gridLayout_4.addWidget(self.Var_dd, 1, 0, 1, 1)

        self.to_l = QLabel(self.variable_from_to_frame)
        self.to_l.setObjectName(u"to_l")
        sizePolicy2.setHeightForWidth(self.to_l.sizePolicy().hasHeightForWidth())
        self.to_l.setSizePolicy(sizePolicy2)
        self.to_l.setFont(font)

        self.gridLayout_4.addWidget(self.to_l, 0, 2, 1, 1)

        self.from_entry = floatLineEdit(self.variable_from_to_frame)
        self.from_entry.setObjectName(u"from_entry")
        sizePolicy3.setHeightForWidth(self.from_entry.sizePolicy().hasHeightForWidth())
        self.from_entry.setSizePolicy(sizePolicy3)
        self.from_entry.setProperty("minValue", 0.000000000000000)
        self.from_entry.setProperty("maxValue", 1000000000.000000000000000)
        self.from_entry.setProperty("decimals", 10)

        self.gridLayout_4.addWidget(self.from_entry, 1, 1, 1, 1)

        self.Scale_box = QGroupBox(self.variable_from_to_frame)
        self.Scale_box.setObjectName(u"Scale_box")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy5.setHorizontalStretch(1)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.Scale_box.sizePolicy().hasHeightForWidth())
        self.Scale_box.setSizePolicy(sizePolicy5)
        self.Scale_box.setFont(font)
        self.verticalLayout_3 = QVBoxLayout(self.Scale_box)
        self.verticalLayout_3.setSpacing(1)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(-1, 1, -1, 1)
        self.lin_button = QRadioButton(self.Scale_box)
        self.scale_button_group = QButtonGroup(variable_from_to_scale)
        self.scale_button_group.setObjectName(u"scale_button_group")
        self.scale_button_group.addButton(self.lin_button)
        self.lin_button.setObjectName(u"lin_button")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy6.setHorizontalStretch(1)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.lin_button.sizePolicy().hasHeightForWidth())
        self.lin_button.setSizePolicy(sizePolicy6)
        self.lin_button.setMaximumSize(QSize(16777215, 24))
        self.lin_button.setIconSize(QSize(12, 12))
        self.lin_button.setChecked(True)

        self.verticalLayout_3.addWidget(self.lin_button)

        self.log_button = QRadioButton(self.Scale_box)
        self.scale_button_group.addButton(self.log_button)
        self.log_button.setObjectName(u"log_button")
        sizePolicy6.setHeightForWidth(self.log_button.sizePolicy().hasHeightForWidth())
        self.log_button.setSizePolicy(sizePolicy6)
        self.log_button.setMaximumSize(QSize(16777215, 24))
        self.log_button.setIconSize(QSize(12, 12))

        self.verticalLayout_3.addWidget(self.log_button)

        self.verticalLayout_3.setStretch(0, 2)
        self.verticalLayout_3.setStretch(1, 1)

        self.gridLayout_4.addWidget(self.Scale_box, 0, 3, 2, 1)

        self.gridLayout_4.setRowStretch(0, 1)
        self.gridLayout_4.setColumnStretch(0, 1)

        self.horizontalLayout.addWidget(self.variable_from_to_frame)

        QWidget.setTabOrder(self.Var_dd, self.from_entry)
        QWidget.setTabOrder(self.from_entry, self.to_entry)
        QWidget.setTabOrder(self.to_entry, self.lin_button)
        QWidget.setTabOrder(self.lin_button, self.log_button)

        self.retranslateUi(variable_from_to_scale)
        self.Var_dd.currentTextChanged.connect(variable_from_to_scale.on_var_change)
        self.scale_button_group.buttonClicked.connect(variable_from_to_scale.on_scale_change)
        self.from_entry.textChanged.connect(variable_from_to_scale.on_from_change)
        self.to_entry.textChanged.connect(variable_from_to_scale.on_to_change)

        QMetaObject.connectSlotsByName(variable_from_to_scale)
    # setupUi

    def retranslateUi(self, variable_from_to_scale):
        variable_from_to_scale.setWindowTitle(QCoreApplication.translate("variable_from_to_scale", u"Form", None))
        self.From_l.setText(QCoreApplication.translate("variable_from_to_scale", u"from", None))
        self.Var_l.setText(QCoreApplication.translate("variable_from_to_scale", u"Variable", None))
        self.to_entry.setText(QCoreApplication.translate("variable_from_to_scale", u"1.00", None))
        self.Var_dd.setItemText(0, QCoreApplication.translate("variable_from_to_scale", u"Param 2", None))
        self.Var_dd.setItemText(1, QCoreApplication.translate("variable_from_to_scale", u"Param 1", None))

        self.to_l.setText(QCoreApplication.translate("variable_from_to_scale", u"to", None))
        self.from_entry.setText(QCoreApplication.translate("variable_from_to_scale", u"0.00", None))
        self.Scale_box.setTitle("")
        self.lin_button.setText(QCoreApplication.translate("variable_from_to_scale", u"Linear", None))
        self.log_button.setText(QCoreApplication.translate("variable_from_to_scale", u"Logarithmic", None))
    # retranslateUi

