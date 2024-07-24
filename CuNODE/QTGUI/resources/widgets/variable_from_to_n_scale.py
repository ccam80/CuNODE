# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'variable_from_to_n_scale.ui'
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
from qtpy.QtWidgets import (QAbstractScrollArea, QApplication, QButtonGroup, QComboBox,
    QFrame, QGridLayout, QGroupBox, QLabel,
    QLayout, QRadioButton, QSizePolicy, QTextEdit,
    QVBoxLayout, QWidget)

class Ui_variable_from_to_n_scale(object):
    def setupUi(self, variable_from_to_n_scale):
        if not variable_from_to_n_scale.objectName():
            variable_from_to_n_scale.setObjectName(u"variable_from_to_n_scale")
        variable_from_to_n_scale.resize(516, 94)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(variable_from_to_n_scale.sizePolicy().hasHeightForWidth())
        variable_from_to_n_scale.setSizePolicy(sizePolicy)
        self.verticalLayout = QVBoxLayout(variable_from_to_n_scale)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.variable_from_to_frame = QFrame(variable_from_to_n_scale)
        self.variable_from_to_frame.setObjectName(u"variable_from_to_frame")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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
        self.Var_l = QLabel(self.variable_from_to_frame)
        self.Var_l.setObjectName(u"Var_l")
        font = QFont()
        font.setPointSize(12)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.Var_l.setFont(font)

        self.gridLayout_4.addWidget(self.Var_l, 0, 0, 1, 1)

        self.Var_dd = QComboBox(self.variable_from_to_frame)
        self.Var_dd.addItem("")
        self.Var_dd.addItem("")
        self.Var_dd.setObjectName(u"Var_dd")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        sizePolicy2.setHorizontalStretch(1)
        sizePolicy2.setVerticalStretch(1)
        sizePolicy2.setHeightForWidth(self.Var_dd.sizePolicy().hasHeightForWidth())
        self.Var_dd.setSizePolicy(sizePolicy2)
        self.Var_dd.setFont(font)

        self.gridLayout_4.addWidget(self.Var_dd, 1, 0, 1, 1)

        self.From_l = QLabel(self.variable_from_to_frame)
        self.From_l.setObjectName(u"From_l")
        self.From_l.setFont(font)

        self.gridLayout_4.addWidget(self.From_l, 0, 1, 1, 1)

        self.from_entry = QTextEdit(self.variable_from_to_frame)
        self.from_entry.setObjectName(u"from_entry")
        self.from_entry.setEnabled(True)
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        sizePolicy3.setHorizontalStretch(1)
        sizePolicy3.setVerticalStretch(1)
        sizePolicy3.setHeightForWidth(self.from_entry.sizePolicy().hasHeightForWidth())
        self.from_entry.setSizePolicy(sizePolicy3)
        self.from_entry.setMaximumSize(QSize(16777215, 40))
        font1 = QFont()
        font1.setPointSize(10)
        font1.setStrikeOut(False)
        font1.setKerning(True)
        self.from_entry.setFont(font1)
        self.from_entry.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)

        self.gridLayout_4.addWidget(self.from_entry, 1, 1, 1, 1)

        self.Scale_box = QGroupBox(self.variable_from_to_frame)
        self.Scale_box.setObjectName(u"Scale_box")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy4.setHorizontalStretch(1)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.Scale_box.sizePolicy().hasHeightForWidth())
        self.Scale_box.setSizePolicy(sizePolicy4)
        self.Scale_box.setFont(font)
        self.verticalLayout_3 = QVBoxLayout(self.Scale_box)
        self.verticalLayout_3.setSpacing(1)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(-1, 1, -1, 1)
        self.lin_button = QRadioButton(self.Scale_box)
        self.scale_button_group = QButtonGroup(variable_from_to_n_scale)
        self.scale_button_group.setObjectName(u"scale_button_group")
        self.scale_button_group.addButton(self.lin_button)
        self.lin_button.setObjectName(u"lin_button")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy5.setHorizontalStretch(1)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.lin_button.sizePolicy().hasHeightForWidth())
        self.lin_button.setSizePolicy(sizePolicy5)
        self.lin_button.setMaximumSize(QSize(16777215, 24))
        self.lin_button.setChecked(True)

        self.verticalLayout_3.addWidget(self.lin_button)

        self.log_button = QRadioButton(self.Scale_box)
        self.scale_button_group.addButton(self.log_button)
        self.log_button.setObjectName(u"log_button")
        sizePolicy5.setHeightForWidth(self.log_button.sizePolicy().hasHeightForWidth())
        self.log_button.setSizePolicy(sizePolicy5)
        self.log_button.setMaximumSize(QSize(16777215, 24))

        self.verticalLayout_3.addWidget(self.log_button)

        self.verticalLayout_3.setStretch(0, 2)
        self.verticalLayout_3.setStretch(1, 1)

        self.gridLayout_4.addWidget(self.Scale_box, 0, 4, 2, 1)

        self.to_entry = QTextEdit(self.variable_from_to_frame)
        self.to_entry.setObjectName(u"to_entry")
        sizePolicy3.setHeightForWidth(self.to_entry.sizePolicy().hasHeightForWidth())
        self.to_entry.setSizePolicy(sizePolicy3)
        self.to_entry.setMaximumSize(QSize(16777215, 40))
        self.to_entry.setFont(font1)
        self.to_entry.setAcceptRichText(False)

        self.gridLayout_4.addWidget(self.to_entry, 1, 2, 1, 1)

        self.to_l = QLabel(self.variable_from_to_frame)
        self.to_l.setObjectName(u"to_l")
        self.to_l.setFont(font)

        self.gridLayout_4.addWidget(self.to_l, 0, 2, 1, 1)

        self.n_entry = QTextEdit(self.variable_from_to_frame)
        self.n_entry.setObjectName(u"n_entry")
        sizePolicy3.setHeightForWidth(self.n_entry.sizePolicy().hasHeightForWidth())
        self.n_entry.setSizePolicy(sizePolicy3)
        self.n_entry.setMaximumSize(QSize(16777215, 40))

        self.gridLayout_4.addWidget(self.n_entry, 1, 3, 1, 1)

        self.n_l = QLabel(self.variable_from_to_frame)
        self.n_l.setObjectName(u"n_l")
        font2 = QFont()
        font2.setPointSize(12)
        self.n_l.setFont(font2)

        self.gridLayout_4.addWidget(self.n_l, 0, 3, 1, 1)

        self.gridLayout_4.setRowStretch(0, 1)
        self.gridLayout_4.setColumnStretch(0, 1)
        self.gridLayout_4.setColumnStretch(1, 1)
        self.gridLayout_4.setColumnStretch(2, 1)
        self.gridLayout_4.setColumnStretch(3, 1)

        self.verticalLayout.addWidget(self.variable_from_to_frame)


        self.retranslateUi(variable_from_to_n_scale)
        self.Var_dd.currentTextChanged.connect(variable_from_to_n_scale.on_var_change)
        self.from_entry.textChanged.connect(variable_from_to_n_scale.on_from_change)
        self.to_entry.textChanged.connect(variable_from_to_n_scale.on_to_change)
        self.scale_button_group.buttonClicked.connect(variable_from_to_n_scale.on_scale_change)
        self.n_entry.textChanged.connect(variable_from_to_n_scale.on_n_change)

        QMetaObject.connectSlotsByName(variable_from_to_n_scale)
    # setupUi

    def retranslateUi(self, variable_from_to_n_scale):
        variable_from_to_n_scale.setWindowTitle(QCoreApplication.translate("variable_from_to_n_scale", u"Form", None))
        self.Var_l.setText(QCoreApplication.translate("variable_from_to_n_scale", u"Variable", None))
        self.Var_dd.setItemText(0, QCoreApplication.translate("variable_from_to_n_scale", u"Param 2", None))
        self.Var_dd.setItemText(1, QCoreApplication.translate("variable_from_to_n_scale", u"Param 1", None))

        self.From_l.setText(QCoreApplication.translate("variable_from_to_n_scale", u"from", None))
        self.Scale_box.setTitle(QCoreApplication.translate("variable_from_to_n_scale", u"Scale", None))
        self.lin_button.setText(QCoreApplication.translate("variable_from_to_n_scale", u"Linear", None))
        self.log_button.setText(QCoreApplication.translate("variable_from_to_n_scale", u"Logarithmic", None))
        self.to_l.setText(QCoreApplication.translate("variable_from_to_n_scale", u"to", None))
        self.n_l.setText(QCoreApplication.translate("variable_from_to_n_scale", u"# points", None))
    # retranslateUi

