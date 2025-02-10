# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'slider_entry_animate.ui'
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
from qtpy.QtWidgets import (QApplication, QHBoxLayout, QLabel, QSizePolicy,
    QSlider, QToolButton, QWidget)

from gui.resources.widgets.floatLineEdit import floatLineEdit

class Ui_slider_entry_animate(object):
    def setupUi(self, slider_entry_animate):
        if not slider_entry_animate.objectName():
            slider_entry_animate.setObjectName(u"slider_entry_animate")
        slider_entry_animate.resize(400, 63)
        self.horizontalLayout = QHBoxLayout(slider_entry_animate)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(slider_entry_animate)
        self.label.setObjectName(u"label")
        font = QFont()
        font.setPointSize(12)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.label.setFont(font)

        self.horizontalLayout.addWidget(self.label)

        self.slider = QSlider(slider_entry_animate)
        self.slider.setObjectName(u"slider")
        self.slider.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout.addWidget(self.slider)

        self.entry = floatLineEdit(slider_entry_animate)
        self.entry.setObjectName(u"entry")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.entry.sizePolicy().hasHeightForWidth())
        self.entry.setSizePolicy(sizePolicy)

        self.horizontalLayout.addWidget(self.entry)

        self.animate_button = QToolButton(slider_entry_animate)
        self.animate_button.setObjectName(u"animate_button")
        self.animate_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.animate_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.animate_button.setAutoRaise(False)
        self.animate_button.setArrowType(Qt.ArrowType.NoArrow)

        self.horizontalLayout.addWidget(self.animate_button)

        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(1, 2)
        self.horizontalLayout.setStretch(2, 1)
        self.horizontalLayout.setStretch(3, 2)

        self.retranslateUi(slider_entry_animate)
        self.slider.valueChanged.connect(slider_entry_animate.slider_change)
        self.slider.sliderMoved.connect(slider_entry_animate.speed_change)
        self.entry.textChanged.connect(slider_entry_animate.entry_updated)
        self.animate_button.released.connect(slider_entry_animate.animate_button_press)

        QMetaObject.connectSlotsByName(slider_entry_animate)
    # setupUi

    def retranslateUi(self, slider_entry_animate):
        slider_entry_animate.setWindowTitle(QCoreApplication.translate("slider_entry_animate", u"Form", None))
        self.label.setText(QCoreApplication.translate("slider_entry_animate", u"Free Parameter", None))
        self.animate_button.setText(QCoreApplication.translate("slider_entry_animate", u"Animate", None))
    # retranslateUi

