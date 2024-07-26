from qtpy.QtWidgets import QLineEdit
from qtpy.QtGui import QDoubleValidator
from qtpy.QtCore import Property
import sys

class floatLineEdit(QLineEdit):
    """
    QLineEdit which validates floating point inputs. Default range is
    (-sys_floatmax / 2, sys_floatmax / 2), with 10 decimal places.
    Range can be updated through properties in Qt Designer or in code.

    Attributes:
        _min_value (float): Minimum value for the validator.
        _max_value (float): Maximum value for the validator.
        _decimals (int): Number of decimal places for the validator.
    """

    def __init__(self, parent=None):
        """
        Initializes the floatLineEdit with default values and sets up the validator.

        Args:
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super(floatLineEdit, self).__init__(parent)
        self._min_value = -sys.float_info.max / 2
        self._max_value = sys.float_info.max / 2
        self._decimals = 10
        self.updateValidator()

    def updateValidator(self):
        """Updates the QDoubleValidator with the current range and decimal places."""
        validator = QDoubleValidator(self._min_value, self._max_value, self._decimals, self)
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.setValidator(validator)

    def getMinValue(self):
        """Gets the minimum value of the validator.

        Returns:
            float: The minimum value.
        """
        return self._min_value

    def setMinValue(self, value):
        """Sets the minimum value of the validator.

        Args:
            value (float): The new minimum value.
        """
        self._min_value = value
        self.updateValidator()

    def getMaxValue(self):
        """Gets the maximum value of the validator.

        Returns:
            float: The maximum value.
        """
        return self._max_value

    def setMaxValue(self, value):
        """Sets the maximum value of the validator.

        Args:
            value (float): The new maximum value.
        """
        self._max_value = value
        self.updateValidator()

    def getDecimals(self):
        """Gets the number of decimal places of the validator.

        Returns:
            int: The number of decimal places.
        """
        return self._decimals

    def setDecimals(self, value):
        """Sets the number of decimal places of the validator.

        Args:
            value (int): The new number of decimal places.
        """
        self._decimals = value
        self.updateValidator()

    minValue = Property(float, getMinValue, setMinValue)
    maxValue = Property(float, getMaxValue, setMaxValue)
    decimals = Property(int, getDecimals, setDecimals)
