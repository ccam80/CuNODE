# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:58:11 2024

@author: cca79
"""
from qtpy.QtWidgets import QFrame, QAbstractButton
from qtpy.QtCore import Slot, Signal, Property
from gui.resources.widgets.qtdesigner.variable_from_to_n_scale import Ui_variable_from_to_n_scale
import logging

class variable_from_to_n_scale_widget(QFrame, Ui_variable_from_to_n_scale):
    """Contains a drop down menu to select a variable, set some "from", "to",
    and "# points" values, and set linear/log scale.

    Signals:
        variable_changed (str): Emitted when the variable is changed.
        from_changed (str): Emitted when the 'from' value is changed.
        to_changed (str): Emitted when the 'to' value is changed.
        scale_changed (str): Emitted when the scale is changed.
        n_changed (str): Emitted when the number of points is changed.

    Slots:
        append_to_variable_list (str): Adds an item to the variable list.
        clear_variable_list (): Clears the variable list.
        edit_from (str): Edits the 'from' value.
        edit_to (str): Edits the 'to' value.
    """
    variable_changed = Signal(str)
    from_changed = Signal(str, int)
    to_changed = Signal(str, int)
    scale_changed = Signal(str)
    n_changed = Signal(str)

    def __init__(self, parent=None):
        super(variable_from_to_n_scale_widget, self).__init__(parent)
        self.setupUi(self)


    @Slot(str)
    def on_var_change(self, new_value):
        var = self.Var_dd.currentText()
        self.variable_changed.emit(var)
        logging.debug(f"Variable changed to: {var}")

    @Slot(str)
    def on_from_change(self, _from):
        self.from_changed.emit(_from, 0)
        logging.debug(f"From changed to: {_from}")

    @Slot(str)
    def on_to_change(self, _to):
        self.to_changed.emit(_to, 1)
        logging.debug(f"To changed to: {_to}")

    @Slot(QAbstractButton)
    def on_scale_change(self, button):
        scale = button.text()
        self.variable_changed.emit(scale)
        logging.debug(f"scale changed to: {scale}")

    @Slot(str)
    def on_n_change(self, n):
        self.n_changed.emit(n)
        logging.debug(f"To changed to: {n}")


    @Slot(str)
    def edit_from(self, text):
        self.from_entry.setText(text)
        pass

    @Slot(str)
    def edit_to(self, text):
        self.to_entry.setText(text)
        pass

    @Slot(str)
    def edit_n(self, text):
        self.n_entry.setText(text)
        pass


    #*************************************************************************
    # This section is setters/getters for floatLineEdit properties, allowing
    # Max/min/decimal values to be set in QtDesigner
    #*************************************************************************

    def setVariableMin(self, _min):
         """
         Set the minimum value for the from_entry and to_entry floatLineEdit.

         Args:
             value (float): The minimum value to set.
         """
         self.from_entry.setMinValue(_min)
         self.to_entry.setMinValue(_min)

    def getVariableMin(self):
         """
         Get the minimum value for the from_entry floatLineEdit.

         Returns:
             float: The minimum value.
         """
         return self.from_entry.getMinValue()

    variableMin = Property(float, getVariableMin, setVariableMin)

    def setVariableMax(self, _max):
         """
         Set the maximum value for the from_entry and to_entry floatLineEdit.

         Args:
             value (float): The maximum value to set.
         """
         self.from_entry.setMaxValue(_max)
         self.to_entry.setMaxValue(_max)

    def getVariableMax(self):
         """
         Get the maximum value for the from_entry and to_entry floatLineEdit.

         Returns:
             float: The maximum value.
         """
         return self.from_entry.getMaxValue()

    variableMax = Property(float, getVariableMax, setVariableMax)

    def setVariableDecimals(self, decimals):
         """
         Set the number of decimal places for the from_entry and to_entry floatLineEdit.

         Args:
             to (int): The number of decimal places to set.
         """
         self.to_entry.setDecimals(decimals)
         self.from_entry.setDecimals(decimals)

    def getVariableDecimals(self):
         """
         Get the number of decimal places for the from_entry and to_entry floatLineEdit.

         Returns:
             int: The number of decimal places.
         """
         return self.to_entry.getDecimals()

    variableDecimals = Property(int, getVariableDecimals, setVariableDecimals)

    def setToEntryDefault(self, default):
         """
         Set the default text for the to_entry floatLineEdit.

         Args:
             text (str): The default text to set.
         """
         self.to_entry.setText(default)

    def getToEntryDefault(self, default):
         """
         Get the default text for the to_entry floatLineEdit.

         Returns:
             str: The default text.
         """
         return self.to_entry.text()

    toEntryDefault = Property(str, getToEntryDefault, setToEntryDefault)

    def setFromEntryDefault(self, default):
         """
         Set the default text for the from_entry floatLineEdit.

         Args:
             text (str): The default text to set.
         """
         self.from_entry.setText(default)

    def getFromEntryDefault(self, default):
        """
        Get the default text for the from_entry floatLineEdit.

        Returns:
            str: The default text.
        """
        return self.from_entry.text()

    fromEntryDefault = Property(str, getToEntryDefault, setToEntryDefault)

    def setNEntryMinValue(self, _min):
        """
        Set the minimum value for the n_entry floatLineEdit.

        Args:
            n (float): The minimum value to set.
        """
        self.n_entry.setMinValue(_min)

    def getNEntryMinValue(self):
        """
        Get the minimum value for the n_entry floatLineEdit.

        Returns:
            float: The minimum value.
        """
        return self.n_entry.getMinValue()

    nEntryMinValue = Property(float, getNEntryMinValue, setNEntryMinValue)

    def setNEntryMaxValue(self,_max):
        """
        Set the maximum value for the n_entry floatLineEdit.

        Args:
            n (float): The maximum value to set.
        """
        self.n_entry.setMaxValue(_max)

    def getNEntryMaxValue(self):
        """
        Get the maximum value for the n_entry floatLineEdit.

        Returns:
            float: The maximum value.
        """
        return self.n_entry.getMaxValue()

    nEntryMaxValue = Property(float, getNEntryMaxValue, setNEntryMaxValue)

    def setNEntryDecimals(self, decimals):
        """
        Set the number of decimal places for the n_entry floatLineEdit.

        Args:
            n (int): The number of decimal places to set.
        """
        self.n_entry.setDecimals(decimals)

    def getNEntryDecimals(self):
        """
        Get the number of decimal places for the n_entry floatLineEdit.

        Returns:
            int: The number of decimal places.
        """
        return self.n_entry.getDecimals()

    nEntryDecimals = Property(int, getNEntryDecimals, setNEntryDecimals)

    def setNEntryDefault(self, default):
        """
        Set the default text for the n_entry floatLineEdit.

        Args:
            text (str): The default text to set.
        """
        self.n_entry.setText(default)

    def getNEntryDefault(self):
        """
        Get the default text for the n_entry floatLineEdit.

        Returns:
            str: The default text.
        """
        return self.n_entry.text()

    nEntryDefault = Property(str, getNEntryDefault, setNEntryDefault)

    def setVarDdItems(self, items):
        """
        Set the items for the Var_dd QComboBox.

        Args:
            items (list): The list of strings to set as items in the combo box.
        """
        self.Var_dd.clear()
        self.Var_dd.addItems(items)

    def getVarDdItems(self):
        """
        Get the items of the Var_dd QComboBox.

        Returns:
            list: The list of items in the combo box.
        """
        return [self.Var_dd.itemText(i) for i in range(self.Var_dd.count())]

    varDdItems = Property(list, getVarDdItems, setVarDdItems)
