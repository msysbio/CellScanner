from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QComboBox, QLabel, QLineEdit, QDoubleSpinBox,
    QSpinBox, QVBoxLayout, QCheckBox
)

class AxisSelector(QWidget):
    def __init__(self, label_text, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        self.label = QLabel(label_text, self)
        self.combo = QComboBox(self)
        layout.addWidget(self.label)
        layout.addWidget(self.combo)
        self.setLayout(layout)  # Set the layout for this widget
    def set_items(self, items):
        self.combo.clear()
        self.combo.addItems(items)


class StainSelector(QWidget):
    def __init__(self, label_text, tooltip_text, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)

        self.label = QLabel(label_text, self)
        self.combo = QComboBox(self)
        self.combo.setToolTip(tooltip_text)
        self.combo.addItem("Not applicable")

        self.relation = QComboBox(self)
        self.relation.addItems(['>', '<'])

        self.threshold = QLineEdit(self)
        self.threshold.setPlaceholderText(
            "Enter threshold. All events where the threshold is met will be classified as dead."
        )

        layout.addWidget(self.label)
        layout.addWidget(self.combo)
        layout.addWidget(self.relation)
        layout.addWidget(self.threshold)

        self.setLayout(layout)

    def set_items(self, items):
        self.combo.clear()
        self.combo.addItem("Not applicable")  # Keep default
        self.combo.addItems(items)


class LabeledComboBox(QWidget):

    def __init__(self, label_text, items=None, default=None, parent=None):
        super().__init__(parent)

        # Layout
        layout = QHBoxLayout(self)

        # Label
        self.label = QLabel(label_text, self)
        layout.addWidget(self.label)

        # ComboBox
        self.combo = QComboBox(self)
        if items:
            self.combo.addItems(items)
        if default and default in items:
            self.combo.setCurrentText(default)

        layout.addWidget(self.combo)

        self.setLayout(layout)


class LabeledSpinBox(QWidget):
    def __init__(self, label_text, min_value=0, max_value=1000, step=1, default_value=0, parent=None):
        super().__init__(parent)

        # Create a horizontal layout for the widget
        layout = QHBoxLayout(self)

        # Create and add the label
        self.label = QLabel(label_text, self)
        layout.addWidget(self.label)

        # Create and configure the spin box
        self.spin_box = QSpinBox(self)
        self.spin_box.setRange(min_value, max_value)  # Set minimum and maximum values
        self.spin_box.setSingleStep(step)              # Set step size
        self.spin_box.setValue(default_value)          # Set default value
        layout.addWidget(self.spin_box)

        # Set the layout on the widget
        self.setLayout(layout)


def iterate_stains(self):
    """
    Return a dictionary with the extra stains settings as provided by the user in CS GUI
    """
    extra_stains = {}
    for i in range(self.gating_layout.count()):
        item = self.gating_layout.itemAt(i)
        widget = item.widget()
        if widget:
            # Access individual components within the stain container
            stain_layout = widget.layout()
            if stain_layout:
                # Iterate through components of this layout
                for j in range(stain_layout.count()):
                    # Get component's features
                    component = stain_layout.itemAt(j).widget()
                    # Check the different type members of the stain layout
                    if isinstance(component, QComboBox):
                        if component.currentText() in [">", "<"]:
                            sign = component.currentText()
                        else:
                            channel = component.currentText()
                    elif isinstance(component, QLineEdit):
                        label = component.text()
                    elif isinstance(component, QDoubleSpinBox):
                        threshold = component.value()
                extra_stains[channel] = (sign, threshold, label)
    return extra_stains


# Mixin classes
class GatingMixin:
    """
    A mixin that provides gating-related UI functionality.
    A thread on mixin:
    https://stackoverflow.com/questions/533631/what-is-a-mixin-and-why-is-it-useful

    This mixin defines the `toggle_gating_options` method, which shows or hides
    UI elements related to gating based on the state of a checkbox.
    """
    def toggle_gating_options(self):
        is_checked = self.gating_checkbox.isChecked()

        for selector in self.stain_selectors:
            selector.label.setVisible(is_checked)
            selector.combo.setVisible(is_checked)
            selector.relation.setVisible(is_checked)
            selector.threshold.setVisible(is_checked)
            self.threshold_message.setVisible(is_checked)
        try:
            self.new_stain_button.setVisible(is_checked)
        except:
            print("No need for extra stain at the training step")
            pass

class GatingCheckBox:
    """
    """
    def gating_checkbox(self):
        # Add a checkbox to apply gating
        self.gating_layout =  QVBoxLayout()
        self.gating_checkbox = QCheckBox("Apply line gating", self)
        self.gating_checkbox.setToolTip(GuiMessages.GATING_CHECHBOX)
        self.gating_layout.addChildWidget(self.gating_checkbox)
        self.gating_checkbox.stateChanged.connect(self.toggle_gating_options)
        try:
            self.predict_panel_layout.addWidget(self.gating_checkbox)
        except:
            self.train_gating_layout.addWidget(self.gating_checkbox)

        # Add message for strain thresholds
        self.thresholds_layout = QHBoxLayout()
        self.threshold_message = QLabel(
            GuiMessages.GATING_THRESHOLD,
            self
        )
        self.thresholds_layout.addWidget(self.threshold_message)
        try:
            self.predict_panel_layout.addLayout(self.thresholds_layout)
        except:
            self.train_gating_layout.addLayout(self.thresholds_layout)



class LiveDeadDebrisSelectors:
    """
    Mixin class to display the 2 basic stains for live/dead and cells/debris entries.

    """
    def basic_stains(self):

        # Stain 1 selection (for live/dead)
        tooltip_for_stain_1 = (
            "Select the channel that will be used for gating live/dead cells. "
            "All events where the threshold is met will be classified as dead."
        )
        # Stain 2 selection (for debris, optional)
        tooltip_for_stain_2 = (
            "Select the channel that will be used for gating all cells. "
            "All events where the threshold is met will be classified as cells. "
            "The rest of the events will be classified as debris."
        )

        # Pair of basic stains
        self.stain1_selector = StainSelector("Staining inactive cells (e.g. PI):", tooltip_for_stain_1, self)
        self.stain2_selector = StainSelector("Staining all cells (e.g. SYBR/DAPI):", tooltip_for_stain_2, self)

        try:
            self.predict_panel_layout.addWidget(self.stain1_selector)
            self.predict_panel_layout.addWidget(self.stain2_selector)
        except:
            self.train_gating_layout.addWidget(self.stain1_selector)
            self.train_gating_layout.addWidget(self.stain2_selector)

        self.stain_selectors = [
            self.stain1_selector,
            self.stain2_selector
        ]



# ToolTips
class GuiMessages:

    UNCERTAINTY_TOOLTIP = (
        "Set threshold for filtering out uncertain predictions. "
        "If you just trained a model, CellScanner computed already the threshold allowing the highest accuracy and set it as default. "
        "If you are loading a model, you can set the threshold manually, and if you are using a previously trained model, "
        "you can use its corresponding model_statistics file to remember the threshold suggested. "
        "To use the widely used threshold of 0.5 of the maximum entropy, set this value to -1.0 and CellScanner will apply this."
    )

    UNCERTAINTY_CHECKBOX = "Apply filtering on the predictions based on their uncertainty scores."

    USER_STAIN_TOOLTIP = (
        "Select the channel that will be used for gating cells. "
        "All events where the threshold is met will be classified according to the label you provide."
    )

    AXIS_SELECTION = "Choose the Channels that will be used as x, y, z axis for the 3D plot:"


    GATING_CHECHBOX = (
        "When staining for both inactive and total cells, CellScanner will also return"
        "the living cells, by combining findings from these 2 stains."
    )
    GATING_THRESHOLD = (
        "Important: Some visualization software may transform raw data."
        "Ensure you set the threshold based on the raw data, not post-transformation."
    )


