from PyQt5.QtWidgets import QWidget, QLabel, QStackedWidget, QVBoxLayout
from .assignment import LinearSumAssignment, GridAssignment
from .qt_widgets import LabeledDoubleSpinBox, LabeledComboBox, NDarray_to_QPixmap
from numpy.typing import NDArray
import numpy as np
from typing import Optional

class AssignmentWidget(QWidget):
    
    def __init__(self, background_image: Optional[NDArray] = None, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.assignment = None
        self.background_image = background_image 
        self.declare_components()
        self.layout_components()
    
    def declare_components(self):
        
        self.assignment_method_combobox = LabeledComboBox(self)
        self.assignment_method_combobox.setText('ID assignment')
        self.assignment_method_combobox.addItem('linear sum')
        self.assignment_method_combobox.addItem('grid')
        self.assignment_method_combobox.currentIndexChanged.connect(self.on_method_change)

        # linear sum 
        self.parameters_linearsum = QWidget()
        self.max_distance_spinbox = LabeledDoubleSpinBox(self.parameters_linearsum)
        self.max_distance_spinbox.setText('distance')
        self.max_distance_spinbox.setRange(0,1000)
        self.max_distance_spinbox.setValue(10.0)
        self.max_distance_spinbox.valueChanged.connect(self.on_assignment_update)

        # grid
        self.parameters_grid = QWidget()
        self.background_image_label = QLabel(self.parameters_grid)
        if self.background_image is not None:
            self.background_image_label.setPixmap(NDarray_to_QPixmap(self.background_image))

        # stack
        self.assignment_parameter_stack = QStackedWidget(self)
        self.assignment_parameter_stack.addWidget(self.parameters_linearsum)
        self.assignment_parameter_stack.addWidget(self.parameters_grid)

    def layout_components(self):
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.assignment_method_combobox)
        main_layout.addWidget(self.assignment_parameter_stack)

        linearsum_layout = QVBoxLayout(self.parameters_linearsum)
        linearsum_layout.addWidget(self.max_distance_spinbox)
        linearsum_layout.addStretch()

        grid_layout = QVBoxLayout(self.parameters_grid)
        grid_layout.addWidget(self.background_image_label)
        grid_layout.addStretch()

    def on_method_change(self, index):
        self.assignment_parameter_stack.setCurrentIndex(index)
        self.on_assignment_update()

    def on_assignment_update(self):
        
        method = self.assignment_method_combobox.currentIndex()
        if method == 0:
            self.assignment = LinearSumAssignment(
                self.max_distance_spinbox.value()
            )
        elif method == 1:
            LUT = np.zeros((10,10)) # todo
            self.assignment = GridAssignment(
                LUT = LUT
            )
