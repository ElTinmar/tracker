from PyQt5.QtWidgets import QWidget, QLabel, QStackedWidget, QVBoxLayout
from .assignment import LinearSumAssignment, GridAssignment
from qt_widgets import LabeledDoubleSpinBox, LabeledComboBox, NDarray_to_QPixmap
from image_tools import ROISelectorDialog
from numpy.typing import NDArray
import numpy as np
from typing import Optional

#TODO make it accept different image formats than single precision

class AssignmentWidget(QWidget):
    
    def __init__(self, background_image: Optional[NDArray] = None, *args, **kwargs):
        # NOTE THIS IS EXPECTING A SINGLE PRECISION IMAGE
        
        super().__init__(*args, **kwargs)
        
        self.assignment = None
        self.background_image = None
        if background_image is not None:
            self.background_image = (255*background_image).astype(np.uint8)
        self.grid_dialog = ROISelectorDialog(image = self.background_image)
        self.declare_components()
        self.layout_components()
        self.on_assignment_update()

    def set_background_image(self, height, width, background_image_bytes: bytes) -> None:
        # NOTE THIS IS EXPECTING A SINGLE PRECISION IMAGE

        image = np.frombuffer(background_image_bytes, dtype=np.float32).reshape(height,width)
        self.background_image = (255*image).astype(np.uint8)
        self.grid_dialog = ROISelectorDialog(image = self.background_image)
    
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
            while not self.grid_dialog.exec_():
                print('select ROIs and click on done')
            rois = self.grid_dialog.ROIs
            LUT = np.zeros_like(self.background_image) 
            for id, rect in enumerate(rois):
                LUT[rect.top():rect.top()+rect.height(), rect.left():rect.left()+rect.width()] = id 
            #TODO update background image to show LUT
            self.assignment = GridAssignment(
                LUT = LUT
            )
    
    def get_assignment(self):
        return self.assignment
