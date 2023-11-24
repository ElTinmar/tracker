from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout
from .animal import *
from qt_widgets import NDarray_to_QPixmap, LabeledDoubleSpinBox, LabeledSpinBox
import cv2
from geometry import Affine2DTransform
from typing import Any
from numpy.typing import NDArray

# TODO maybe group settings into collapsable blocks


class AnimalTrackerWidget(QWidget):
    def __init__(self, *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        self.tracker = None
        self.declare_components()
        self.layout_components()

    def declare_components(self) -> None:

        self.image = QLabel(self)
        self.mask = QLabel(self)
        self.image_overlay = QLabel(self)
        
        # pix per mm
        self.pix_per_mm = LabeledDoubleSpinBox(self)
        self.pix_per_mm.setText('pixels / mm')
        self.pix_per_mm.setRange(0,1000)
        self.pix_per_mm.setValue(40)
        self.pix_per_mm.setSingleStep(1.0)
        self.pix_per_mm.valueChanged.connect(self.update_tracker)

        # pix per mm
        self.target_pix_per_mm = LabeledDoubleSpinBox(self)
        self.target_pix_per_mm.setText('target pixels / mm')
        self.target_pix_per_mm.setRange(0,1000)
        self.target_pix_per_mm.setValue(7.5)
        self.target_pix_per_mm.setSingleStep(0.5)
        self.target_pix_per_mm.valueChanged.connect(self.update_tracker)

        # animal intensity
        self.animal_intensity = LabeledDoubleSpinBox(self)
        self.animal_intensity.setText('animal intensity')
        self.animal_intensity.setRange(0,1)
        self.animal_intensity.setValue(0.07)
        self.animal_intensity.setSingleStep(0.01)
        self.animal_intensity.valueChanged.connect(self.update_tracker)

        # gamma
        self.animal_gamma = LabeledDoubleSpinBox(self)
        self.animal_gamma.setText('animal gamma')
        self.animal_gamma.setRange(0,100)
        self.animal_gamma.setValue(1.0)
        self.animal_gamma.setSingleStep(0.1)
        self.animal_gamma.valueChanged.connect(self.update_tracker) 

        # constrast
        self.animal_contrast = LabeledDoubleSpinBox(self)
        self.animal_contrast.setText('animal contrast')
        self.animal_contrast.setRange(0,100)
        self.animal_contrast.setValue(1.0)
        self.animal_contrast.setSingleStep(0.1)
        self.animal_contrast.valueChanged.connect(self.update_tracker) 

        # brightness
        self.animal_brightness = LabeledDoubleSpinBox(self)
        self.animal_brightness.setText('animal brightness')
        self.animal_brightness.setRange(0,1)
        self.animal_brightness.setValue(1.0)
        self.animal_brightness.setSingleStep(0.025)
        self.animal_brightness.valueChanged.connect(self.update_tracker) 

        # animal size
        self.min_animal_size_mm = LabeledDoubleSpinBox(self)
        self.min_animal_size_mm.setText('min animal size (mm)')
        self.min_animal_size_mm.setRange(0,1000)
        self.min_animal_size_mm.setValue(1.0)
        self.min_animal_size_mm.setSingleStep(0.25)
        self.min_animal_size_mm.valueChanged.connect(self.update_tracker)

        #
        self.max_animal_size_mm = LabeledDoubleSpinBox(self)
        self.max_animal_size_mm.setText('max animal size (mm)')
        self.max_animal_size_mm.setRange(0,1000)
        self.max_animal_size_mm.setValue(30.0)
        self.max_animal_size_mm.setSingleStep(0.5)
        self.max_animal_size_mm.valueChanged.connect(self.update_tracker)

        # animal length
        self.min_animal_length_mm = LabeledDoubleSpinBox(self)
        self.min_animal_length_mm.setText('min animal length (mm)')
        self.min_animal_length_mm.setRange(0,100)
        self.min_animal_length_mm.setValue(1.0)
        self.min_animal_length_mm.setSingleStep(0.25)
        self.min_animal_length_mm.valueChanged.connect(self.update_tracker)

        #
        self.max_animal_length_mm = LabeledDoubleSpinBox(self)
        self.max_animal_length_mm.setText('max animal length (mm)')
        self.max_animal_length_mm.setRange(0,100)
        self.max_animal_length_mm.setValue(12.0)
        self.max_animal_length_mm.setSingleStep(0.25)
        self.max_animal_length_mm.valueChanged.connect(self.update_tracker)

        # animal width
        self.min_animal_width_mm = LabeledDoubleSpinBox(self)
        self.min_animal_width_mm.setText('min animal width (mm)')
        self.min_animal_width_mm.setRange(0,100)
        self.min_animal_width_mm.setValue(0.4)
        self.min_animal_width_mm.setSingleStep(0.05)
        self.min_animal_width_mm.valueChanged.connect(self.update_tracker)

        #
        self.max_animal_width_mm = LabeledDoubleSpinBox(self)
        self.max_animal_width_mm.setText('max animal width (mm)')
        self.max_animal_width_mm.setRange(0,100)
        self.max_animal_width_mm.setValue(2.5)
        self.max_animal_width_mm.setSingleStep(0.05)
        self.max_animal_width_mm.valueChanged.connect(self.update_tracker)

        # pad value  
        self.pad_value_mm = LabeledDoubleSpinBox(self)
        self.pad_value_mm.setText('Bbox size (mm)')
        self.pad_value_mm.setRange(0,10)
        self.pad_value_mm.setValue(4.0)
        self.pad_value_mm.setSingleStep(0.1)
        self.pad_value_mm.valueChanged.connect(self.update_tracker)

        #ksize_blur_mm 
        self.blur_sz_mm = LabeledDoubleSpinBox(self)
        self.blur_sz_mm.setText('blur size (mm)')
        self.blur_sz_mm.setRange(0,1000)
        self.blur_sz_mm.setValue(1/7.5)
        self.blur_sz_mm.setSingleStep(1/7.5)
        self.blur_sz_mm.valueChanged.connect(self.update_tracker)
                
        # median filter size
        self.median_filter_sz_mm = LabeledDoubleSpinBox(self)
        self.median_filter_sz_mm.setText('medfilt size (mm)')
        self.median_filter_sz_mm.setRange(0,1000)
        self.median_filter_sz_mm.setValue(1/7.5)
        self.median_filter_sz_mm.setSingleStep(1/7.5)
        self.median_filter_sz_mm.valueChanged.connect(self.update_tracker)

        self.zoom = LabeledSpinBox(self)
        self.zoom.setText('zoom (%)')
        self.zoom.setRange(0,500)
        self.zoom.setValue(100)
        self.zoom.setSingleStep(25)
        self.zoom.valueChanged.connect(self.update_tracker)

    def layout_components(self) -> None:

        parameters = QVBoxLayout()
        parameters.addWidget(self.pix_per_mm)
        parameters.addWidget(self.target_pix_per_mm)
        parameters.addWidget(self.animal_intensity)
        parameters.addWidget(self.animal_gamma)
        parameters.addWidget(self.animal_contrast)
        parameters.addWidget(self.animal_brightness)
        parameters.addWidget(self.min_animal_size_mm)
        parameters.addWidget(self.max_animal_size_mm)
        parameters.addWidget(self.min_animal_length_mm)
        parameters.addWidget(self.max_animal_length_mm)
        parameters.addWidget(self.min_animal_width_mm)
        parameters.addWidget(self.max_animal_width_mm)
        parameters.addWidget(self.pad_value_mm)   
        parameters.addWidget(self.blur_sz_mm)
        parameters.addWidget(self.median_filter_sz_mm)
        parameters.addStretch()

        images = QVBoxLayout()
        images.addWidget(self.image)
        images.addWidget(self.mask)
        images.addWidget(self.image_overlay)

        images_and_zoom = QVBoxLayout()
        images_and_zoom.addWidget(self.zoom)
        images_and_zoom.addLayout(images)
        images_and_zoom.addStretch()

        mainlayout = QHBoxLayout()
        mainlayout.addLayout(images_and_zoom)
        mainlayout.addLayout(parameters)

        self.setLayout(mainlayout)

    def update_tracker(self) -> None:

        tracker_param = AnimalTrackerParamTracking(
            pix_per_mm = self.pix_per_mm.value(),
            target_pix_per_mm = self.target_pix_per_mm.value(),
            animal_intensity = self.animal_intensity.value(),
            animal_gamma = self.animal_gamma.value(),
            animal_contrast = self.animal_contrast.value(),
            animal_brightness = self.animal_brightness.value(),
            min_animal_size_mm = self.min_animal_size_mm.value(),
            max_animal_size_mm = self.max_animal_size_mm.value(),
            min_animal_length_mm = self.min_animal_length_mm.value(),
            max_animal_length_mm = self.max_animal_length_mm.value(),
            min_animal_width_mm = self.min_animal_width_mm.value(),
            max_animal_width_mm = self.max_animal_width_mm.value(),
            blur_sz_mm = self.blur_sz_mm.value(),
            median_filter_sz_mm = self.median_filter_sz_mm.value(),
            pad_value_mm = self.pad_value_mm.value()
        )
        self.tracker = AnimalTracker(tracker_param)

        overlay_param = AnimalTrackerParamOverlay(
            pix_per_mm=self.target_pix_per_mm.value()
        )
        self.overlay = AnimalOverlay(overlay_param)

    def display(self, tracking: AnimalTracking) -> None:

        if tracking is not None:
            
            s = self.tracker.tracking_param.resize
            T = Affine2DTransform.scaling(s, s)
            overlay = self.overlay.overlay(tracking.image, tracking, T)

            zoom = self.zoom.value()/100.0
            image = cv2.resize(tracking.image,None,None,zoom,zoom,cv2.INTER_NEAREST)
            mask = cv2.resize(tracking.mask,None,None,zoom,zoom,cv2.INTER_NEAREST)
            overlay = cv2.resize(overlay,None,None,zoom,zoom,cv2.INTER_NEAREST)

            self.image.setPixmap(NDarray_to_QPixmap(image))
            self.mask.setPixmap(NDarray_to_QPixmap(mask))
            self.image_overlay.setPixmap(NDarray_to_QPixmap(overlay))
            self.update()
