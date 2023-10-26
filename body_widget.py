from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout
from trackers.body import BodyTracker, BodyTrackerParamOverlay, BodyTrackerParamTracking
from qt_widgets import NDarray_to_QPixmap, LabeledDoubleSpinBox, LabeledSpinBox
import cv2

# TODO maybe group settings into collapsable blocks


class BodyTrackerWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = None
        self.declare_components()
        self.layout_components()

    def declare_components(self):
        self.image = QLabel(self)
        self.mask = QLabel(self)
        self.image_overlay = QLabel(self)
        
        # pix per mm
        self.pix_per_mm = LabeledDoubleSpinBox(self)
        self.pix_per_mm.setText('pixels / mm')
        self.pix_per_mm.setRange(0,1000)
        self.pix_per_mm.setValue(40.0)
        self.pix_per_mm.setSingleStep(1.0)
        self.pix_per_mm.valueChanged.connect(self.update_tracker)

        # pix per mm
        self.target_pix_per_mm = LabeledDoubleSpinBox(self)
        self.target_pix_per_mm.setText('target pixels / mm')
        self.target_pix_per_mm.setRange(0,1000)
        self.target_pix_per_mm.setValue(7.5)
        self.target_pix_per_mm.setSingleStep(0.5)
        self.target_pix_per_mm.valueChanged.connect(self.update_tracker)

        # body intensity
        self.body_intensity = LabeledDoubleSpinBox(self)
        self.body_intensity.setText('body intensity')
        self.body_intensity.setRange(0,1)
        self.body_intensity.setValue(0.06)
        self.body_intensity.setSingleStep(0.01)
        self.body_intensity.valueChanged.connect(self.update_tracker)
    
        # gamma
        self.body_gamma = LabeledDoubleSpinBox(self)
        self.body_gamma.setText('body gamma')
        self.body_gamma.setRange(0,100)
        self.body_gamma.setValue(3.0)
        self.body_gamma.setSingleStep(0.1)
        self.body_gamma.valueChanged.connect(self.update_tracker) 

        # constrast
        self.body_contrast = LabeledDoubleSpinBox(self)
        self.body_contrast.setText('body contrast')
        self.body_contrast.setRange(0,100)
        self.body_contrast.setValue(1.5)
        self.body_contrast.setSingleStep(0.1)
        self.body_contrast.valueChanged.connect(self.update_tracker) 

        # norm
        self.body_norm = LabeledDoubleSpinBox(self)
        self.body_norm.setText('body norm')
        self.body_norm.setRange(0,1)
        self.body_norm.setValue(0.3)
        self.body_norm.setSingleStep(0.025)
        self.body_norm.valueChanged.connect(self.update_tracker) 

        # body size
        self.min_body_size_mm = LabeledDoubleSpinBox(self)
        self.min_body_size_mm.setText('min body size (mm)')
        self.min_body_size_mm.setRange(0,1000)
        self.min_body_size_mm.setValue(2.0)
        self.min_body_size_mm.setSingleStep(0.25)
        self.min_body_size_mm.valueChanged.connect(self.update_tracker)

        #
        self.max_body_size_mm = LabeledDoubleSpinBox(self)
        self.max_body_size_mm.setText('max body size (mm)')
        self.max_body_size_mm.setRange(0,10000)
        self.max_body_size_mm.setValue(30.0)
        self.max_body_size_mm.setSingleStep(0.5)
        self.max_body_size_mm.valueChanged.connect(self.update_tracker)

        # body length
        self.min_body_length_mm = LabeledDoubleSpinBox(self)
        self.min_body_length_mm.setText('min body length (mm)')
        self.min_body_length_mm.setRange(0,100)
        self.min_body_length_mm.setValue(2.0)
        self.min_body_length_mm.setSingleStep(0.25)
        self.min_body_length_mm.valueChanged.connect(self.update_tracker)

        #
        self.max_body_length_mm = LabeledDoubleSpinBox(self)
        self.max_body_length_mm.setText('max body length (mm)')
        self.max_body_length_mm.setRange(0,100)
        self.max_body_length_mm.setValue(6.0)
        self.max_body_length_mm.setSingleStep(0.25)
        self.max_body_length_mm.valueChanged.connect(self.update_tracker)

        # body width
        self.min_body_width_mm = LabeledDoubleSpinBox(self)
        self.min_body_width_mm.setText('min body width (mm)')
        self.min_body_width_mm.setRange(0,100)
        self.min_body_width_mm.setValue(0.4)
        self.min_body_width_mm.setSingleStep(0.05)
        self.min_body_width_mm.valueChanged.connect(self.update_tracker)

        #
        self.max_body_width_mm = LabeledDoubleSpinBox(self)
        self.max_body_width_mm.setText('max body width (mm)')
        self.max_body_width_mm.setRange(0,100)
        self.max_body_width_mm.setValue(1.2)
        self.max_body_width_mm.setSingleStep(0.05)
        self.max_body_width_mm.valueChanged.connect(self.update_tracker)

        #ksize_blur_mm 
        self.blur_sz_mm = LabeledDoubleSpinBox(self)
        self.blur_sz_mm.setText('blur size (mm)')
        self.blur_sz_mm.setRange(0,2)
        self.blur_sz_mm.setValue(1/7.5)
        self.blur_sz_mm.setSingleStep(1/7.5)
        self.blur_sz_mm.valueChanged.connect(self.update_tracker)
                
        # median filter size
        self.median_filter_sz_mm = LabeledDoubleSpinBox(self)
        self.median_filter_sz_mm.setText('medfilt size (mm)')
        self.median_filter_sz_mm.setRange(0,2)
        self.median_filter_sz_mm.setValue(1/7.5)
        self.median_filter_sz_mm.setSingleStep(1/7.5)
        self.median_filter_sz_mm.valueChanged.connect(self.update_tracker)

        self.zoom = LabeledSpinBox(self)
        self.zoom.setText('zoom (%)')
        self.zoom.setRange(0,500)
        self.zoom.setValue(300)
        self.zoom.setSingleStep(25)
        self.zoom.valueChanged.connect(self.update_tracker)

    def layout_components(self):

        parameters = QVBoxLayout()
        parameters.addWidget(self.pix_per_mm)
        parameters.addWidget(self.target_pix_per_mm)
        parameters.addWidget(self.body_intensity)
        parameters.addWidget(self.body_gamma)
        parameters.addWidget(self.body_contrast)
        parameters.addWidget(self.body_norm)
        parameters.addWidget(self.min_body_size_mm)
        parameters.addWidget(self.max_body_size_mm)
        parameters.addWidget(self.min_body_length_mm)
        parameters.addWidget(self.max_body_length_mm)
        parameters.addWidget(self.min_body_width_mm)
        parameters.addWidget(self.max_body_width_mm)
        parameters.addWidget(self.blur_sz_mm)
        parameters.addWidget(self.median_filter_sz_mm)
        parameters.addStretch()

        images = QVBoxLayout()
        images.addWidget(self.zoom)
        images.addWidget(self.image)
        images.addWidget(self.mask)
        images.addWidget(self.image_overlay)
        images.addStretch()

        mainlayout = QHBoxLayout()
        mainlayout.addLayout(images)
        mainlayout.addLayout(parameters)

        self.setLayout(mainlayout)

    def update_tracker(self):
        overlay_param = BodyTrackerParamOverlay(
            pix_per_mm = self.target_pix_per_mm.value(),
        )
        tracker_param = BodyTrackerParamTracking(
            pix_per_mm = self.pix_per_mm.value(),
            target_pix_per_mm = self.target_pix_per_mm.value(),
            body_intensity = self.body_intensity.value(),
            body_gamma = self.body_gamma.value(),
            body_contrast = self.body_contrast.value(),
            body_norm = self.body_norm.value(),
            min_body_size_mm = self.min_body_size_mm.value(),
            max_body_size_mm = self.max_body_size_mm.value(),
            min_body_length_mm = self.min_body_length_mm.value(),
            max_body_length_mm = self.max_body_length_mm.value(),
            min_body_width_mm = self.min_body_width_mm.value(),
            max_body_width_mm = self.max_body_width_mm.value(),
            blur_sz_mm = self.blur_sz_mm.value(),
            median_filter_sz_mm = self.median_filter_sz_mm.value()
        )
        self.tracker = BodyTracker(tracker_param, overlay_param)

    def display(self, tracking):
        if tracking is not None:
            overlay = self.tracker.overlay_local(tracking)

            zoom = self.zoom.value()/100.0
            image = cv2.resize(tracking.image,None,None,zoom,zoom,cv2.INTER_NEAREST)
            mask = cv2.resize(tracking.mask,None,None,zoom,zoom,cv2.INTER_NEAREST)
            overlay = cv2.resize(overlay,None,None,zoom,zoom,cv2.INTER_NEAREST)

            self.image.setPixmap(NDarray_to_QPixmap(image))
            self.mask.setPixmap(NDarray_to_QPixmap(mask))
            self.image_overlay.setPixmap(NDarray_to_QPixmap(overlay))
            self.update()
