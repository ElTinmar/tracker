from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout
from trackers.eyes import EyesTracker, EyesTrackerParamOverlay, EyesTrackerParamTracking
from qt_widgets import NDarray_to_QPixmap, LabeledDoubleSpinBox, LabeledSpinBox
import cv2

# TODO maybe group settings into collapsable blocks

class EyesTrackerWidget(QWidget):
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
        self.pix_per_mm.setText('pix_per_mm')
        self.pix_per_mm.setRange(0,1000)
        self.pix_per_mm.setValue(40)
        self.pix_per_mm.setSingleStep(1.0)
        self.pix_per_mm.valueChanged.connect(self.update_tracker) 

        # target pix per mm
        self.target_pix_per_mm = LabeledDoubleSpinBox(self)
        self.target_pix_per_mm.setText('target pix_per_mm')
        self.target_pix_per_mm.setRange(0,1000)
        self.target_pix_per_mm.setValue(40)
        self.target_pix_per_mm.setSingleStep(1.0)
        self.target_pix_per_mm.valueChanged.connect(self.update_tracker) 

        # eye gamma
        self.eye_gamma = LabeledDoubleSpinBox(self)
        self.eye_gamma.setText('eye gamma')
        self.eye_gamma.setRange(0,100)
        self.eye_gamma.setValue(3.0)
        self.eye_gamma.setSingleStep(0.1)
        self.eye_gamma.valueChanged.connect(self.update_tracker) 

        # eye constrast
        self.eye_contrast = LabeledDoubleSpinBox(self)
        self.eye_contrast.setText('eye contrast')
        self.eye_contrast.setRange(0,100)
        self.eye_contrast.setValue(1.5)
        self.eye_contrast.setSingleStep(0.1)
        self.eye_contrast.valueChanged.connect(self.update_tracker) 

        # eye norm
        self.eye_norm = LabeledDoubleSpinBox(self)
        self.eye_norm.setText('eye norm')
        self.eye_norm.setRange(0,1)
        self.eye_norm.setValue(0.3)
        self.eye_norm.setSingleStep(0.025)
        self.eye_norm.valueChanged.connect(self.update_tracker) 

        # eye dynthresh
        self.eye_dyntresh_res = LabeledSpinBox(self)
        self.eye_dyntresh_res.setText('dyntresh res')
        self.eye_dyntresh_res.setRange(0,100)
        self.eye_dyntresh_res.setValue(20)
        self.eye_dyntresh_res.setSingleStep(1)
        self.eye_dyntresh_res.valueChanged.connect(self.update_tracker) 

        # eye size
        self.eye_size_lo_mm = LabeledDoubleSpinBox(self)
        self.eye_size_lo_mm.setText('min. eye size')
        self.eye_size_lo_mm.setRange(0,100)
        self.eye_size_lo_mm.setValue(0.8)
        self.eye_size_lo_mm.setSingleStep(0.05)
        self.eye_size_lo_mm.valueChanged.connect(self.update_tracker) 

        self.eye_size_hi_mm = LabeledDoubleSpinBox(self)
        self.eye_size_hi_mm.setText('max. eye size')
        self.eye_size_hi_mm.setRange(0,100)
        self.eye_size_hi_mm.setValue(10.0)
        self.eye_size_hi_mm.setSingleStep(1.0)
        self.eye_size_hi_mm.valueChanged.connect(self.update_tracker) 

        # crop_dimension_mm 
        self.crop_dimension_x_mm = LabeledDoubleSpinBox(self)
        self.crop_dimension_x_mm.setText('crop X (mm)')
        self.crop_dimension_x_mm.setRange(0,3)
        self.crop_dimension_x_mm.setValue(1.0)
        self.crop_dimension_x_mm.setSingleStep(0.1)
        self.crop_dimension_x_mm.valueChanged.connect(self.update_tracker)

        self.crop_dimension_y_mm = LabeledDoubleSpinBox(self)
        self.crop_dimension_y_mm.setText('crop Y (mm)')
        self.crop_dimension_y_mm.setRange(0,3)
        self.crop_dimension_y_mm.setValue(1.5)
        self.crop_dimension_y_mm.setSingleStep(0.1)
        self.crop_dimension_y_mm.valueChanged.connect(self.update_tracker)

        # crop offset 
        self.crop_offset_mm = LabeledDoubleSpinBox(self)
        self.crop_offset_mm.setText('Y offset eyes')
        self.crop_offset_mm.setRange(-5,5)
        self.crop_offset_mm.setValue(-0.30)
        self.crop_offset_mm.setSingleStep(0.05)
        self.crop_offset_mm.valueChanged.connect(self.update_tracker) 

        #ksize_blur_mm 
        self.blur_sz_mm = LabeledDoubleSpinBox(self)
        self.blur_sz_mm.setText('blur size (mm)')
        self.blur_sz_mm.setRange(0,2)
        self.blur_sz_mm.setValue(0.06)
        self.blur_sz_mm.setSingleStep(0.01)
        self.blur_sz_mm.valueChanged.connect(self.update_tracker)
                
        # median filter size
        self.median_filter_sz_mm = LabeledDoubleSpinBox(self)
        self.median_filter_sz_mm.setText('medfilt size (mm)')
        self.median_filter_sz_mm.setRange(0,2)
        self.median_filter_sz_mm.setValue(0.06)
        self.median_filter_sz_mm.setSingleStep(0.01)
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
        parameters.addWidget(self.eye_gamma)
        parameters.addWidget(self.eye_contrast)
        parameters.addWidget(self.eye_norm)
        parameters.addWidget(self.eye_dyntresh_res)
        parameters.addWidget(self.eye_size_lo_mm)
        parameters.addWidget(self.eye_size_hi_mm)
        parameters.addWidget(self.crop_dimension_x_mm)
        parameters.addWidget(self.crop_dimension_y_mm)
        parameters.addWidget(self.crop_offset_mm)
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
        overlay_param = EyesTrackerParamOverlay(
            pix_per_mm = self.target_pix_per_mm.value(),
        )
        tracker_param = EyesTrackerParamTracking(
            pix_per_mm = self.pix_per_mm.value(),
            target_pix_per_mm = self.target_pix_per_mm.value(),
            eye_gamma = self.eye_gamma.value(),
            eye_contrast = self.eye_contrast.value(),
            eye_norm = self.eye_norm.value(),
            eye_dyntresh_res = self.eye_dyntresh_res.value(),
            eye_size_lo_mm = self.eye_size_lo_mm.value(),
            eye_size_hi_mm = self.eye_size_hi_mm.value(),
            crop_offset_mm = self.crop_offset_mm.value(),
            crop_dimension_mm = (self.crop_dimension_x_mm.value(), self.crop_dimension_y_mm.value()),
            blur_sz_mm = self.blur_sz_mm.value(),
            median_filter_sz_mm = self.median_filter_sz_mm.value(),
        )
        self.tracker = EyesTracker(tracker_param, overlay_param)

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