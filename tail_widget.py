from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout
from trackers.tail  import TailTracker, TailTrackerParamOverlay, TailTrackerParamTracking
from qt_widgets import NDarray_to_QPixmap, LabeledDoubleSpinBox, LabeledSpinBox
import cv2

# TODO maybe group settings into collapsable blocks

class TailTrackerWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = None
        self.declare_components()
        self.layout_components()

    def declare_components(self):
        self.image = QLabel(self)
        self.image_overlay = QLabel(self)

        # pix per mm
        self.pix_per_mm = LabeledDoubleSpinBox(self)
        self.pix_per_mm.setText('pix_per_mm')
        self.pix_per_mm.setRange(0,1000)
        self.pix_per_mm.setValue(40.0)
        self.pix_per_mm.setSingleStep(1.0)
        self.pix_per_mm.valueChanged.connect(self.update_tracker) 

        # target pix per mm
        self.target_pix_per_mm = LabeledDoubleSpinBox(self)
        self.target_pix_per_mm.setText('target pix_per_mm')
        self.target_pix_per_mm.setRange(0,1000)
        self.target_pix_per_mm.setValue(20.0)
        self.target_pix_per_mm.setSingleStep(0.5)
        self.target_pix_per_mm.valueChanged.connect(self.update_tracker) 

        # tail contrast
        self.tail_contrast = LabeledDoubleSpinBox(self)
        self.tail_contrast.setText('tail contrast')
        self.tail_contrast.setRange(0,10)
        self.tail_contrast.setValue(1.0)
        self.tail_contrast.setSingleStep(0.1)
        self.tail_contrast.valueChanged.connect(self.update_tracker) 
        
        # tail gamma
        self.tail_gamma = LabeledDoubleSpinBox(self)
        self.tail_gamma.setText('tail gamma')
        self.tail_gamma.setRange(0,10)
        self.tail_gamma.setValue(0.75)
        self.tail_gamma.setSingleStep(0.05)
        self.tail_gamma.valueChanged.connect(self.update_tracker) 
        
        # tail norm 
        self.tail_norm = LabeledDoubleSpinBox(self)
        self.tail_norm.setText('tail norm')
        self.tail_norm.setRange(0,1)
        self.tail_norm.setValue(0.2)
        self.tail_norm.setSingleStep(0.025)
        self.tail_norm.valueChanged.connect(self.update_tracker) 
        
        # arc angle deg
        self.arc_angle_deg = LabeledDoubleSpinBox(self)
        self.arc_angle_deg.setText('tail max angle (deg)')
        self.arc_angle_deg.setRange(0,360)
        self.arc_angle_deg.setValue(120)
        self.arc_angle_deg.setSingleStep(2.5)
        self.arc_angle_deg.valueChanged.connect(self.update_tracker) 

        # n_tail_points
        self.n_tail_points = LabeledSpinBox(self)
        self.n_tail_points.setText('#tail points')
        self.n_tail_points.setRange(0,100)
        self.n_tail_points.setValue(12)
        self.n_tail_points.setSingleStep(1)
        self.n_tail_points.valueChanged.connect(self.update_tracker)

        # tail_length_mm 
        self.tail_length_mm = LabeledDoubleSpinBox(self)
        self.tail_length_mm.setText('tail_length_mm')
        self.tail_length_mm.setRange(0,10)
        self.tail_length_mm.setValue(2.6)
        self.tail_length_mm.setSingleStep(0.025)
        self.tail_length_mm.valueChanged.connect(self.update_tracker)

        # n_pts_arc
        self.n_pts_arc = LabeledSpinBox(self)
        self.n_pts_arc.setText('angle res.')
        self.n_pts_arc.setRange(0,100)
        self.n_pts_arc.setValue(20)
        self.n_pts_arc.setSingleStep(1)
        self.n_pts_arc.valueChanged.connect(self.update_tracker)

        # n_pts_interp  
        self.n_pts_interp = LabeledSpinBox(self)
        self.n_pts_interp.setText('n_pts_interp')
        self.n_pts_interp.setRange(0,200)
        self.n_pts_interp.setValue(40)
        self.n_pts_interp.setSingleStep(1)
        self.n_pts_interp.valueChanged.connect(self.update_tracker)

        # dist_swim_bladder_mm  
        self.dist_swim_bladder_mm = LabeledDoubleSpinBox(self)
        self.dist_swim_bladder_mm.setText('Offset tail Y (mm)')
        self.dist_swim_bladder_mm.setRange(0,3)
        self.dist_swim_bladder_mm.setValue(0.2)
        self.dist_swim_bladder_mm.setSingleStep(0.025)
        self.dist_swim_bladder_mm.valueChanged.connect(self.update_tracker)

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
       
        # crop dimensions
        self.crop_dimension_x_mm = LabeledDoubleSpinBox(self)
        self.crop_dimension_x_mm.setText('crop X (mm)')
        self.crop_dimension_x_mm.setRange(0,10)
        self.crop_dimension_x_mm.setValue(3.5)
        self.crop_dimension_x_mm.setSingleStep(0.1)
        self.crop_dimension_x_mm.valueChanged.connect(self.update_tracker)
               
        self.crop_dimension_y_mm = LabeledDoubleSpinBox(self)
        self.crop_dimension_y_mm.setText('crop Y (mm)')
        self.crop_dimension_y_mm.setRange(0,10)
        self.crop_dimension_y_mm.setValue(3.5)
        self.crop_dimension_y_mm.setSingleStep(0.1)
        self.crop_dimension_y_mm.valueChanged.connect(self.update_tracker)
       
        # crop offset
        self.crop_offset_tail_mm = LabeledDoubleSpinBox(self)
        self.crop_offset_tail_mm.setText('crop offset (mm)')
        self.crop_offset_tail_mm.setRange(-5,5)
        self.crop_offset_tail_mm.setValue(2.25)
        self.crop_offset_tail_mm.setSingleStep(-0.05)
        self.crop_offset_tail_mm.valueChanged.connect(self.update_tracker)

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
        parameters.addWidget(self.tail_contrast)
        parameters.addWidget(self.tail_gamma)
        parameters.addWidget(self.tail_norm)
        parameters.addWidget(self.arc_angle_deg)
        parameters.addWidget(self.n_tail_points)
        parameters.addWidget(self.n_pts_arc)
        parameters.addWidget(self.n_pts_interp)
        parameters.addWidget(self.tail_length_mm)
        parameters.addWidget(self.dist_swim_bladder_mm)
        parameters.addWidget(self.crop_dimension_x_mm)
        parameters.addWidget(self.crop_dimension_y_mm)
        parameters.addWidget(self.crop_offset_tail_mm)
        parameters.addWidget(self.blur_sz_mm)
        parameters.addWidget(self.median_filter_sz_mm)
        parameters.addStretch()

        images = QVBoxLayout()
        images.addWidget(self.zoom)
        images.addWidget(self.image)
        images.addWidget(self.image_overlay)
        images.addStretch()

        mainlayout = QHBoxLayout()
        mainlayout.addLayout(images)
        mainlayout.addLayout(parameters)

        self.setLayout(mainlayout)

    def update_tracker(self):
        overlay_param = TailTrackerParamOverlay(
            pix_per_mm = self.target_pix_per_mm.value(),
            color_tail = (255, 128, 128),
            thickness = 2
        )
        tracker_param = TailTrackerParamTracking(
            pix_per_mm = self.pix_per_mm.value(),
            target_pix_per_mm = self.target_pix_per_mm.value(),
            tail_contrast = self.tail_contrast.value(),
            tail_gamma = self.tail_gamma.value(),
            tail_norm = self.tail_norm.value(),
            arc_angle_deg = self.arc_angle_deg.value(),
            n_tail_points = self.n_tail_points.value(),
            n_pts_arc = self.n_pts_arc.value(),
            n_pts_interp = self.n_pts_interp.value(), 
            tail_length_mm = self.tail_length_mm.value(),
            dist_swim_bladder_mm = self.dist_swim_bladder_mm.value(),
            crop_dimension_mm = (self.crop_dimension_x_mm.value(), self.crop_dimension_y_mm.value()),
            crop_offset_tail_mm = self.crop_offset_tail_mm.value(),
            blur_sz_mm = self.blur_sz_mm.value(),
            median_filter_sz_mm = self.median_filter_sz_mm.value(),
        )
        self.tracker = TailTracker(tracker_param, overlay_param)

    def display(self, tracking):
        if tracking is not None:
            overlay = self.tracker.overlay_local(tracking)

            zoom = self.zoom.value()/100.0
            image = cv2.resize(tracking.image,None,None,zoom,zoom,cv2.INTER_NEAREST)
            overlay = cv2.resize(overlay,None,None,zoom,zoom,cv2.INTER_NEAREST)

            self.image.setPixmap(NDarray_to_QPixmap(image))
            self.image_overlay.setPixmap(NDarray_to_QPixmap(overlay))
            self.update()