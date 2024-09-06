from tracker.animal import AnimalTrackerWidget
from tracker.body import BodyTrackerWidget
from tracker.eyes import EyesTrackerWidget
from tracker.tail import TailTrackerWidget 
from .core import MultiFishTracker, MultiFishOverlay, MultiFishTracking
from .tracker import MultiFishTracker_CPU
from .overlay import MultiFishOverlay_opencv
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QDockWidget, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from typing import Optional
from qt_widgets import NDarray_to_QPixmap, LabeledSpinBox, FileSaveLabeledEditButton
import cv2
import json

# TODO add widget to chose accumulator method (useful when you want to actually do the tracking)
# TODO add widget to show background subtracted image histogram 

class TrackerWidget(QMainWindow):

    def __init__(
            self, 
            animal_tracker_widget: AnimalTrackerWidget,
            body_tracker_widget: Optional[BodyTrackerWidget],
            eyes_tracker_widget: Optional[EyesTrackerWidget],
            tail_tracker_widget: Optional[TailTrackerWidget],
            tracker_class: MultiFishTracker = MultiFishTracker_CPU,
            overlay_class: MultiFishOverlay = MultiFishOverlay_opencv,
            *args, **kwargs
        ) -> None:

        super().__init__(*args, **kwargs)
        self.animal_tracker_widget = animal_tracker_widget        
        self.body_tracker_widget = body_tracker_widget
        self.eyes_tracker_widget = eyes_tracker_widget
        self.tail_tracker_widget = tail_tracker_widget
        self.tracker = None
        self.tracker_class = tracker_class
        self.overlay_class = overlay_class
        self.declare_components()
        self.layout_components()

    def declare_components(self) -> None:

        self.image_overlay = QLabel(self)

        self.zoom = LabeledSpinBox(self)
        self.zoom.setText('zoom (%)')
        self.zoom.setRange(0,500)
        self.zoom.setValue(66)
        self.zoom.setSingleStep(25)

        self.save_tracking_param = FileSaveLabeledEditButton()
        self.save_tracking_param.setText('Save tracking parameters:')
        self.save_tracking_param.textChanged.connect(self.save_tracking)

    def layout_components(self) -> None:

        main_widget = QWidget()

        images_and_zoom = QVBoxLayout()
        images_and_zoom.addWidget(self.save_tracking_param)
        images_and_zoom.addWidget(self.zoom)
        images_and_zoom.addWidget(self.image_overlay)
        images_and_zoom.addStretch()
    
        if self.body_tracker_widget is not None:
            dock_widget = QDockWidget('Single Animal', self)
            self.tabs = QTabWidget()
            self.tabs.setMovable(True)
            self.tabs.setTabsClosable(True)
            self.tabs.tabCloseRequested.connect(self.on_tab_close)
            self.tabs.addTab(self.body_tracker_widget, 'body')
            if self.eyes_tracker_widget is not None:
                self.tabs.addTab(self.eyes_tracker_widget, 'eyes')
            if self.tail_tracker_widget is not None:
                self.tabs.addTab(self.tail_tracker_widget, 'tail')      
            dock_widget.setWidget(self.tabs)  
            self.addDockWidget(Qt.RightDockWidgetArea, dock_widget)

        mainlayout = QHBoxLayout(main_widget)
        mainlayout.addLayout(images_and_zoom)
        mainlayout.addWidget(self.animal_tracker_widget)

        self.setCentralWidget(main_widget)

    def on_tab_close(self, index) -> None:

        text = self.tabs.tabText(index)
        if text == 'eyes':
            self.body_tracker_widget = None
            self.tabs.removeTab(index)
        elif text == 'tail':
            self.body_tracker_widget = None
            self.tabs.removeTab(index)

    def save_tracking(self, filename):

        param = {}
        param['animal'] = self.animal_tracker_widget.tracker.tracking_param.to_dict()

        if self.body_tracker_widget is not None:
            param['body'] = self.body_tracker_widget.tracker.tracking_param.to_dict()

        if self.eyes_tracker_widget is not None:
            param['eyes'] = self.eyes_tracker_widget.tracker.tracking_param.to_dict()

        if self.tail_tracker_widget is not None:
            param['tail'] = self.tail_tracker_widget.tracker.tracking_param.to_dict()
        
        with open(filename, 'w') as fp:
            json.dump(param, fp)

    def update_tracker(self) -> None:
        
        self.animal_tracker_widget.update_tracker()
        animal_tracker = self.animal_tracker_widget.tracker
        animal_overlay = self.animal_tracker_widget.overlay

        body_tracker = None
        eyes_tracker = None
        tail_tracker = None
        body_overlay = None
        eyes_overlay = None
        tail_overlay = None
        
        if self.body_tracker_widget is not None:
            self.body_tracker_widget.update_tracker()
            body_tracker = self.body_tracker_widget.tracker
            body_overlay = self.body_tracker_widget.overlay
        
        if self.eyes_tracker_widget is not None:
            self.eyes_tracker_widget.update_tracker()
            eyes_tracker = self.eyes_tracker_widget.tracker
            eyes_overlay = self.eyes_tracker_widget.overlay
        
        if self.tail_tracker_widget is not None:
            self.tail_tracker_widget.update_tracker()
            tail_tracker = self.tail_tracker_widget.tracker
            tail_overlay = self.tail_tracker_widget.overlay

        # TODO update this 
        num_animals = 10

        self.tracker = self.tracker_class(
            num_animals,
            None,
            animal_tracker,
            body_tracker,
            eyes_tracker,
            tail_tracker
        )

        self.overlay = self.overlay_class(
            animal_overlay,
            body_overlay,
            eyes_overlay,
            tail_overlay
        )

    def display(self, tracking: MultiFishTracking) -> None:

        if tracking is not None:

            overlay = self.overlay.overlay(tracking.image, tracking)
            zoom = self.zoom.value()/100.0
            if (overlay is not None) and (overlay.size > 0): 
                overlay = cv2.resize(overlay,None,None,zoom,zoom,cv2.INTER_NEAREST)
                self.image_overlay.setPixmap(NDarray_to_QPixmap(overlay))

            self.animal_tracker_widget.display(tracking.animals)
            current_id = self.animal_tracker_widget.current_id
            try:
                if (self.body_tracker_widget is not None) and (tracking.body is not None):
                    self.body_tracker_widget.display(tracking.body[current_id])
                if (self.eyes_tracker_widget is not None) and (tracking.eyes is not None):
                    self.eyes_tracker_widget.display(tracking.eyes[current_id])
                if (self.tail_tracker_widget is not None) and (tracking.tail is not None):
                    self.tail_tracker_widget.display(tracking.tail[current_id])
            except KeyError:
                pass

