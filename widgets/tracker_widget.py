from gui.animal_widget import AnimalTrackerWidget
from gui.body_widget import BodyTrackerWidget
from gui.eye_widget import EyesTrackerWidget
from gui.tail_widget import TailTrackerWidget 
from trackers.tracker import Tracker
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QDockWidget, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from typing import Protocol, Optional
from scipy.spatial.distance import cdist
import numpy as np
from qt_widgets import NDarray_to_QPixmap, LabeledSpinBox
import cv2

# TODO add widget to chose assignment method
# TODO add widget to chose accumulator method (useful when you want to actually do the tracking)
# TODO add widget to show background subtracted image histogram 

class Assignment(Protocol):
    pass

class TrackerWidget(QMainWindow):

    def __init__(
            self, 
            assignment: Assignment, 
            animal_tracker_widget: AnimalTrackerWidget,
            body_tracker_widget: Optional[BodyTrackerWidget],
            eyes_tracker_widget: Optional[EyesTrackerWidget],
            tail_tracker_widget: Optional[TailTrackerWidget],
            *args, **kwargs
        ) -> None:

        super().__init__(*args, **kwargs)
        self.assignment = assignment
        self.animal_tracker_widget = animal_tracker_widget        
        self.body_tracker_widget = body_tracker_widget
        self.eyes_tracker_widget = eyes_tracker_widget
        self.tail_tracker_widget = tail_tracker_widget
        self.tracker = None
        self.current_id = 0
        self.declare_components()
        self.layout_components()

    def declare_components(self):
        self.image_overlay = QLabel(self)
        self.image_overlay.mousePressEvent = self.on_mouse_click

        self.zoom = LabeledSpinBox(self)
        self.zoom.setText('zoom (%)')
        self.zoom.setRange(0,500)
        self.zoom.setValue(66)
        self.zoom.setSingleStep(25)

    def layout_components(self):

        main_widget = QWidget()

        images_and_zoom = QVBoxLayout()
        images_and_zoom.addWidget(self.zoom)
        images_and_zoom.addWidget(self.image_overlay)
        images_and_zoom.addStretch()
    
        if self.body_tracker_widget is not None:
            dock_widget = QDockWidget('Single Animal', self)
            tabs = QTabWidget()
            tabs.addTab(self.body_tracker_widget, 'body')
            if self.eyes_tracker_widget is not None:
                tabs.addTab(self.eyes_tracker_widget, 'eyes')
            if self.tail_tracker_widget is not None:
                tabs.addTab(self.tail_tracker_widget, 'tail')      
            dock_widget.setWidget(tabs)  
            self.addDockWidget(Qt.RightDockWidgetArea, dock_widget)

        mainlayout = QHBoxLayout(main_widget)
        mainlayout.addLayout(images_and_zoom)
        mainlayout.addWidget(self.animal_tracker_widget)

        self.setCentralWidget(main_widget)
        
    def update_tracker(self):
        body_tracker = None
        eyes_tracker = None
        tail_tracker = None
        self.animal_tracker_widget.update_tracker()
        animal_tracker = self.animal_tracker_widget.tracker
        if self.body_tracker_widget is not None:
            self.body_tracker_widget.update_tracker()
            body_tracker = self.body_tracker_widget.tracker
        if self.eyes_tracker_widget is not None:
            self.eyes_tracker_widget.update_tracker()
            eyes_tracker = self.eyes_tracker_widget.tracker
        if self.tail_tracker_widget is not None:
            self.tail_tracker_widget.update_tracker()
            tail_tracker = self.tail_tracker_widget.tracker

        self.tracker = Tracker(
            self.assignment,
            None,
            animal_tracker,
            body_tracker,
            eyes_tracker,
            tail_tracker
        )

    def display(self, tracking):
        if tracking is not None:
            overlay = self.tracker.overlay_local(tracking)
            zoom = self.zoom.value()/100.0
            overlay = cv2.resize(overlay,None,None,zoom,zoom,cv2.INTER_NEAREST)
            self.image_overlay.setPixmap(NDarray_to_QPixmap(overlay))

            self.animal_tracker_widget.display(tracking['animals'])
            try:
                if self.body_tracker_widget is not None:
                    self.body_tracker_widget.display(tracking['body'][self.current_id])
                if self.eyes_tracker_widget is not None:
                    self.eyes_tracker_widget.display(tracking['eyes'][self.current_id])
                if self.tail_tracker_widget is not None:
                    self.tail_tracker_widget.display(tracking['tail'][self.current_id])
            except KeyError:
                pass

    def on_mouse_click(self, event):
        x = event.pos().x()
        y = event.pos().y() 

        mouse_position = np.array([[x, y]])
        zoom = self.zoom.value()
        mouse_position = mouse_position * 100.0/zoom

        centroids = self.assignment.get_centroids()
        ID = self.assignment.get_ID()
        dist = cdist(mouse_position, centroids)
        self.current_id = ID[np.argmin(dist)]