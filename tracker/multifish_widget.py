from .animal_widget import AnimalTrackerWidget
from .body_widget import BodyTrackerWidget
from .eyes_widget import EyesTrackerWidget
from .tail_widget import TailTrackerWidget 
from .assignment_widget import AssignmentWidget
from .multifish import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QDockWidget, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from typing import Protocol, Optional
from scipy.spatial.distance import cdist
import numpy as np
from qt_widgets import NDarray_to_QPixmap, LabeledSpinBox
import cv2

# TODO add widget to chose accumulator method (useful when you want to actually do the tracking)
# TODO add widget to show background subtracted image histogram 

class Assignment(Protocol):
    pass

class TrackerWidget(QMainWindow):

    def __init__(
            self, 
            assignment_widget: AssignmentWidget, 
            animal_tracker_widget: AnimalTrackerWidget,
            body_tracker_widget: Optional[BodyTrackerWidget],
            eyes_tracker_widget: Optional[EyesTrackerWidget],
            tail_tracker_widget: Optional[TailTrackerWidget],
            *args, **kwargs
        ) -> None:

        super().__init__(*args, **kwargs)
        self.assignment_widget = assignment_widget
        self.animal_tracker_widget = animal_tracker_widget        
        self.body_tracker_widget = body_tracker_widget
        self.eyes_tracker_widget = eyes_tracker_widget
        self.tail_tracker_widget = tail_tracker_widget
        self.tracker = None
        self.current_id = 0
        self.declare_components()
        self.layout_components()

    def declare_components(self) -> None:

        self.image_overlay = QLabel(self)
        self.image_overlay.mousePressEvent = self.on_mouse_click

        self.zoom = LabeledSpinBox(self)
        self.zoom.setText('zoom (%)')
        self.zoom.setRange(0,500)
        self.zoom.setValue(66)
        self.zoom.setSingleStep(25)

    def layout_components(self) -> None:

        main_widget = QWidget()

        images_and_zoom = QVBoxLayout()
        images_and_zoom.addWidget(self.assignment_widget)
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

    def update_tracker(self) -> None:
        
        self.animal_tracker_widget.update_tracker()
        animal_tracker = self.animal_tracker_widget.tracker
        animal_overlay = self.animal_tracker_widget.overlay

        body_tracker = None
        eyes_tracker = None
        tail_tracker = None
        
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

        assignment = self.assignment_widget.get_assignment()

        if assignment is not None:

            self.tracker = MultiFishTracker(
                assignment,
                None,
                animal_tracker,
                body_tracker,
                eyes_tracker,
                tail_tracker
            )

            self.overlay = MultiFishOverlay(
                animal_overlay,
                body_overlay,
                eyes_overlay,
                tail_overlay
            )

    def display(self, tracking: MultiFishTracking) -> None:

        if tracking is not None:

            overlay = self.overlay.overlay(tracking)
            
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

    def on_mouse_click(self, event) -> None:

        x = event.pos().x()
        y = event.pos().y() 

        mouse_position = np.array([[x, y]])
        zoom = self.zoom.value()
        mouse_position = mouse_position * 100.0/zoom

        assignment = self.assignment_widget.get_assignment()
        if assignment is not None:
            centroids = assignment.get_centroids()
            ID = assignment.get_ID()
            dist = cdist(mouse_position, centroids)
            self.current_id = ID[np.argmin(dist)]