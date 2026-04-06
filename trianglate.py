from dataclasses import dataclass
import math
import cv2
import numpy as np

@dataclass
class tri_data:
    d: float = None
    cx: float = None
    cy: float = None
    b: float = None
    f: float = None
    x1: float = None
    y1: float = None
    x2: float = None
    y2: float = None
    x: float = None
    y: float = None
    z: float = None
    f1: np.ndarray = None
    f2: np.ndarray = None
    h: int = None
    w: int = None
    rf1: np.ndarray = None
    zoom_factor: float = None
    center_x: int = None
    center_y: int = None
    start_x: int = None
    start_y: int = None
    f1_cropped: np.ndarray = None


class tri:
    def __init__(self):
        self.data = tri_data()

    def Triangulate(self,f,b,p1,p2,cx,cy):
        #set inital values

        self.data.f = f
        self.data.b = b
        self.data.x1 = p1[0]
        self.data.y1 = p1[1]
        self.data.x2 = p2[0]
        self.data.y2 = p2[1]
        self.data.cx,self.data.cy = cx,cy
        #triangluate
        self.data.d = self.data.x1-self.data.x2
        self.data.x = ((self.data.x1-self.data.cx) * self.data.b) / self.data.d
        self.data.y = ((self.data.y1-self.data.cy) * self.data.b) / self.data.d
        self.data.z = (self.data.f * self.data.b) / self.data.d

        #return
        return self.data.x, self.data.y, self.data.z

    def match_res(self, f, zf):
        # Store inputs
        self.data.f1 = f

        # --- ZOOM f1 ---
        self.data.zoom_factor = zf# store zoom factor in dataclass
        self.data.rf1 = cv2.resize(self.data.f1, None, fx=self.data.zoom_factor, fy=self.data.zoom_factor)

        # Crop center region from rf1 to match f2 size
        self.data.center_y = self.data.rf1.shape[0] // 2
        self.data.center_x = self.data.rf1.shape[1] // 2
        self.data.w = self.data.f1.shape[1]
        self.data.h = self.data.f1.shape[0]
        self.data.start_x = self.data.center_x - self.data.w // 2
        self.data.start_y = self.data.center_y - self.data.h // 2

        self.data.f1_cropped = self.data.rf1[
                               self.data.start_y:self.data.start_y + self.data.h,
                               self.data.start_x:self.data.start_x + self.data.w
                               ]
        return self.data.f1_cropped
