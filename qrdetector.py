#!/usr/bin/env python
# coding: utf-8

'''
input: image or frame
output: approximate coordinates and QR code clipping
'''

import cv2
import numpy as np
import math
import logging; logging.basicConfig(level=logging.INFO)


class qrdetector():
    def __init__(self, img):
        self.img = img
        self.valid = set()
        self.contour_all = []
        self.found = []
        self.boxes = []
        self.line_point = []
        self.contours = None

    def __find_pdp(self, hierarchy):
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for i in range(len(self.contours)):
                k = i
                c = 0
                while hierarchy[k][2] != -1:
                    k = hierarchy[k][2]
                    c = c + 1
                if c >= 5:
                    self.found.append(i)
            if len(self.found) >= 3:
                return True
        return False

    def __find_boxes(self):
        for i in self.found:
            rect = cv2.minAreaRect(self.contours[i])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box = [tuple(x) for x in box]
            self.boxes.append(box)
        if len(self.boxes[0]) >= 3:
            return True
        return False

    def __cv_distance(self, P, Q):
        return int(math.sqrt(pow((P[0] - Q[0]), 2) + pow((P[1] - Q[1]), 2)))

    def __cord_offset(self, a1, a2, b1, b2):
        a1_n = (a1[0] + np.int0((a2[0] - a1[0]) / 14), a1[1] + np.int0((a2[1] - a1[1]) / 14))
        b1_n = (b1[0] + np.int0((b2[0] - b1[0]) / 14), b1[1] + np.int0((b2[1] - b1[1]) / 14))
        a2_n = (a2[0] + np.int0((a1[0] - a2[0]) / 14), a2[1] + np.int0((a1[1] - a2[1]) / 14))
        b2_n = (b2[0] + np.int0((b1[0] - b2[0]) / 14), b2[1] + np.int0((b1[1] - b2[1]) / 14))
        return ((a1_n, b1_n), (a2_n, b2_n))

    def __createLineIterator(self, P1, P2, img):
        """
        Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

        Parameters:
            -P1: a numpy array that consists of the coordinate of the first point (x,y)
            -P2: a numpy array that consists of the coordinate of the second point (x,y)
            -img: the image being processed

        Returns:
            -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
        """
        # define local variables for readability
        imageH = img.shape[0]
        imageW = img.shape[1]
        P1X = P1[0]
        P1Y = P1[1]
        P2X = P2[0]
        P2Y = P2[1]
        # difference and absolute difference between points
        # used to calculate slope and relative location between points
        dX = P2X - P1X
        dY = P2Y - P1Y
        dXa = np.abs(dX)
        dYa = np.abs(dY)
        # predefine numpy array for output based on distance between points
        itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
        itbuffer.fill(np.nan)
        # Obtain coordinates along the line using a form of Bresenham's algorithm
        negY = P1Y > P2Y
        negX = P1X > P2X
        if P1X == P2X:  # vertical line segment
            itbuffer[:, 0] = P1X
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
        elif P1Y == P2Y:  # horizontal line segment
            itbuffer[:, 1] = P1Y
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
        else:  # diagonal line segment
            steepSlope = dYa > dXa
            if steepSlope:
                slope = dX.astype(np.float32) / dY.astype(np.float32)
                if negY:
                    itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
                else:
                    itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
                itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
            else:
                slope = dY.astype(np.float32) / dX.astype(np.float32)
                if negX:
                    itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
                else:
                    itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
                itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y
        # Remove points outside of image
        colX = itbuffer[:, 0]
        colY = itbuffer[:, 1]
        itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]
        # Get intensities from img ndarray
        itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]
        return itbuffer

    def __isTimingPattern(self, line):
        while any(line) and line[0] != 0:
            line = line[1:]
        while any(line) and line[-1] != 0:
            line = line[:-1]
        if any(line):
            c = []
            count = 1
            l = line[0]
            for p in line[1:]:
                if p == l:
                    count = count + 1
                else:
                    c.append(count)
                    count = 1
                l = p
            c.append(count)
            if len(c) < 5:
                return False
            threshold = 5
            return np.var(c) < threshold
        return False

    def __check(self, a, b, bi_img):
        s1_ab = ()
        s2_ab = ()
        s1 = np.iinfo(np.int32(10)).max
        s2 = s1
        for ai in a:
            for bi in b:
                d = self.__cv_distance(ai, bi)
                if d < s2:
                    if d < s1:
                        s1_ab, s2_ab = (ai, bi), s1_ab
                        s1, s2 = d, s1
                    else:
                        s2_ab = (ai, bi)
                        s2 = d
        if s1_ab and s2_ab:
            a1, a2 = s1_ab[0], s2_ab[0]
            b1, b2 = s1_ab[1], s2_ab[1]
            s1_ab_n, s2_ab_n = self.__cord_offset(a1, a2, b1, b2)
            line_1 = self.__createLineIterator(s1_ab_n[0], s1_ab_n[1], bi_img)[:, 2]
            line_2 = self.__createLineIterator(s2_ab_n[0], s2_ab_n[1], bi_img)[:, 2]
            if self.__isTimingPattern(line_1) or self.__isTimingPattern(line_2):
                return True
        return False

    def get_tribox(self, img):
        _, bi_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        for i in range(len(self.boxes)):
            for j in range(i + 1, len(self.boxes)):
                if self.__check(self.boxes[i], self.boxes[j], bi_img):
                    self.valid.add(i)
                    self.valid.add(j)
        if len(self.valid) >= 3:
            while len(self.valid) > 0:
                c = self.found[self.valid.pop()]
                for sublist in self.contours[c]:
                    self.contour_all.append(sublist)
            rect_all = cv2.minAreaRect(np.array(self.contour_all))
            tri_box = cv2.boxPoints(rect_all)
            tri_box = np.array(tri_box)
            return tri_box
        return None

    def __order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(self, pts):
        rect = self.__order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self.img, M, (maxWidth, maxHeight))
        return warped

    def detector(self, flag=0):
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
        edges = cv2.Canny(img_gray, 100, 200)
        _, self.contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if self.__find_pdp(hierarchy):
            if self.__find_boxes():
                tri_box = self.get_tribox(img_gray)
                if tri_box is not None:
                    warped = self.four_point_transform(tri_box)
                    if flag == 0:
                        # return the image
                        return warped
                    elif flag == 1:
                        # return the coordinates of points
                        return tri_box
