#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 22:35:34 2022

@author: jonatas e gustavo
"""

import numpy as np
import cv2

im = cv2.imread('images/image.jpeg')
blur = cv2.blur(im,(15,1))
imgray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
imgray1 = cv2.equalizeHist(imgray)
ret,thresh = cv2.threshold(imgray1,90,250,43)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

img = cv2.drawContours(im, contours, -2, (0,255,0), 5)

cv2.imshow('Troncos',img)
cv2.waitKey(0)
cv2.destroyAllWindows()