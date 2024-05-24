'''************************************************************************** 
utility file
Author: fvilmos, https://github.com/fvilmos
***************************************************************************'''
import numpy as np
import cv2
import os

def puttext_bg(img,text='',position=(10,160), font_type=None, font_size=0.4, font_color=[0,255,0],bg_color=[0,0,0],font_thickness=1):
    """
    Text with background color
    Args:
        img: input image
        text (str, optional): Defaults to ''.
        position (tuple, optional): Defaults to (10,160).
        font_type (_type_, optional): Defaults to None.
        font_size (float, optional): Defaults to 0.4.
        font_color (list, optional): Defaults to [0,255,0].
        bg_color (list, optional): Defaults to [0,0,0].
        font_thickness (int, optional): Defaults to 1.
    """

    if font_type is None:
        font_type = cv2.FONT_HERSHEY_SIMPLEX
    (t_w,t_h),_ = cv2.getTextSize(text, font_type, font_size, font_thickness)
    cv2.rectangle(img, position, (position[0] + t_w, position[1] + t_h), bg_color, -1)
    cv2.putText(img,text ,(position[0], int(position[1]+t_h+font_size-1)),font_type,font_size,font_color,font_thickness)

def plot_bbox(img, label="", bbox=[10,10,30,30], color=[0,255,0], shape=(256,256), factor=1, style="xywh"):
    """
    Plot bounding box 

    Args:
        img (_type_): input RGB image
        label (str, optional): class name. Defaults to "".
        bbox (list, optional): bounding box. Defaults to [10,10,30,30].
        color (list, optional): box color. Defaults to [0,255,0].
        shape (tuple, optional): image shape (h,w). Defaults to (256,256).
        factor (int, optional): multiplication factor. Defaults to 1.
        style (str, optional): box stype xyxy or xywh. Defaults to "xywh".

    Returns:
        _type_: image with owerlayed bbox
    """
    img = cv2.resize(img,shape)
    bbox *= factor
    x0,y0,x1,y1 = bbox
    
    if style=="xywh":
        cv2.rectangle(img,(int(x0-x1//2),int(y0-y1//2)),(int(x0+x1//2),int(y0+y1//2)),color,2)
    else:
        cv2.rectangle(img,(int(x0),int(y0)),(int(x1),int(y1)),color,2)
    puttext_bg(img,text=str(label),position=(int(x0+5),int(y0-5)),bg_color=[0,0,0])
    return img

