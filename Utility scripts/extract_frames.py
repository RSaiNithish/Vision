import os
import cv2
import numpy as np

def extract_frames(path,frame_number=None):
    """This function will extract frames from a given video

    Args:
        path (str): path to the video file

    Returns:
        list: returns a list of frames which are of type numpy array
    """
    video = cv2.VideoCapture(path)
    sucess,frame = video.read()
    frame_count = 0
    frames = []
    while sucess:
        frame_count+=1
        if frame_number:
            if frame_number == frame_count:
                return frame
        frames.append(frame)
        sucess,frame = video.read()
    return frames
