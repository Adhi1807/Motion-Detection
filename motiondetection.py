# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 19:38:48 2023

@author: Adhithya
"""

import cv2

# Function to calculate absolute difference between two frames
def get_frame_diff(prev_frame, cur_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference between the two frames
    frame_diff = cv2.absdiff(prev_gray, cur_gray)
    
    # Apply thresholding to remove noise
    _, frame_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    
    return frame_diff

# Open video capture
video = cv2.VideoCapture(0)

# Read two consecutive frames
ret, prev_frame = video.read()
ret, cur_frame = video.read()

while ret:
    # Get the difference between the two frames
    frame_diff = get_frame_diff(prev_frame, cur_frame)
    
    # Find contours of moving objects
    contours, _ = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through contours and filter based on area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Adjust the threshold according to your needs
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(cur_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a rectangle around the moving object
    
    # Display the current frame
    cv2.imshow("Motion Detection", cur_frame)
    
    # Update frames
    prev_frame = cur_frame
    ret, cur_frame = video.read()
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
video.release()
cv2.destroyAllWindows()
