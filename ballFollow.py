from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# Update the color range for a tennis ball
greenLower = (20, 65, 5)  # Lower range for yellow/greenish-yellow
greenUpper = (64, 255, 255)  # Upper range for yellow/greenish-yellow

# Define a deque (double-ended queue) to store recent ball positions
pts = deque(maxlen=args["buffer"])
prev_pt = None
prediction_enabled = False
predicted_pt = None

# Initialize the video stream
if not args.get("video", False):
    vs = VideoStream(src=0).start() # Use the default camera as video source
else:
    vs = cv2.VideoCapture(args["video"])# Use a video file as video source
    time.sleep(2.0)

# Start the main loop
while True:
    frame = vs.read()  # Read a frame from the video stream
    frame = frame[1] if args.get("video", False) else frame  # For video files, select the second item

    if frame is None:
        break

    frame = imutils.resize(frame, width=800)   # Resize the frame for faster processing

    blurred = cv2.GaussianBlur(frame, (15, 15), 0) # Apply Gaussian blur to reduce noise in the frame

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Convert the frame to the HSV color space

   # Create a binary mask to isolate the ball color
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours of the ball in the mask
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    center = None

  # If contours are found
    if len(contours) > 0:
        # Find the largest contour (assumed to be the ball)
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # If the radius is large enough, draw a circle around the ball
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

         # Calculate the centroid of the ball contour
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        prev_pt = center

    # Add the centroid to the deque
    if center is not None:
        pts.appendleft(center)

    # Connect the centroids in the deque with lines
    if len(pts) >= 2:
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # Display the frame and mask
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    # Check for the 'q' key to quit the program
    if key == ord("q"):
        break
# Clean up resources
if not args.get("video", False):
    vs.stop()# Release the camera
else:
    vs.release()# Release the video file

cv2.destroyAllWindows()# Close all windows
