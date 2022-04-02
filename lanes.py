import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)                              #Converting color image to greyscale for easier computation
    blur = cv2.GaussianBlur(gray, (5,5),0)                                      #Applies Gaussian Blur to smoothen the image so that there is no false edge detection. extra info[Here the standard is a (5,5) kernal, and 0 deviation]
    canny = cv2.Canny(blur,50,150)                                              #Canny finds the derivative of the pixels.So if there is a high value of derivative, it indicates change in gradient. [The low and high threshold is 50 and 150 default]
    return canny


def region_of_interest(image):
    height = image.shape[0]                                                     #Gives the number of rows in the image array, which is the value of height
    triangle = np.array([
    [(200, height), (1100, height), (550, 250)]
    ])                                                                          #sets an array with the triangles dimensions
    mask = np.zeros_like(image)                                                 #creates a mask of the dimensions of image that is black (fills with zeroes)
    cv2.fillPoly(mask, triangle, 255)                                           #fills the polygon shape with white pixels(255), It also deals with multiple polygons but we have only one.. so in triangle we make it an array(basically add another[])
    masked_image = cv2.bitwise_and(image, mask)                                 #Does the bitwise_and operation, that is 1+1=1, or 1+0=0, OR 0+0=0.. Thus only highlighting the region of interest
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)                                           #Creates a mask with diimension of image
    if lines is not None:                                                       #Checking if there is lines
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)                                    #Actually creates a 2D array ( 1 row,4 colums which are actually x1,y1,etc). We reshape it into a 1D array and assign the values directly to x1, y1, etc
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)               #OpenCV function to create a line(image, two coordinates, color, thickness)
    return line_image

def make_coordinates(image, line_parameters):
    slope,intercept = line_parameters
    y1 = image.shape[0]                                                         #image shape is an array and its first element has y1(it is the max value)
    y2 = int(y1*(3/5))                                                          #We just want the lines to go 3/5 times of y1 value in the upward axis
    x1 = int((y1 - intercept)/slope)                                            # y=mx+b, therfore x=(y-b)/m
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image, lines):
    left_fit = []                                                               #left lane coordinates
    right_fit = []                                                              #right lane coordinates
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2), 1)                             #Numpy functions that fits a polynomial(y=mx+b) to given x and y points and returns the slope(m) and y intercepts(b). The 1 at the end indicates 1st degree polynomial
        slope = parameters[0]                                                   #parameters gives an array of 1 row and 2 columns..the slope is the first element in the array
        intercept = parameters[1]                                               #intercept is accessed.. it is the 2nd element in the array
        if slope < 0:
            left_fit.append((slope, intercept))                                 #the left lane has a negative slope as when x increases, y decreases
        else:
            right_fit.append((slope, intercept))                                #The right lane has a positive slope bcoz as x increases, y inncreases
    left_fit_average = np.average(left_fit, axis=0)                             #Going down vertically(axis=0) and finding average
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)                       #invoking the make_coordinates function
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


#TO MAKE LINES FROM IMAGE
# image = cv2.imread('test_image.jpg')                                          #Reads the picture and stores it as a matrix of pixels
# lane_image = np.copy(image)                                                   #Always recommended to store a copy of image and do manipulations on that while using arrays
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5 )   #Hough lines(pic,resolution(no:of pixels, 1 degree to radian), threshold = 100 intersections, place-holder array (just an empty array), Length of a line in pixels we will accept in the output()  )
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)              #Combines both the pictures-(the main picture's contrast is multiplied by 0.8 and the lines are multiplied by 1 therfore the lines appear brighter) then added with a factor of 1(not significant)
# cv2.imshow('Results', combo_image)                                            #Shows the picture.. First paramter is the heading of dialogue box, 2nd param is the image
# cv2.waitKey(0)                                                                #To make the dialogue box stay for specified time.. (0) indicates it stays till our command


#TO MAKE LINES FROM VIDEO
cap = cv2.VideoCapture("test2.mp4")                                             #Reads our video file
while(cap.isOpened()):                                                          #checks if video is opned and if true or not
    _, frame = cap.read()                                                       #first value is a boolean(not interested), frame of the video
    canny_image = canny(frame)                                                  #replaced lane_image with frame
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5 )   #Hough lines(pic,resolution(no:of pixels, 1 degree to radian), threshold = 100 intersections, place-holder array (just an empty array), Length of a line in pixels we will accept in the output()  )
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)                 #Combines both the pictures-(the main picture's contrast is multiplied by 0.8 and the lines are multiplied by 1 therfore the lines appear brighter) then added with a factor of 1(not significant)
    cv2.imshow('Results', combo_image)                                          #Shows the picture.. First parameter is the heading of dialogue box, 2nd param is the image
    if cv2.waitKey(3) == ord('q'):                                               #1 ms wait between frames and checks if the keyboard button "q" is pressed
        break
cap.release()                                                                   #To close the video
cv2.destroyAllWindows()
