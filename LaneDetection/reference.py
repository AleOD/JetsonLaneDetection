import cv2
import numpy as np

#Gstreamer pipeline settings
def gstreamer_pipeline(
    capture_width=600,
    capture_height=400,
    display_width=600,
    display_height=400,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def canny(img):
    if img is None:
        cap.release()
        cv2.destroyAllWindows()
        exit()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 85
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    canny= blur
    canny = cv2.Canny(gray, 10, 50,L2gradient = True)
    return canny

def region_of_interest(canny):
   height = canny.shape[0]
   width = canny.shape[1]
   mask = np.zeros_like(canny)
   triangle = np.array([[
   (-50, height),
   (width/2, 0),
   (width+50, height),]], np.int32)
   cv2.fillPoly(mask, triangle, 255)
   masked_image = cv2.bitwise_and(canny, mask)
   return masked_image

def houghLines(cropped_canny):
   return cv2.HoughLinesP(cropped_canny, 10, np.pi/180, 20, 
       np.array([]), minLineLength=40, maxLineGap=200)

def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)
 
def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)
    return line_image
 
def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1*3.0/5)      
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]
 
def average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0: 
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line  = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
while True:    
    ret,frame = cap.read()
    canny_image = canny(frame)
    cropped_canny = region_of_interest(canny_image)
    cv2.imshow("canny_image",canny_image)
    cv2.imshow("Normal",frame)
    cv2.imshow("ROI",cropped_canny)

    #lines = houghLines(cropped_canny)
    #averaged_lines = average_slope_intercept(frame, lines)
    #line_image = display_lines(frame, averaged_lines)
    #combo_image = addWeighted(frame, line_image)
    #cv2.imshow("result", combo_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

