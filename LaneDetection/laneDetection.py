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
    kernel = 101
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    canny = blur
    #canny = cv2.Canny(gray, 4, 100,L2gradient = True)
    canny = cv2.Canny(gray, 500, 475)
    return canny

def region_of_interest(canny):
   height = canny.shape[0]
   width = canny.shape[1]
   mask = np.zeros_like(canny)
   trapezoid = np.array([[
   (0, height),
   (width, height),
   (width*9/10,height/10),
   (width/10, height/10),
   ]], np.int32)
   cv2.fillPoly(mask, trapezoid, 255)
   #cv2.fillPoly(mask, trapezoid, 0)
   masked_image = cv2.bitwise_and(canny, mask)
   #cv2.polylines(masked_image, trapezoid, True, 255, 2)
   return masked_image

def houghLines(cropped_canny,width):
   return cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 3, 
       np.array([]), minLineLength=40, maxLineGap=5)
def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 0.0)
 
def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            print("Valores de la linea")
            print(x1,x2,y1,y2)
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)
    return line_image
 
def make_points(image, line):
    print("******Voy a imprimir line /n ****")
    print(line)
    slope, intercept  = line
    print("******Voy a imprimir slope /n ****")
    print(slope)
    y1 = int(round(image.shape[0],2)) #height
    y2 = int(round(y1*1.0/5,2))      
    x1 = int(round((y1 - intercept)//slope,2))
    x2 = int(round((y2 - intercept)//slope,2))
    return np.array([x1, y1, x2, y2])
 
def average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    # x1 = 1
    # x2 = 1
    # y1 = 1
    # y2 = 1
    if lines is None:
        print("*************/n     mori   /n  ****************")
        return None
    for line in lines:
        print("*************/n No    mori   /n  ****************")
        x1,y1,x2,y2 = line.reshape(4)
        print("Coordenadas son")
        print(x1,y1,x2,y2)
        fit = np.polyfit((x1,x2), (y1,y2), 1)
        print(fit)
        print("After polyfit")
        slope = fit[0]
        intercept = fit[1]
        if slope < 0: 
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    if left_fit == []:
        print("******************* No hubo izquierdo ****************")
        #left_fit_average = [[]]
        #left_line = [[]]
        return None
    else:
        print("******************* Si hubo izquierdo ****************")    
        left_fit_average  = np.average(left_fit, axis=0)    
        left_line  = make_points(image, left_fit_average)
        print(left_line)

    if right_fit == []:
        print("******************* No hubo derecho ****************")
        #right_fit_average = [[]]
        #right_line = [[]]
        return None
    else:
        print("******************* Si hubo derecho ****************")
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_points(image, right_fit_average)
        print(right_line)
    
    #left_fit_average  = np.average(left_fit, axis=0) 
    #left_line  = make_points(image, left_fit_average)
    #right_fit_average = np.average(right_fit, axis=0)
    #right_line = make_points(image, right_fit_average)

    averaged_lines = [left_line, right_line]
    return averaged_lines

h_min = 0
h_max = 20
s_min = 65
s_max = 190
v_min = 88
v_max = 200



cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
while True:
    #Prefiltrado Naranja    
    ret,frame = cap.read()

    # height = frame.shape[0]
    width = frame.shape[1]

    imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(frame, frame, mask=mask)


    canny_image = canny(imgResult)
    cropped_canny = region_of_interest(canny_image)
    lines = houghLines(cropped_canny,width)
    averaged_lines = average_slope_intercept(frame, lines)
    #print(lines)
    line_image = display_lines(frame, averaged_lines)
    #line_image = display_lines(frame, lines)
    combo_image = addWeighted(frame, line_image)
    #cv2.imshow("Canny",canny_image)
    cv2.imshow("ROI",cropped_canny)

    cv2.imshow("result", combo_image)
    cv2.imshow("Oranged",imgResult)
    #cv2.imshow("Normal",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

