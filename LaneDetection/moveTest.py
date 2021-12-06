#!/usr/bin/env python

import cv2
import numpy as np
import rospy
from std_msgs.msg import Float32

#Global variables
a=0
error = 0
cumError = 0
rateError = 0
lastError = 0
setpoint = 0.7
lapSpeed = 0.4
kp = 6
ki = 2
kd = 8
outMin = -15
outMax = 15
throttleVal = -0.16 #
offsetRight = 0.0 # offset de giro a la derecha
#Graph
dataListErrOut = []
dataListSlo = []

def _map(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

# Function to mapping values
def mapFnc(value, fromMin, fromMax, toMin, toMax):
    # Figure out how 'wide' each range is
    fromSpan = fromMax - fromMin
    toSpan = toMax - toMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - fromMin) / float(fromSpan)

    # Convert the 0-1 range into a value in the right range.
    return toMin + (valueScaled * toSpan)

# PID
def computePID(inp):
    global cumError, lastError
    T = 0.1
    error = setpoint - inp  #Determine error
    cumError += error*T     #compute integral
    rateError = (error - lastError)/T # compute derivative
    
    out = kp*error + ki*cumError + kd*rateError #PID output
    lastError = error   #remember current error
    print("Errors: ")
    print(error)
    #print(cumError)
    #print(rateError)
    print("Out")
    print(out)
    outmapped = mapFnc(out, outMin, outMax, -0.95, 0.95)
    print("Out Mapeado")
    print(outmapped)
    print
    print
    dataListErrOut.append(str(error)+"\t" + str(out) + "\t" + str(outmapped))
    if(outmapped > 0.95):
        return float(0.95)
    elif(outmapped < -0.95):
        return float(-0.95)
    else:
        return float(outmapped)
    #return out #have function return the PID output

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

    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 4)
        
    gray = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
    kernel = 101
    #blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    #canny = blur
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
   (width*9/10,height/4),
   (width/10, height/4),
   ]], np.int32)
   cv2.fillPoly(mask, trapezoid, 255)
   #cv2.fillPoly(mask, trapezoid, 0)
   masked_image = cv2.bitwise_and(canny, mask)
   #cv2.polylines(masked_image, trapezoid, True, 255, 2)
   return masked_image

def houghLines(cropped_canny,width):
   return cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 10, 
       np.array([]), minLineLength=55, maxLineGap=15)
def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 0.0)
 
def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        if lines[0] is not None:
            x1=lines[0][0]
            y1=lines[0][1]
            x2=lines[0][2]
            y2=lines[0][3]
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)
        if lines[1] is not None:
            x1=lines[1][0]
            y1=lines[1][1]
            x2=lines[1][2]
            y2=lines[1][3]
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)
        #for x1, y1, x2, y2 in lines:
        #    #print("Valores de la linea")
        #    #print(x1,x2,y1,y2)
        #    cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)
    return line_image
 
def make_points(image, line):
    #print("******Voy a imprimir line /n ****")
    #print(line)
    slope, intercept  = line
    #print("******Voy a imprimir slope /n ****")
    #print(slope)
    y1 = int(round(image.shape[0],2)) #height
    y2 = int(round(y1*1.0/5,2))      
    x1 = int(round((y1 - intercept)//slope,2))
    x2 = int(round((y2 - intercept)//slope,2))
    return np.array([x1, y1, x2, y2]), slope
 
def average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    # x1 = 1
    # x2 = 1
    # y1 = 1
    # y2 = 1
    if lines is None:
        #print("*************/n     mori   /n  ****************")
        return None, [0.0,0.0]
    for line in lines:
        #print("*************/n No    mori   /n  ****************")
        x1,y1,x2,y2 = line.reshape(4)
        #print("Coordenadas son")
        #print(x1,y1,x2,y2)
        if x1==x2 or y1==y2:
            #print("Valores iguales")
            return None,[0.0,0.0]
        fit = np.polyfit((x1,x2), (y1,y2), 1)
        #print(fit)
        #print("After polyfit")
        slope = fit[0]
        intercept = fit[1]
        if slope < 0: 
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    if left_fit == []:
        #print("******************* No hubo izquierdo ****************")
        boolLeft=0
        slopeLeft = 0.0
        #return None, None
    else:
        #print("******************* Si hubo izquierdo ****************")    
        left_fit_average  = np.average(left_fit, axis=0)    
        left_line,slopeLeft  = make_points(image, left_fit_average)
        boolLeft=1
        #print(left_line)

    if right_fit == []:
        #print("******************* No hubo derecho ****************")
        boolRight=0
        averaged_lines = None
        slopeRight = 0.0
        #return None, None
    else:
        #print("******************* Si hubo derecho ****************")
        right_fit_average = np.average(right_fit, axis=0)
        right_line,slopeRight = make_points(image, right_fit_average)
        boolRight=1
        #print(right_line)

    slopeValues = [slopeLeft,slopeRight]

    if boolLeft == 1:
        averaged_lines = [left_line, None]
    if boolRight == 1:
        averaged_lines = [None, right_line]
    if (boolRight==1) and (boolLeft==1):
        averaged_lines = [left_line, right_line]
    if (boolRight==0) and (boolLeft==0):
        averaged_lines = None
    

    return averaged_lines, slopeValues


def movement(slopeVal,pub_throttle,pub_steering):
    slopeLeft=slopeVal[0]
    slopeRight=slopeVal[1]
    caso = 0 #
    #pub_throttle.publish(-0.3)
    print("********pendiente izquierda")
    print(slopeLeft)
    print("********pendiente derecha")
    print(slopeRight)
    #dataListSlo.append(str(slopeLeft) + "\t" + str(slopeRight)  + "\t" + str(slopeLeft+slopeRight))

    #steeringVal = 0.0
    if((-slopeLeft > setpoint) and (slopeRight > setpoint)): #Caso 1
        throttleVal = -0.4
        steeringVal = 0.0
        caso = 1
    elif(-slopeLeft > 0.0 and -slopeLeft < setpoint): #Caso 4,6,8
        if (slopeRight > 0.0 and slopeRight < setpoint): #Caso 8
            throttleVal = -0.4
            steeringVal = 0.0
            caso = 8
        elif (slopeRight == 0.0): #Caso 6
            throttleVal = -lapSpeed
            #steeringVal = _map(-slopeLeft, 0.1, setpoint, 0.95, 0.1)
            steeringVal = 0.45 + offsetRight
            caso = 6
        else: #Caso 4
            throttleVal = -0.4
            steeringVal = 0.0 ## avanzar
            caso = 4
    elif(slopeRight > 0.0 and slopeRight < setpoint): #Caso 2,7
        if(slopeLeft==0.0): #Caso 7
            throttleVal = -lapSpeed
            #steeringVal = _map(slopeRight, 0.1, setpoint, 0.95, 0.1)
            steeringVal = -0.45
            caso = 7
        else: #Caso 2
            throttleVal = -0.4
            steeringVal = -0.0
            caso = 2
    elif(-slopeLeft > setpoint and slopeRight==0.0): #Caso 3
        throttleVal = -0.4
        steeringVal = 0.55 + offsetRight
        caso = 3
    elif(slopeRight > setpoint and slopeLeft==0.0): #Caso 5
        throttleVal = -0.4
        steeringVal = -0.55
        caso = 5
    #steeringVal = computePID(slopeLeft+slopeRight)
    pub_steering.publish(steeringVal)
    pub_throttle.publish(throttleVal)
    return caso


h_min = 0
h_max = 20
s_min = 50
s_max = 255
v_min = 110
v_max = 255




def mainCamera():
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    pub_throttle = rospy.Publisher('throttle', Float32, queue_size=8)
    pub_steering = rospy.Publisher('steering', Float32, queue_size=8)
    rospy.init_node('teleop', anonymous=True)
    rate = rospy.Rate(200) # 100hz
    caso = 0

    #pub_throttle.publish(-1.0)
    #pub_steering.publish(-1.0)

    while not rospy.is_shutdown() or a==1:
        
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
        averaged_lines, slopeValues = average_slope_intercept(frame, lines)
        if (slopeValues[0] == 0.0) and (slopeValues[1] == 0.0):
            #print("NO me voy a mover*****************")
            pub_throttle.publish(0.0)
            pub_steering.publish(0.0)
            
        else:
            
            #print("****** Me voy a mover")
            caso = movement(slopeValues,pub_throttle,pub_steering)
            #print(lines)
        line_image = display_lines(frame, averaged_lines)
        #line_image = display_lines(frame, lines)
        combo_image = addWeighted(frame, line_image)
        text1 = cv2.putText(combo_image, str(slopeValues), (10,40), cv2.FONT_HERSHEY_DUPLEX , 1, (255,0,255), 2)
        imageFinal = cv2.putText(text1, str(caso), (10,80), cv2.FONT_HERSHEY_DUPLEX , 1, (255,0,255), 2)
        #cv2.imshow("Canny",canny_image)
        #cv2.imshow("ROI",cropped_canny)

        cv2.imshow("result", imageFinal)
        #cv2.imshow("Oranged",imgResult)
        #cv2.imshow("Normal",frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            a=1
            cap.release()
            cv2.destroyAllWindows()
            # with open("dataErrOut.txt", "w") as f:
            #     for elemento in dataListErrOut:
            #         f.write(elemento)
            #         f.write("\n")
            
            # with open("dataSlo.txt", "w") as f:
            #     for elemento in dataListSlo:
            #         f.write(elemento)
            #         f.write("\n")
        rate.sleep()

    

if __name__ == '__main__':
    try:
        mainCamera()
    except rospy.ROSInterruptException:
        pass
