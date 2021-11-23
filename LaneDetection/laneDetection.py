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
    kernel = 201
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    canny= blur
    canny = cv2.Canny(gray, 5, 40,L2gradient = True)
    return canny

def region_of_interest(canny):
   height = canny.shape[0]
   width = canny.shape[1]
   mask = np.zeros_like(canny)
   triangle = np.array([[
   (0, height/1.75),
   (width/2, 0),
   (width, height/1.75),]], np.int32)
   cv2.fillPoly(mask, triangle, 255)
   masked_image = cv2.bitwise_and(canny, mask)
   return masked_image

h_min = 0
h_max = 83
s_min = 74
s_max = 230
v_min = 135
v_max = 255



cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
while True:
    #Prefiltrado Naranja    
    ret,frame = cap.read()

    height = frame.shape[0]
    width = frame.shape[1]

    pts = np.float32([[0,0],[width,0],[0,height],[width,height]])

    matrix = cv2.getPerspectiveTransform(pts,pts)
    imgOut=cv2.warpPerspective(frame,matrix,(width,height))

    imgHSV = cv2.cvtColor(imgOut, cv2.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(frame, frame, mask=mask)


    canny_image = canny(imgResult)
    cropped_canny = region_of_interest(canny_image)
    cv2.imshow("canny_image",canny_image)
    cv2.imshow("Normal",frame)
    cv2.imshow("Resized",imgOut)
    cv2.imshow("ROI",cropped_canny)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

