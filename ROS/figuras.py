import cv2
import rospy
from std_msgs.msg import Float32


videoSource = 0 #para encontrar webcam
a=0

cap = cv2.VideoCapture(videoSource)


def teleop():
    #Setup topics publishing and nodes
    pub_throttle = rospy.Publisher('throttle', Float32, queue_size=8)
    pub_steering = rospy.Publisher('steering', Float32, queue_size=8)
    rospy.init_node('teleop', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    print("Running figuras.py")
    print("Triangulo: Forward")
    print("Rectangulo: Steer Left")
    print("Pentagono: Backward")
    print("Circulo: Steer Right")
    print("Cuadrado: Throttle stop")
    print("Hexagono: Steering reset")

    while not rospy.is_shutdown() or a==1:
      success, image =cap.read()
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      canny = cv2.Canny(gray, 10, 150)
      canny = cv2.dilate(canny, None, iterations=1)
      canny = cv2.erode(canny, None, iterations=1)
      #_, th = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
      #_,cnts,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# OpenCV 3
      cnts,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# OpenCV 4
      #cv2.drawContours(image, cnts, -1, (0,255,0), 2)

      for c in cnts:
        epsilon = 0.01*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,epsilon,True)
        #print(len(approx))
        x,y,w,h = cv2.boundingRect(approx)

        if len(approx)==3:
          cv2.putText(image,'Triangulo', (x,y-5),1,1.5,(0,255,0),2)
          pub_throttle.publish(-1.0)

        if len(approx)==4:
          aspect_ratio = float(w)/h
          print('aspect_ratio= ', aspect_ratio)
          if aspect_ratio == 1:
            cv2.putText(image,'Cuadrado', (x,y-5),1,1.5,(0,255,0),2)
            pub_throttle.publish(0.0)
          else:
            cv2.putText(image,'Rectangulo', (x,y-5),1,1.5,(0,255,0),2)
            pub_steering.publish(-1.0)

        if len(approx)==5:
          cv2.putText(image,'Pentagono', (x,y-5),1,1.5,(0,255,0),2)
          pub_throttle.publish(1.0)

        if len(approx)==6:
          cv2.putText(image,'Hexagono', (x,y-5),1,1.5,(0,255,0),2)
          pub_steering.publish(0.0)

        if len(approx)>10:
          cv2.putText(image,'Circulo', (x,y-5),1,1.5,(0,255,0),2)
          pub_steering.publish(1.0)

        cv2.drawContours(image, [approx], 0, (0,255,0),2)
        cv2.imshow('image',image)
        
      if cv2.waitKey(500) & 0xFF == 27:  # ESC
        a=1
      
      rate.sleep()




if __name__ == '__main__':
    try:
        teleop()
    except rospy.ROSInterruptException:
        pass
