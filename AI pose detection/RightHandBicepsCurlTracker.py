import cv2
import mediapipe as mp 
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a)#first
    b = np.array(b)#mid
    c= np.array(c)#end
    
    radians = np.arctan2(c[1]-b[1],c[0]-b[0])- np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle



cap = cv2.VideoCapture(0)

# Curl Counter Variables
counter = 0 
stage = None
# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
     while cap.isOpened():
        ret,frame = cap.read()
        #recolor Image to RGB
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Make Detection
        results = pose.process(image)
        #recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        # Extract  landmakrs
        try:
            landmarks = results.pose_landmarks.landmark
            #getting cordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            #calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            # Visualize angle just below the elbow
            elbow_coordinates = np.multiply(elbow, [frame.shape[1], frame.shape[0]]).astype(int)
            text_coordinates = (elbow_coordinates[0], elbow_coordinates[1] + 30) # Adjust Y coordinate to display text below the elbow
            #visualize angle
            cv2.putText(image, str(angle), 
                        text_coordinates, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Curl Counter Logic
            if angle >160:
                stage = "Down"
            if angle < 40 and stage == 'Down':
                stage = "Up"
                counter +=1
                
        except:
            pass
        
        # Render curl Counter
        # setup status bar
        cv2.rectangle(image,(0,0),(255,73),(245,117,16),-1)
        
        # Rep data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
        #render directions
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                                 )
        cv2.imshow("Mediapipe Feed",image)
        if cv2.waitKey(25) & 0xff == ord('p'):
            break
    
cap.release()
cv2.destroyAllWindows()