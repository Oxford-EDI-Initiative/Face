#OBS
import obswebsocket
from obswebsocket import obsws, requests
import cv2

# Connect to OBS WebSocket
ws = obsws("localhost", 4444, "password")  # Default port is 4444, set your password
ws.connect()

# Switch to a different scene
ws.call(requests.SetCurrentScene("Deepfake Scene"))

cap = cv2.VideoCapture(0)  # OBS Virtual Camera is usually listed as '0'

while True:
  
  ret, frame = cap.read()

  # Apply your deepfake or face-swapping algorithm on the frame here
  processed_frame = apply_deepfake(frame)

  #Show the processed frame 
  cv2.imshow("Processed Video", processed_frame)

  # Break the loop on 'q' press
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
    
  ws.call(requests.SetCurrentScene("Scene 1"))

  # Send the frame to OBS VirtualCam or other source

cap.release()
cv2.destroyAllWindows()


import dlib

detector = dlib.get_frontal_face_detector()

def apply_deepfake(frame):
  pass
    #run model return the deepfaked frame
    #will use another file for this currently being deloped
  

  return frame


ws.disconnect()










