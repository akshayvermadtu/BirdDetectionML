import time
import cv2
import RPi.GPIO as G
import numpy as np
from imutils.video import VideoStream
import imutils
 
# Are we using the Pi Camera?
usingPiCamera = True
# Set initial frame size.
frameSize = (320, 240)
rows = open("synset_words.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
G.setmode(G.BCM)
G.setup(14,G.OUT)

target = ["cock" , "hen" , "ostrich" , "Struthio" ,  "brambling", "Fringilla" , "montifringilla" , "goldfinch" , "Carduelis",
"carduelis","house" "finch", "linnet", "junco", "indigo bunting", 'robin', "American robin", "bulbul", "jay", "magpie",
"chickadee", "water ouzel", "dipper", "kite", "bald eagle", "American eagle", "vulture", "great grey owl", "quail", 
"ruffed grouse", "partridge", "ptarmigan", "black grouse", "African grey", "sulphur-crested cockatoo,coucal", 
"hummingbird", "jacamar", "red-breasted merganser", "black swan","black stork", "American coot", "marsh hen", 
"ruddy turnstone", "red-backed sandpiper", "oystercatcher"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "squeezenet_v1.1.caffemodel")
print("[INFO] starting video stream...")
# Initialize mutithreading the video stream.
vs = VideoStream(src=0,usePiCamera=usingPiCamera, resolution=frameSize,
        framerate=0.5).start()
# Allow the camera to warm up.
time.sleep(2.0)

while True:
    # Get the next frame.
    frame = vs.read()
    
    # If using a webcam instead of the Pi Camera,
    # we take the extra step to change frame size.
    if not usingPiCamera:
        frame = imutils.resize(frame, width=frameSize[0])
    
    #print("showing....")
    blob = cv2.dnn.blobFromImage(frame, 1, (224, 224), (104, 117, 123))
    net.setInput(blob)
    preds = net.forward()
    
    preds = preds.reshape((1, len(classes)))
    idxs = np.argsort(preds[0])[::-1][:1]
    
    for (i, idx) in enumerate(idxs):
        print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,classes[idx], preds[0][idx]))
        if classes[idx] in target:
            print("Warning.............................................")
            G.output(14,True)
            time.sleep(2.0)
            G.output(14,False)

    # display the output image
    #cv2.imshow("Image", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
 
# Cleanup before exit.
cv2.destroyAllWindows()
vs.stop()
G.cleanup()
