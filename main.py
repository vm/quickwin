import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2

# Get user supplied values
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30)
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
(x, y, w, h) = faces[0]
cropped = image[x:x+w, y:y+h]

cv2.imshow("Faces found", cropped)
cv2.waitKey(0)
