import cv2
import os
import numpy as np

####KNN CODE###
def distance(v1, v2):
	# Eucledian 
	return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]
		# Compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]
################################

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

dataset_path = "./FaceData/"
labels = []
class_id = 0
names = {}
face_data = []

for fx in os.listdir(dataset_path):
	if fx.endswith(".npy"):
		names[class_id] = fx[:-4]
		print("Loading File ",fx)
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)

		#createLabels

		target = class_id * np.ones((data_item.shape[0],))
		labels.append(target)
		class_id+=1

X = np.concatenate(face_data,axis = 0) # 24,30000
Y = np.concatenate(labels,axis = 0) # 24,30000

trainset = np.concatenate((X,Y.reshape(-1,1)),axis = 1)

while True:
	ret,frame = cam.read()
	if ret == False:
		print("Something went wrong")
		continue

	key_pressed = cv2.waitKey(1)&0xFF
	if(key_pressed == ord('q')):
		break

	faces = face_cascade.detectMultiScale(frame,1.3,5)
	
	if(len(faces) == 0):
		cv2.imshow("Video",frame)
		continue

	for face in faces:
		x,y,w,h = face
		face_section = frame[y-10:y+10+h,x-10:x+10+w]
		face_section = cv2.resize(face_section,(100,100))
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		pred = knn(trainset,face_section.flatten())
		name = names[int(pred)]
		cv2.putText(frame,name,(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		#cv2.putText(frame,name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)

	cv2.imshow("Video",frame)

cam.release()
cv2.destroyAllWindows()