import cv2
from skimage import feature
from sklearn.externals import joblib
from datetime import datetime as dt
import train
import statistics 
import pandas as pd

def test_model():	
	
	svm_model=joblib.load('svm_face_train_modelnew.pkl')
	num_of_sample = 10
	vid = cv2.VideoCapture(0) # to open the camera
	# haar cascade for frontal face
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	iter1=0
	li=[]
	while(iter1<num_of_sample):
		r,frame = vid.read();# capture a single frame
		frame = cv2.resize(frame,(640,480)) # resizig the frame
		im1 = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)# gray scale conversion of
		# color image
		face=face_cascade.detectMultiScale(im1,1.3,5)
		for x,y,w,h in (face):
		    # [255,0,0] #[B,G,R] 0 to 255 
		    cv2.rectangle(frame,(x,y),(x+w,y+h),[0,0,255],4)
		    iter1=iter1+1
		    im_f = im1[y:y+h,x:x+w]
		    im_f = cv2.resize(im_f,(112,92))#orl face matching size
		    
		    feat,hog_image = feature.hog(im_f,orientations=8,pixels_per_cell=(16,16),
		                                 visualize=True,block_norm='L2-Hys',
		                                 cells_per_block=(1,1))
		    val1=svm_model.predict(feat.reshape(1,-1))
			#val1=neural_model.predict(feat.reshape(1,-1))
			
		    str1=str(val1)
		    li.append(int(val1))   
		    
		    cv2.putText(frame,str1,(x,y), cv2.FONT_ITALIC, 1,
		               (255,0,255),2,cv2.LINE_AA)
		     
		    
		cv2.imshow('frame',frame)# display
		cv2.waitKey(1)
	vid.release()
	cv2.destroyAllWindows()
	got_label = statistics.mode(li) #calvulating mode for minimum error 
	df = pd.read_csv("./target.csv").drop("Unnamed: 0",axis = 1)
	got_name = str(df.loc[int(got_label),"name"])
	print(got_name)
	df = pd.read_csv("./Attendence_record/"+got_name+".csv").drop("Unnamed: 0",axis=1)
	length = len(df)
	#marking the attandence
	if length==0:
		df.loc[0]=[dt.now().strftime("%d/%m/%y"),dt.now().strftime("%H:%M"),"Entered"]
	elif (df.loc[length-1,"Attendence"]=="Entered"):
		df.loc[length]=[dt.now().strftime("%d/%m/%y"),dt.now().strftime("%H:%M"),"Exit"]
	elif (df.loc[length-1,"Attendence"]=="Exit"):
		df.loc[length]=[dt.now().strftime("%d/%m/%y"),dt.now().strftime("%H:%M"),"Entered"]
	df.to_csv("./Attendence_record/"+got_name+".csv")
	print("Attendence Marked")
if __name__=="__main__":
	test_model()
    
    
    
    
