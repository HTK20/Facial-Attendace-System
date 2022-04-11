import cv2 #importing all the modules
import os
import pandas as pd

def face_record():
	try:
		os.mkdir("recorded_images")
	except FileExistsError:
		print("Folder already Existed")
	try:
		os.mkdir("Attendence_record")
	except FileExistsError:
		print("Folder already existed")
	def na():
		try:
			name = input("Enter your name")				
			os.mkdir("./recorded_images/"+name)
		except FileExistsError:
			print("name already existed try another")
			name = na()
		return(name)
	name = na()
	df = pd.DataFrame(columns = ["Date","Time","Attendence"]) #creating a csv file to keep a record of the attandence of a user
	df.to_csv("./Attendence_record/"+name+".csv")
	num_of_sample = 20
	vid = cv2.VideoCapture(0) # to open the camera
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')# haar cascade for frontal face
	iter1=0
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
		    cv2.putText(frame,'sample no.'+str(iter1),(x,y), cv2.FONT_ITALIC, 1,
		               (255,0,255),2,cv2.LINE_AA)
		    path2 = './recorded_images/{}/{}.png'.format(name,iter1) # path to save the image
		    cv2.imwrite(path2,im_f) # to save the image 
		    
		cv2.imshow('frame',frame)# display
		cv2.waitKey(1)
	vid.release()
	cv2.destroyAllWindows()

if __name__=="__main__":
	face_record()  
    
    
    
