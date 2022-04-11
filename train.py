import numpy as np 
import matplotlib.image as mimg
from skimage import feature
from sklearn import svm
from sklearn.externals import joblib
import os
import pandas as pd
def train_model():
	df = pd.DataFrame(columns = ["name"])
	count=-1
	label = -1
	a,files,b = next(os.walk("./recorded_images")) #to find out the no: of users
	train_data=np.zeros((15*len(files),280)) #creating a 2D array of zeroes to store the features of a images later 
	train_label=np.zeros((15*len(files))) #creating a array for storing labels of the images 
	print(files)
	for i in files:
		label = label+1
		for j in range(1,16): 
		    count=count+1
		    path = './recorded_images/{}/{}.png'.format(i,j)
		    im = mimg.imread(path)
		    feat,hog_image = feature.hog(im,orientations=8,pixels_per_cell=(16,16), #calculating the features of a face
		                                 visualize=True,block_norm='L2-Hys',
		                                 cells_per_block=(1,1))
		    train_data[count,:]=feat.reshape(1,-1) #converting the features in a 1D array and storing it in a 2D array
		    train_label[count]=label #storing the label of the particular user
		    print(i,j,label)
		df.loc[label]=i

	# model creation
	svm_model = svm.SVC(kernel='linear',gamma='scale')
	
	# train the model
	svm_model = svm_model.fit(train_data,train_label)

	joblib.dump(svm_model,'svm_face_train_modelnew.pkl') #saving the created model 
	print(df)
	df.to_csv("./target.csv")
	print('training done ')
if __name__=="__main__":
	train_model()
