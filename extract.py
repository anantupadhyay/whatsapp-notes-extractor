import os
import numpy as np 
from glob import glob
from keras.preprocessing.image import *
from model import CNN_model

model = CNN_model()
model.load_weights('model_training/wts.h5')

WA_img_path = 'img_dir/'
notes_path = WA_img_path + 'notes/'

if not os.path.exists(notes_path):
	os.mkdir(notes_path)

def predict(file_path):
	img = load_img(file_path, target_size=(128,128,3))
	x = img_to_array(img) / 255.
	y = model.predict(np.expand_dims(x, axis=0))
	return np.squeeze(y) > 0.5

files = glob("img_dir/**.**")
# print files
cn = 0
for cnt, file_path in enumerate(files):
	if not cnt%10:
		print (str(cnt) + ' files examined')

	# print file_path
	var = predict(file_path)

	if var:
		cn += 1
		file_name = file_path.split('/')[-1]
		os.rename(file_path, notes_path+file_name)

print "{} Notes found".format(cn)