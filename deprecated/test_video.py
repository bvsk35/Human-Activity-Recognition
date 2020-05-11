# Import required libraries
import json
from pprint import pprint
import glob, os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
import random
from random import randint
import time
import json
import tensorflow as tf
import torch.optim
import torch.nn.parallel
from torch.nn import functional as F
import models
from utils import extract_frames, load_frames, render_frames

# Args
parser = argparse.ArgumentParser(description="Test on a single video", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--video_path', help='Path to test video file', type=str, default=None)
parser.add_argument('--video_json_path', help='Path to openpose JSON files for test video', type=str, default=None)
parser.add_argument('--load', help='Path for the weights of the trained model', type=str, default=None)
parser.add_argument('--home_dir', help='Path for main script', type=str, default=None)
parser.add_argument('--num_segments', type=int, default=16)
parser.add_argument('--arch', type=str, default='resnet3d50', choices=['resnet50', 'resnet3d50'])
parser.add_argument('--generate_data', help='Generate the data for test video', default=True)
args = parser.parse_args()

# Generate the data
if args.generate_data:
	'''
	Extract Keypoints:
	Extract x, y co-ordinates for all pose keypoints and all frames. 
	This code can handle only one video at a time i.e. keypoints from one test video can be used per run.
	'''
	print('Generating data for testing......')
	data_path = args.video_json_path
	os.chdir(data_path)
	print('Current directory:', os.getcwd())

	kps = []
	openpose_2person_count = 0

	for file in sorted(glob.glob("*.json")):
		with open(file) as data_file: 
			data = json.load(data_file)
			if len(data["people"]) > 1:
				pprint("More than one detection in file, check the noise:")
				openpose_2person_count += 1
				print(file)
			frame_kps = []
			try: 
				pose_keypoints = data["people"][0]["pose_keypoints_2d"]
			except:
				continue
			j = 0
			for i in range(36):
				frame_kps.append(pose_keypoints[j])
				j += 1
				if (j+1)%3 == 0:
					j += 1
			kps.append(frame_kps)

	# Check the shape of the data
	kps_np = np.array(kps)
	print('Chekc the shape of data: ', kps_np.shape)
	print(len(kps))

	# Check how many frames contained more than 1 person 
	print('No. of frames with more that 1 person: ', openpose_2person_count)

	'''
	Create .txt file
	Write all the data into a .txt file. Each row indicates all the frames that were extracted, and in each row 
	we will have 36 entries corresponding to x, y postions for all 18 keypoints (joints).
	'''
	file_name = "test_video" + ".txt"
	with open(file_name, "w") as text_file:
		for i in range(len(kps)):
			for j in range(36):
				text_file.write('{}'.format(kps[i][j]))
				if j < 35:
					text_file.write(',')
			text_file.write('\n')

	'''
	Convert .txt file into a testing data point
	num_steps depends on rate of video and window_width to be used in this case camera was 22Hz 
	and a window_width of 1.5s was wanted, giving 22*1.5 = 33
	'''
	num_steps = 32
	overlap = 0.8125 # 0 = 0% overlap, 1 = 100%

	for file in sorted(glob.glob("*.txt")):
		data_file = open(file,'r')
		file_text = data_file.readlines() 
		num_frames = len(file_text)
		num_framesets = int((num_frames - num_steps)/(num_steps*(1-overlap)))+1
		data_file.close
		file_name = "data_point" + ".txt"
		x_file = open(file_name, 'a')
		for frameset in range(0, num_framesets):
			start = int(frameset*num_steps*(1-overlap))
			for line in range(start,(start+num_steps)):
				x_file.write(file_text[line])
		x_file.close()

	'''
	Create time stamps
	'''
	# No. of timesteps per series
	n_steps = 32 
	file = open(args.video_json_path+'/data_point.txt', 'r')
	temp = np.array([elem for elem in [row.split(',') for row in file]], dtype=np.float32)
	file.close()
	blocks = int(len(temp)/32)
	temp = np.array(np.split(temp,blocks))
	a = np.linspace(0, np.around(len(kps)/32), temp.shape[0])
	np.savetxt('time_stamp.txt', a)
else:
	data_path = args.video_json_path

'''
Running the tests on video
Model 1:
'''
n_input = 36  # No. of input parameters per timestep
n_hidden = 34 # Hidden layer num of features
n_steps = 32 # No. of timesteps per series
n_classes = 6 # No. of output classes/labels
LABELS = [    
	"JUMPING",
	"JUMPING_JACKS",
	"BOXING",
	"WAVING_2HANDS",
	"WAVING_1HAND",
	"CLAPPING_HANDS"
] 

def load_X(X_path):
	file = open(X_path, 'r')
	X_ = np.array([elem for elem in [row.split(',') for row in file]], dtype=np.float32)
	file.close()
	blocks = int(len(X_) / 32)
	X_ = np.array(np.split(X_,blocks))
	return X_

def LSTM_RNN(_X, _weights, _biases):
	_X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
	_X = tf.reshape(_X, [-1, n_input])   
	# ReLU activation function
	_X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
	# Split data because rnn cell needs a list of inputs for the RNN inner loop
	_X = tf.split(_X, n_steps, 0) 

	# Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
	# With dropout prob of 0.5 
	lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
	lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
	lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell] * 2, state_is_tuple=True)
	outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)
	outputs = tf.contrib.layers.batch_norm(outputs)

	lstm_last_output = outputs[-1]
	
	# Linear activation
	# return tf.add(tf.matmul(lstm_last_output, _weights['out']), _biases['out'], name='Pred')
	pred = tf.matmul(lstm_last_output, _weights['out']) + _biases['out']
	return pred

def one_hot(y_):
	# One hot encoding of the network outputs
	y_ = y_.reshape(len(y_))
	n_values = int(np.max(y_)) + 1
	return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def tf_reset():
	try:
		sess.close()
	except:
		pass
	tf.reset_default_graph()
	return tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))

test_path = args.load
sess = tf_reset()

# Graph input/output
x = tf.placeholder(tf.float32, [None, 32, n_input], name='x')
y = tf.placeholder(tf.float32, [None, n_classes], name='y')

# Graph weights
weights = {
	'hidden': tf.Variable(tf.random_normal([n_input, n_hidden]), name='W1'), 
	'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0), name='W2')
}
biases = {
	'hidden': tf.Variable(tf.random_normal([n_hidden]), name='b1'),
	'out': tf.Variable(tf.random_normal([n_classes]), name='b2')
}

pred = LSTM_RNN(x, weights, biases)

# Restore the model 
saver = tf.train.Saver()
saver.restore(sess, test_path+"/RNN_Model_Trained.ckpt")

# Output prediction on test video
test_vid = data_path + '/data_point.txt'
test_vid = load_X(test_vid)
one_hot_predictions = sess.run([pred], feed_dict={x: test_vid})

'''
Model 2:
'''
data_path = args.home_dir
os.chdir(data_path)
print('Current directory:', os.getcwd())

# Load model
model = models.load_model(args.arch)

# Get dataset categories
categories = models.load_categories()

# Load the video frame transform
transform = models.load_transform()

# Obtain video frames
print('Extracting frames using ffmpeg...')
frames = extract_frames(args.video_path, args.num_segments)


# Prepare input tensor
if args.arch == 'resnet3d50':
	# [1, num_frames, 3, 224, 224]
	input = torch.stack([transform(frame) for frame in frames], 1).unsqueeze(0)
else:
	# [num_frames, 3, 224, 224]
	input = torch.stack([transform(frame) for frame in frames])

# Make video prediction
with torch.no_grad():
	logits = model(input)
	h_x = F.softmax(logits, 1).mean(dim=0)
	probs, idx = h_x.sort(0, True)

# Output prediction on test video
temp = []
for i in range(0, 5):
	temp.append(categories[idx[i]])

'''
Save the Plots and JSON files
'''
if 'clapping' in temp or 'applauding' in temp:
	timestamp = np.loadtxt(args.video_json_path + '/time_stamp.txt')
	v = np.squeeze(np.array(one_hot_predictions))
	y = np.argmax(np.squeeze(v), axis=1)
	y_plot = np.where(y!=5, 0, 1)
	aidx = np.where(y==1)[0]
	for i in aidx:
		if i == 0:
			y_plot[i+1] = 1 if y_plot[i+1] == 0 else y_plot[i+1]
		elif (i+1) == len(y_plot):
			y_plot[i] = 1 if y_plot[i] == 0 else y_plot[i]
		else:
			y_plot[i-1] = 1 if y_plot[i-1] == 0 else y_plot[i-1]
			y_plot[i+1] = 1 if y_plot[i+1] == 0 else y_plot[i+1]
	temp_1 = np.concatenate((timestamp.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
	action = ["Detect Clapping Hands"]
	with open('test_video_out.json', 'w') as json_file:
	  json.dump(action, json_file)
	  json.dump(LABELS, json_file)
	  json.dump(temp_1.T.tolist(), json_file)

	plt.plot(timestamp, y_plot)
	plt.xlabel(r'Time stamps $\rightarrow$')
	plt.ylabel(r'Labels $\rightarrow$')
	plt.title('Label 0: No Clapping and Label 1: Clapping')
	plt.savefig('Test_1.jpeg')

	plt.plot(timestamp, y, '.')
	plt.xlabel(r'Time stamps $\rightarrow$')
	plt.ylabel(r'Labels $\rightarrow$')
	plt.title('All Labels')
	plt.savefig('Test_2.jpeg')
else:
	timestamp = np.loadtxt(args.video_json_path + '/time_stamp.txt')
	v = np.squeeze(np.array(one_hot_predictions))
	y = np.argmax(np.squeeze(v), axis=1)
	y_plot = []
	for i in y:
		if i==3 or i==4 or i==5:
			y_plot.append(1)
		else:
			y_plot.append(0)
	temp_1 = np.concatenate((timestamp.reshape((-1, 1)), y.reshape((-1, 1))), axis=1)
	action = ["Detect Clapping Hands"]
	with open('test_video_out.json', 'w') as json_file:
	  json.dump(action, json_file)
	  json.dump(LABELS, json_file)
	  json.dump(temp_1.T.tolist(), json_file)

	plt.plot(timestamp, y_plot)
	plt.xlabel(r'Time stamps $\rightarrow$')
	plt.ylabel(r'Labels $\rightarrow$')
	plt.title('Label 0: No Clapping and Label 1: Clapping')
	plt.savefig('Test_1.jpeg')

	plt.plot(timestamp, y, '.')
	plt.xlabel(r'Time stamps $\rightarrow$')
	plt.ylabel(r'Labels $\rightarrow$')
	plt.title('All Labels')
	plt.savefig('Test_2.jpeg')



