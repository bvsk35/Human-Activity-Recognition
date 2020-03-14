import json
from pprint import pprint
import glob, os
import numpy as np
from datetime import date

# Change the data path where all the .json files are for the test video
data_path = r"Data/MHAD/Test/"
os.chdir(data_path)
print(os.getcwd())

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
        pose_keypoints = data["people"][0]["pose_keypoints_2d"]
        j = 0
        for i in range(36):
            frame_kps.append(pose_keypoints[j])
            j += 1
            if (j+1)%3 == 0:
                j += 1
        kps.append(frame_kps)

# Check the shape of the data
kps_np = np.array(kps)
print(kps_np.shape)
print(len(kps))

# Check how many frames contained more than 1 person 
print(openpose_2person_count)

# Change the data path where .txt file will be stored for testing
data_path = r"Data/MHAD/Database/"
os.chdir(data_path)
print(os.getcwd())

file_name = "testvid_" + str(date.today()) + ".txt"
with open(file_name, "w") as text_file:
    for i in range(len(kps)):
        for j in range(36):
            text_file.write('{}'.format(kps[i][j]))
            if j < 35:
                text_file.write(',')
        text_file.write('\n')


# num_steps depends on rate of video and window_width to be used
# in this case camera was 22Hz and a window_width of 1.5s was wanted, giving 22*1.5 = 33
num_steps = 32
overlap = 0.8125 # 0 = 0% overlap, 1 = 100%

count = 0
for file in sorted(glob.glob("*.txt")):
    data_file = open(file,'r')
    file_text = data_file.readlines() 
    num_frames = len(file_text)
    num_framesets = int((num_frames - num_steps)/(num_steps*(1-overlap)))+1
    data_file.close
    file_name = "datapoint_" + str(count) + "_testvid_" + str(date.today()) + ".txt"
    x_file = open(file_name, 'a')
    for frameset in range(0, num_framesets):
        start = int(frameset*num_steps*(1-overlap))
        for line in range(start,(start+num_steps)):
            x_file.write(file_text[line])
    x_file.close()