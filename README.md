# Human-Activity-Recognition
## RNN-LSTM based Human Activity Recognition 

Running the code is straightforward. Only the `data_paths` must be changed according to where the files are located. 
### How to train?
- Main the code is present in the file `RNN_Human_Activity_Recognition.ipynb`.
- First download the data text files from the following google drive [link](https://drive.google.com/open?id=192z92yZSQwaWbhjAHMumah09bmry6saH). Please download all the files present under the folder `MHAD`.
- Paste above files in convient location of your choice. Then change the following variables present in `RNN_Human_Activity_Recognition.ipynb`.
  - Change `data_path = '/content/gdrive/My Drive/Deep Learning/MHAD/'` to where the text files are saved. And run the code to train. 
  - `X_train.txt`, `X_test.txt`, `Y_train.txt`, and `Y_test.txt` are training and testing files. `datapoint_1.txt`, `datapoint_2.txt`, `datapoint_3.txt`, `datapoint_4.txt`, and `datapoint_5.txt` are the video files on which prediction was shown. 
  - Labels in the data set: `jumping, jumping jacks, boxing, waving one hand, waving two hands, and clapping hands`
  
  
## Project Submission 8
### How to generate data from a sample video and do prediction
`ABCD`
### Trained Model weights
`ABCD`
Please go to my google drive [link](https://drive.google.com/open?id=192z92yZSQwaWbhjAHMumah09bmry6saH) and then go to the `checkpoints\Project Submission 4` folder. 
### Results
#### Accuracy on Test set: ~99%
![Accuracy](images/Project%20Submission%204/Acc.jpeg)
#### Confusion Matrix: possible confusion between labels clapping hands and waving hands, boxing
![Confusion Matrix](images/Project%20Submission%204/Confusion.jpeg)
#### Test video predictions [link](https://youtu.be/G_8_L7a7mLI)
![Predictions](images/Project%20Submission%204/Test1.jpeg)
  
## Project Submission 4, 5, and 6
### How to generate data from a sample video?
- In this repo under the folder `misc` I have uploaded many helper codes. For currently the data provided [here](https://tele-immersion.citris-uc.org/berkeley_mhad) are `.pgm` files for each frame. To get a video out of these frames first run the `ConvertPGMtoPNG.ipynb` (do necessary changes in the file) on all `.pgm` files to get `.png` files. Then run following command in terminal to convert all the `.png` files to a video `ffmpeg -i %0d.png -vcodec libx264 --pix_fmt yuv420p test.mp4`
- If you have video then please first run Openpose on it and save all the body landmarks in `.json` files. Store them in a folder.
- Then please run the `Gen_Data_For_Testing.py` to generate the data text file which can fed into the network to get the predictions. You will also need to provide the time stamps for the video to get prediction plots.
  - You will find the variable `data_path` twice in the above `.py` file. First time set `data_path` variable to location where the `.json` files are. Next time `data_path` variable to where you want to save the test data `.txt` file.
### How to do prediction on a sample video?
- To run the test on a sample video run the `RNN_Human_Activity_Recognition.ipynb` and give the file location for the test data file in the variable `data_path`. Note: I currently need time stamps text file to generate the plots. 
### Trained Model weights
Please go to my google drive [link](https://drive.google.com/open?id=192z92yZSQwaWbhjAHMumah09bmry6saH) and then go to the `checkpoints\Project Submission 4` folder. There I have uploaded the latest trained weights of the model. I am still figuring out how to load to trained model directly and do predictions using it. 
### Results
#### Accuracy on Test set: 97%
![Accuracy](images/Project%20Submission%204/Acc.jpeg)
#### Confusion Matrix: possible confusion between labels clapping hands and waving hands, boxing
![Confusion Matrix](images/Project%20Submission%204/Confusion.jpeg)
#### Test video predictions [link](https://youtu.be/G_8_L7a7mLI)
![Predictions](images/Project%20Submission%204/Test1.jpeg)


## References
- [Moments in Time](http://moments.csail.mit.edu/)
- [LSTM for Human Activity Recognition](https://github.com/stuarteiffert/RNN-for-Human-Activity-Recognition-using-2D-Pose-Input)
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- [Optical Flow](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html)
- [HMDB](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
