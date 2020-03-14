# Utility Functions

The following scripts were used in the creation of the dataset for RNN for Human Activity Recognition - 2D Pose Input.
They were run in the listed order below. Please note that any directory references will need to be changed before use.


`run_openpose.sh:` runs openpose on all images in Image\_DIR, outputting to Output\_DIR

`convert_json_to_text.py:` Converts output of OpenPose (.json) to a .txt file

`create_db.ps:` Creates database from converted OpenPose output files (should now be in .txt format) for use with RNN for Human Activity Recognition - 2D Pose Input

`Gen_Data_For_Testing:` If you want to test the neural network for a random video then paste all the OpenPose (.json) files in a folder and give path to the folder and run this file. It will create .txt file and this can be used for testing the neural network. 
