# Utility Functions

The following scripts were used in the creation of the dataset for RNN for Human Activity Recognition - 2D Pose Input.
They were run in the listed order below. Please note that any directory references will need to be changed before use.


`run_openpose.sh:` runs openpose on all images in IMAGE\_DIR, outputting to OUTPUT\_DIR

`convert_json_to_text.py:` Converts output of OpenPose (.json) to a .txt file

`create_db.ps:` Creates database from converted OpenPose output files (should now be in .txt format) for use with RNN for Human Activity Recognition - 2D Pose Input

`Gen_Data_For_Testing:` If you want to test the neural network for a random video then paste all the OpenPose (.json) files in a folder and give path to that folder and run this .py file. It will create .txt file on which we can test the predictions of the trained neural network. 
