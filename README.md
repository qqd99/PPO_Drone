# PPO_Drone
This project try to control drone to avoid collision while going to the target using airsim plugin for UE5.2.<br>
The observation includes the current view in front of the drone and the distance to the goal. There are three actions available: move forward, rotate clockwise 30 degree, rotate counter clockwise 30 degree.<br>
To replicate this project, make sure the image feeding to model has shape (144, 256, 1), and it is the depth image (I use depth image with max range is 20 meters, so with locations in more than 20 meters, all the pixel will have the same value)<br><br>
For more information about the training environment, take a look at step_1.py<br><br>
Demo: test.mp3<br><br>
Try the parameters I have trained: https://drive.google.com/file/d/1NOyIOupuXWfMCotDGLAbR21mpS10KsxG/view?usp=sharing
