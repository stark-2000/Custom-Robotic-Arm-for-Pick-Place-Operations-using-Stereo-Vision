### Instruction to run perception_stack
# Download the code.zip to Downloads Folder & Extract
## Instructions:
 - Open Terminal and Run the following commands:
 ```
 cd Downloads
 cd code
 python3 main.py
 exit
 ```
 - The results will be printed to the terminal
 - OpenCV window will open and show the disparity map, Stereo IR frames, RGB cam feed and YOLO object detection overlayed on RGB cam feed.
 - Press q key to exit the program. Might be required to press multiple times. 
 - All the feed is in 640 * 480 resolution.
 - YOLOv8 model's weights are present in the same folder as the code.


### Instruction to run ros_moveit_stack
# Download the ws_wav_mov.zip to Downloads Folder & Extract
## Instructions:
 - Open Terminal and Run the following commands:
 ```
 cd Downloads
 cd code
 ```
 Extract the ws_wav_mov.zip to the same folder
 ```
 cd ws_wav_mov
 ```
 - Source ros and launch the demo.launch file
 ```
 source ./devel/setup.bash
 ```
 ```
 roslaunch ws_wav_mov demo.launch
 ```
 - Rebuild the workspace if required
 - In the RVIZ, you can set a goal pose by either clicking the predefined one or random and see the manipultor move to the goal pose.
 - The hardware API is present in the backend which gets the joint angles from moveit ros topic and sends to hardware arduino in manipulator using i2c interface and USB to serial converter.


## Dependencies:
 - ultralytics Library - for YOLOv8
    - (if req open terminal and run the following commands)
    ```
    pip install ultralytics
    ```

 - openCV library
    - (if req open terminal and run the following commands)
    ```
    sudo pip install opencv-contrib-python
    ```

 - pyrealsense2 library
    - (if req open terminal and run the following commands)
    ```
    pip install pyrealsense2
    ```

 - ROS Noetic
    - Follow the instructions on the official ROS website to install ROS Noetic
    - Link: https://wiki.ros.org/noetic/Installation/Ubuntu

 - RVIZ/Gazebo
    - Follow the instructions on the official ROS website to install RVIZ/Gazebo


## Software Structure:
 - main.py - working code using realsense's hw depth data, uses YOLOv8 to obtain pixel coordinates, converts to mm and uses epipolar geometry to calc real world X, Y, Z
 - object_det_yolov8.py - working code used by main.py. Detects bottle using YOLOv8 net, gets bounding box coordinates, calc center of bounding box, draws the result using openCV
 - debug.py - same as main.py (does the same fxn), has some experimental lines of code to compute depth without using realsense's hw and use openCV instead