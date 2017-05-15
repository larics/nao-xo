nao-xo
======

Nao plays tic-tac-toe againts human opponent.

## Run the program
To play, position the robot in front of the playing field and run:

    ./main_xo.py -i <robot address>

e.g.

    ./main_xo.py -i herrflick.local
**Note**: Works with naoqi v2.1. Behaviors from the choregraphe folder must be installed on the robot!

## Dependencies 
In order to run the program, you need:

 - Ubuntu (could work on Windows, tested on Ubuntu 14.04 and Ubuntu 16.04)
 - python 2.7.x (tested with 2.7.6 on Ubuntu 14.04 and 2.7.12 on Ubuntu 16.04)
 - pynaoqi v2.1
 - python OpenCV bindings (tested with OpenCV 2.4.9., there may be issues with OpenCV3.0+)
 - numpy

## Pre-game setup
In order to play with the robot, you need:

 1. Playing field
 2. Calibrated camera
 3. Red and yellow objects that represent the crosses (red) and the noughts (yellow)
 4. Robot assistant

### Printing and setting up the playing field
The robot plays by detecting the lines of the playing field. For the robot to do so, you need to prepare a table on which the robot will play and then put the playing field on top of the table. The table should not be too low or too high, as the workspace of the robot is limited. We use the box NAO came in which is about 30 cm tall.

We prepare the field by drawing the grid in some of the drawing tools (i.e. Visio if you use MS tools, LibreOffice Draw on Ubuntu). It is important to use a tool that lets you specify distance between the lines of the grid in metric units. Make sure all distances are equal. We use the field where distances between the lines are 6 cm. You can use smaller grid, but take into account the precision of NAO's hands. Larger grids are almost infeasible as NAO cannot reach the upper boxes of the playing field.

Once you print the grid, you need to specify the dimensions by using the `setFieldSize` method in `nao_xo.py`. Make sure to pass the value in meters. The default is 0.06 (6 cm). Also, make sure that the lines are sharp, especially around the intersections, as we use the intersections to calculate the position of the field. 

### Camera calibration
Camera calibration parameters are used to extract the 3D position of the playing field with respect to the robot frame. You can perform camera calibration by following one of the many available guides (such as [this one](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html)), although the default parameters may work well.

The parameters are stored in the `naoxo_definitions.py` in the variable `intrinsic_params`. The format is `[fx, fy, cx, cy, s]`.

If the parameters are okay, and all intersections are calculated correctly, and the field size is correctly set, you should see a green box around the playing field in one of the images that are displayed when you run `main_xo.py`.

### Objects to play with
The robot and its human opponent play by placing the objects in the boxes of the playing field. As already states, we use red crosses and yellow noughts which we carved out of erasers. To use objects of different colors, you need to change the code in the `imgproc_xo.py`,  namely `getContours_O` and `getContours_X` methods. Both of them use segmentation in HSV color space. 

### Assisting the robot
Since we use a larger field to account for imprecision in the end-effector placement by the default NAO kinematics, the robot cannot reach all the boxes on the playing field with one hand. Therefore, NAO plays with left hand for the 3 leftmost boxes of the playing field, and with the right hand for other boxes. It helps if you place the playing field to be slightly more to the right of the robot instead of it being dead center with respect to the robot.

NAO cannot take the object by itself, so it needs an assistant which will put the object in its hand before its turn. The robot signals which hand it wants to use by opening it. This sometimes fails if the hand was already open so make sure to check where the robot wants to play (boxes 1,4,7 should be played with the left hand, others are played with the right hand)

## Playing with the robot
Put the robot in squatting pose in front of the box. Connect to the robot, preferably using cable, and run the main program:

    ./main_xo.py -i <robot address>

The robot should stand up and a bunch of windows will appear on your screen:

 - **Grayscale**: displays the grayscale image. Make sure edges of the lines are sharp in this image
 - **Edges**: displays the edges in the image. Make sure the lines of the field are correctly identified.
 - **Lines**: displays all lines found in the image by using red lines. Make sure there are at least a few lines detected for each line in the playing field. Do not worry if there are multiple lines detected.
 - **Intersections**: displays all valid intersections found in the image by using blue circles. Make sure all four intersection of the playing field are correctly detected. 
 - **Image**: shows original image with additional information. If all the lines and intersections are correctly identified you should see a green box around the playing field. 
 - **Game**: displays the state of the game. It works only if everything of the above is correctly performed as it needs the positions of intersections to check in which box the object is. If everything is working, make sure it updates when you put an object on the field. If it does not, you need to tinker with the object detection (`getContours_O` and `getContours_X`)

### Starting the game
If you see the green box around the playing field, and the state of the game is correctly updated, the robot is ready to play. If the robot does not see the playing field, it helps to move its head a bit to change the angle (we are not sure why).

You start the game by pressing the front tactile sensors. There are three cases that the robot is prepared for:

 1. The field is empty
 2. The field has one object on it
 3. There are more objects on the playing field

In the first case, the robot assumes that it is playing first, and uses crosses. In the second case, the robot knows that it is playing second using noughts. Yes, the crosses always go first in our case. In the third case, the robot will say that it does not know what to do with so many objects on the playing field and the program will exit. Having only one nought on the playing field will also cause the program to exit (i.e. robot cannot play second with crosses, only with noughts).

### It is the robots turn
When it is the robots turn to play, you will see something similar to this displayed in the terminal:

    [INFO ] Calculated goal position of box 5: [0.264, 0.007, 0.293]
    [INFO ] Using RHand
    [INFO ] Put object in RHand and touch arm tactile sensor

The robot should also put its hand on top of the table and open it. Then, the robots assistant needs to put the object robot is playing with and touch the tactile sensors on the hand to signal that the object is in the hand. The robot will close the hand and execute the move.

**Note:** The strategy for the robot was written by 8th-graders, the robot can lose sometimes. However, the strategy is not well documented and we are not sure how easy it would be to replace it with something else. 

### It is a humans turn
Once the robot makes its move, it waits for the human player to play. You should see something similar to this in the terminal:

    [INFO ] New state ['-', '-', 'x', '-', 'o', '-', '-', '-', '-']
    [INFO ] Old state ['-', '-', 'x', '-', 'o', '-', '-', '-', '-']
The robot is now waiting to see and increase in the number of the objects human is playing with (i.e. if human has played one move with crosses, then the robot placed the nought, the robot is now waiting to see 2 crosses on the playing field). If the human takes too much time, the robot will say that it is waiting for him to play.

When the human plays, you should see something like this:

    [INFO ] New state ['-', '-', 'x', 'x', 'o', '-', '-', '-', '-']
    [INFO ] Old state ['-', '-', 'x', '-', 'o', '-', '-', '-', '-']
   
and the robot will conclude that the human has played an once again perform the calculations for its next move. Then, you just help him with the object and repeat until the game is finished. 

### Game is over
After the game is finished, the robot will execute the behavior corresponding to the end-state of the game and the program will exit. Make sure the behaviors are installed (use Choregraphe to upload the behaviors to the robot). 
