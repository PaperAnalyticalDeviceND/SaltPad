# SaltPad
Salt Pad analysis software

Information on the Paper Analytical Device Project can be found here: http://padproject.nd.edu

The PAD repository contains a mixture of Python and C++ code to allow the analysis of the Salt Pads, from Notre Dame University, for detection of drug contents. The C++ code must be compiled, on Linux/OSX (assuming OpenCV libraries installed):
>cmake .
>make

To run the code:
>python saltpad.py —o auto -t template2.png test/6416.jpg

where file to be processed, test/6416.jpg, template file, -t template2.png, results CSV file to test/6416.csv, —o auto.

Run ‘python saltpad.py’ to get a full list of parameters.

Now added gui.py to provide a basic GUI to interact with the analysis software. To run
>python gui.py

or

>./gui.py

Now has the ability to convert the raw data to ppm Iodine through the calibration file 'calibration.csv'. Instruction 
on the use of the file are included in the file. Basic instuctions:

Calibration file for salt PAD
First identify the calibration set to use
>selected,ND-lab,

Calibration section must start with 'calibration' folowed bt a unique identifier for the calibration set
Note comma after calibration identifier and NO SPACES in the name and no space in front of the name..
format is: number of wells, well 1 x, well 1 y, ... , well n x, well n y, offset, divisor
>calibration,ND-lab,

>3, 0,2, 1,0, 1,1, 125.37, 17.068
