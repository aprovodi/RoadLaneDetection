This lane detection code is based on Hough Transform and example from OpenCV Cookbook.
The key idea is to detect lines using image processing and Hough transform. After that lines are split into groups by distance and angle, and mean lanes are intersected with bottom x-axis to identify left and right lane.

Executable file accepts full path (where images are located) as an argument. The output are processed '.png' files and 'intercepts.csv' file which is described in task (but might give negative and bigger than width values for extended line).

Inverse perspective transform was also considered (and tried) in this scope, but didn't give better results.
Things to do:
1. Consider previous frames for lane detection (use Kalman filter).
2. Use vanishing point to identify proper lanes (because this algorithm searches for lines which are closest to the bottom-center from both sides).
3. Consider learning algorithms for this task, as current implementation might be tricky and error-prone in different weather and lighting conditions.
