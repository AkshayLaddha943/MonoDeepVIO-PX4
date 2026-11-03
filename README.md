# DeepVIO-PX4
Simulating Drone Navigation in PX4 Environment via Traditional EKF-based Monocular VIO and extending it to CNN-Visual-Inertial Fusion

## Setup & Structure
The current setup is tested in PX4-Gazebo Harmonic environment with a satellite map overlaid on the Gazebo Surface with a downward facing Monocular Camera & IMU Sensor. 

The folder structure - 

**imu_nav:** Implementation for pure Inertial Odometry (with an option to add bias and Gaussian noise)  

**vo_nav:** Implementation of Visual Odometry, evaluated using ORB-FLANN & Sparse LK Tracker  

**vio_nav:** ESEKF-driven Visual & Inertial Fusion for Localization  

**deep_vio_nav:** Implementation of CNN-based VIO with Pose Graph Optimization  

## This repository covers the following parts of a multi-sensor fusion system

- [Visual Odometry]

- [Inertial Navigation System]

- [Traditional VIO]

- [Learning-based VIO]


## Visual Odometry
For the Visual Odometry setup, we compare the performance of ORB + FLANN_LSH Matcher with Optical Flow Tracker. For preprocessing, we apply CLAHE (Contrast Enhancement) to the image for better matching performance.



## Inertial Navigation System
A simple strap-down INS system, which integrates accelerometer data twice to get position, and gyroscope data to track attitude, further, a rotation matrix or quaternion is applied to transform accelerations from body frame to navigation frame. In order to make it more realistic, Gaussian Noise were applied to the IMU sensor (since, in simulation, it is bound to work ideally). 
