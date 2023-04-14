---
title: ZED2 with ORB-SLAM3 (Stereo-IMU mode) step-by-step
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2023-14-04 11:11:14 +0700
categories: [Computer Vision]
tags: [Tutorial]
img_path: /assets/img/post_assest/
render_with_liquid: false
---


## Introduction

It is also a project requirement, and needs to use visual inertial navigation to do some development, so the first step is to do some algorithm testing-simulation and physical testing. After passing the simulation test results, it is finally decided to useORB-SLAM3 to complete the task.

- System version:<b> Ubuntu 20.04, ROS Noetic </b>

## Contents

- Introduction
- Cuda and CuDNN installation
- ZED SDK installation
- Binocular camera, IMU calibration and joint calibration
- Algorithm operation results display
- Discussion

## Cuda and CuDNN installation

- CUDA Toolkit Install
- CuDNN install

## ZED SDK

- ZED SDK
- ZED Python API
- ZED ROS Wrapper

## Sensor Calibration

### Install Calibration tool

- Install dependencies

```bash
sudo apt-get install python-setuptools python rosinstall ipython libeigen3-dev libboost-all-dev doxygen libopencv-dev ros-noetic-vision-opencv ros-noetic-image-transport-plugins ros-noetic-cmake-modules software-properties-common libpoco-dev python-matplotlib python-scipy python-git python-pip libtbb-dev liblapack-dev python-catkin-tool libv4l-dev

sudo add-apt-repository ppa:igraph/ppa
sudo apt-get update
sudo apt-get install python-igraph

```

- Create a new workspace, remember to add the environment variable source ~/kalibr_workspace/devel/setup.bash in bashrc

```bash
mkdir -p ~/kalibr_ws/src
cd ~/kalibr_ws
catkin_make
```

- Download the source code and compile

```bash

cd ~/kalibr_ws/src

git clone -b 20 https://github.com/ethz-asl/kalibr.git

catkin build -DCMAKE_BUILD_TYPE=Release -j4

```

- Download and compile code_utils (here to compile under the workspace created by catkin_make, mine is catkin_ws)

```bash

cd ~/catkin_ws/src

git clone hhttps://github.com/gaowenliang/code_utils.git

sudo apt-get install libw-dev

cd ../

catkin_make

```

- Download and compile imu_utils

```bash
git clone https://github.com/geowenliang/imu_utils
cd ../
catkin_make
```

### Download the calibration board and yaml file

Enter GitHub to download relevant parameter files https://github.com/ethz-asl/kalibr/wiki/downloads#calibration-targets
Select Aprilgrid 6x6 0.8x0.8 m (A0 page), download its pdf and yaml files

![Fig.1](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/tag.png?raw=true)


The following is the yaml file Contents:
Among them The <i> codeOffset </i> parameter is useless and can be deleted, tagSize is the side length of the large black square, tagSpacing is the side length of the small black square/the side length of the large black square.

```bash
terget_type: 'aprilgrid'
tagCols: 6
tagRows: 6
tagSize: 0.088
tagSpacing: 0.3
codeoffset: 0
```

### ZED 2 recording bag ready for calibration

Found under the catkin_ws/src/zed-ros-wrapper/zed_wrapper/params foldercommon.yaml (parameter configuration file), which can configure the camera output resolution, <b> I set the resolution to 2, and the resolution size is 1280 720, but in fact, the image resolution obtained by subscribing to the ros topic during calibration is half of this, that is, 640 360 </b>

Start ZED 2:

```bash
roslaunch zed_wrapper zed2.launch
rqt_image_view # for viewing image topic
```

The default camera release frequency is 15Hz, and the IMU release frequency is 200Hz. Next, we will start recording the bag package for calibration. 

#### Note

(1) Note that when recording, the full picture of the QR code should be exposed in the field of view of the two cameras of the binocular camera, which can be viewed through rqt_image_view. 

(2) Fully move the QR code to every corner of the camera's field of view.

(3) Move three rounds along the three axes of the camera, forward, backward, left, and right, and swing back and forth around the three axes for three rounds. 

The specific operations can be performed first. 

```bash
rosbag record -o Kalib_data_HD720.bag /zed2/zed_node/imu/data_raw /zed2/zed_node/left/image_rect_color /zed2/zed_node/right/image_rect_color 
```

A total of three topics are recorded, camera images and IMU's. There are many people on the Internet who say that the frequency should be lowered, but it is not necessary, just record directly.

### Camera Calibration

Remember Kalib_data_HD720.bag (recorded bag package) and april.yaml (calibration board parameter file)Replace it with the path where your actual location is, It will be marked in a while. Anyway, I did not report an error. If I did report an error, I suggest re-recording. During the calibration process, you can visualize whether the corner detection is good, and find that there are serious errors in corner reprojection; --approx-sync 0.04, where 0.04 can be adjusted to 0.1 according to the situation, and the function is to synchronize the data of each camera.

```bash
roscore
rosrun kalibr kalibr_calibrate_cameras --bag Kalib_data_HD720.bag --topic /zed2/zed_node/left/image_rect_color /zed2/zed_node/right/image_rect_color --models pinhole-radtan pinhole-radtan --target april.yaml
```

After the calibration is completed, the camera's internal reference and other files will be obtained, among which the yaml file can be used for joint calibration.


###  IMU Calibration

(1) Manual Calibration 

```bash
rosbag record -o imu_calibration /zed2/zed_node/imu/data_raw
```

- To record for more than two hours, create a launch file named ZED2_calibration.launch

```bash
<launch>
    <node pkg="imu_utils" type="imu_an" name="imu_an" output="screen">
        <param name="imu_topic" type="string" value= "/zed2/zed_node/imu/data_raw"/>
        <param name="imu_name" type="string" value= "ZED2"/>
        <param name="data_save_path" type="string" value= "$(find imu_utils)/data/"/>
        <param name="max_time_min" type="int" value= "120"/>
        <param name="max_cluster" type="int" value= "200"/>
    </node>
</launch>
```

- Start calibration

```bash
roslaunch imu_utils ZED2_calibration.launch
rosbag play -r 200 imu_calibration.bag  
```

Finally, go to the calibration result file to create the corresponding imu.yaml (take the average value of the calibration results Acc and Gyr and fill in the imu.yaml file)

(2) directly use the official data to create imu.yaml and fill it in

```bash
#Accelerometers
accelerometer_noise_density: 1.4e-03   #Noise density (continuous-time)
accelerometer_random_walk:   8.0e-05   #Bias random walk
 
#Gyroscopes
gyroscope_noise_density:     8.6e-05   #Noise density (continuous-time)
gyroscope_random_walk:       2.2e-06   #Bias random walk
 
rostopic:                    /zed2/zed_node/imu/data_raw      #the IMU ROS topic
update_rate:                200.0     #Hz (for discretization of the values above)
```

### Binocular camera/IMU joint calibration

Note that the files Kalib_data_HD720.bag, camchain-Kalib_data_HD720.yaml, imu.yaml, and april.yaml should use absolute paths as much as possible

```bash
roscore
rosrun kalibr kalibr_calibrate_imu_camera --bag Kalib_data_HD720.bag --cam camchain-Kalib_data_HD720.yaml --imu imu.yaml --target april.yaml
```

Finally, the joint calibration parameters Kalib_data_HD720-imu.yaml can be obtained.

## ZED 2 runs ORB-SLAM3

### Compile and install ORB-SLAM3

Let’s directly quote the content of a blogger here. I have configured it on three computer devices. In fact, one of the problems is <b> environment variable </b>, add export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/home/<user_name>/ORB_SLAM3-master/Examples_old/ROS in bashrc (the path should be replaced by your own), one ismissing dependencies, such as the Sophus library. Others can be installed according to the error information. The official website is very detailed. There is another problem.CmakeList fileWhen compiling, on the one hand, the opencv pointer needs to be changed. In the CmakeList file for compiling ros, the compiling part of the .cc file related to AR is deleted, and there are basically no other problems later.

```bash
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
cd ORB_SLAM3-master
chmod +x build.sh
./build.sh

cd ORB_SLAM3-master/Examples_old/ROS/ORB_SLAM3
mkdir build
cd build
cmake ..
make
```

Add the configuration file below

```bash
cd  ORB_SLAM3-master/Examples_old/ROS/ORB_SLAM3/src
touch zed2_stereo_inertial.cc
touch zed2_stereo_inertial.yaml
```

Add the following information to the zed2_stereo_inertial.cc file

```c++ 

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<vector>
#include<queue>
#include<thread>
#include<mutex>

#include<ros/ros.h>
#include<cv_bridge/cv_bridge.h>
#include<sensor_msgs/Imu.h>

#include<opencv2/core/core.hpp>

#include"../../../include/System.h"
#include"include/ImuTypes.h"

using namespace std;

class ImuGrabber
{
public:
    ImuGrabber(){};
    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

    queue<sensor_msgs::ImuConstPtr> imuBuf;
    std::mutex mBufMutex;
};

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, ImuGrabber *pImuGb, const bool bRect, const bool bClahe): mpSLAM(pSLAM), mpImuGb(pImuGb), do_rectify(bRect), mbClahe(bClahe){}

    void GrabImageLeft(const sensor_msgs::ImageConstPtr& msg);
    void GrabImageRight(const sensor_msgs::ImageConstPtr& msg);
    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);
    void SyncWithImu();

    queue<sensor_msgs::ImageConstPtr> imgLeftBuf, imgRightBuf;
    std::mutex mBufMutexLeft,mBufMutexRight;
   
    ORB_SLAM3::System* mpSLAM;
    ImuGrabber *mpImuGb;

    const bool do_rectify;
    cv::Mat M1l,M2l,M1r,M2r;

    const bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
};



int main(int argc, char **argv)
{
  ros::init(argc, argv, "Stereo_Inertial");
  ros::NodeHandle n("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
  bool bEqual = false;
  if(argc < 4 || argc > 5)
  {
    cerr << endl << "Usage: rosrun ORB_SLAM3 Stereo_Inertial path_to_vocabulary path_to_settings do_rectify [do_equalize]" << endl;
    ros::shutdown();
    return 1;
  }

  std::string sbRect(argv[3]);
  if(argc==5)
  {
    std::string sbEqual(argv[4]);
    if(sbEqual == "true")
      bEqual = true;
  }

  // Create SLAM system. It initializes all system threads and gets ready to process frames.
  ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::IMU_STEREO,true);

  ImuGrabber imugb;
  ImageGrabber igb(&SLAM,&imugb,sbRect == "true",bEqual);
  
    if(igb.do_rectify)
    {      
        // Load settings related to stereo calibration
        cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
        if(!fsSettings.isOpened())
        {
            cerr << "ERROR: Wrong path to settings" << endl;
            return -1;
        }

        cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
        fsSettings["LEFT.K"] >> K_l;
        fsSettings["RIGHT.K"] >> K_r;

        fsSettings["LEFT.P"] >> P_l;
        fsSettings["RIGHT.P"] >> P_r;

        fsSettings["LEFT.R"] >> R_l;
        fsSettings["RIGHT.R"] >> R_r;

        fsSettings["LEFT.D"] >> D_l;
        fsSettings["RIGHT.D"] >> D_r;

        int rows_l = fsSettings["LEFT.height"];
        int cols_l = fsSettings["LEFT.width"];
        int rows_r = fsSettings["RIGHT.height"];
        int cols_r = fsSettings["RIGHT.width"];

        if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
                rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
        {
            cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
            return -1;
        }

        cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,igb.M1l,igb.M2l);
        cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,igb.M1r,igb.M2r);
    }

  // Maximum delay, 5 seconds
  //ros::Subscriber sub_imu = n.subscribe("/imu", 1000, &ImuGrabber::GrabImu, &imugb);
  //ros::Subscriber sub_img_left = n.subscribe("/camera/left/image_raw", 100, &ImageGrabber::GrabImageLeft,&igb);
  //ros::Subscriber sub_img_right = n.subscribe("/camera/right/image_raw", 100, &ImageGrabber::GrabImageRight,&igb);

  ros::Subscriber sub_imu = n.subscribe("/zed2/zed_node/imu/data_raw", 1000, &ImuGrabber::GrabImu, &imugb);
  ros::Subscriber sub_img_left = n.subscribe("/zed2/zed_node/left/image_rect_color", 100, &ImageGrabber::GrabImageLeft,&igb);
  ros::Subscriber sub_img_right = n.subscribe("/zed2/zed_node/right/image_rect_color", 100, &ImageGrabber::GrabImageRight,&igb);



  std::thread sync_thread(&ImageGrabber::SyncWithImu,&igb);

  ros::spin();

  return 0;
}



void ImageGrabber::GrabImageLeft(const sensor_msgs::ImageConstPtr &img_msg)
{
  mBufMutexLeft.lock();
  if (!imgLeftBuf.empty())
    imgLeftBuf.pop();
  imgLeftBuf.push(img_msg);
  mBufMutexLeft.unlock();
}

void ImageGrabber::GrabImageRight(const sensor_msgs::ImageConstPtr &img_msg)
{
  mBufMutexRight.lock();
  if (!imgRightBuf.empty())
    imgRightBuf.pop();
  imgRightBuf.push(img_msg);
  mBufMutexRight.unlock();
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg)
{
  // Copy the ros image message to cv::Mat.
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::MONO8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }
  
  if(cv_ptr->image.type()==0)
  {
    return cv_ptr->image.clone();
  }
  else
  {
    std::cout << "Error type" << std::endl;
    return cv_ptr->image.clone();
  }
}

void ImageGrabber::SyncWithImu()
{
  const double maxTimeDiff = 0.01;
  while(1)
  {
    cv::Mat imLeft, imRight;
    double tImLeft = 0, tImRight = 0;
    if (!imgLeftBuf.empty()&&!imgRightBuf.empty()&&!mpImuGb->imuBuf.empty())
    {
      tImLeft = imgLeftBuf.front()->header.stamp.toSec();
      tImRight = imgRightBuf.front()->header.stamp.toSec();

      this->mBufMutexRight.lock();
      while((tImLeft-tImRight)>maxTimeDiff && imgRightBuf.size()>1)
      {
        imgRightBuf.pop();
        tImRight = imgRightBuf.front()->header.stamp.toSec();
      }
      this->mBufMutexRight.unlock();

      this->mBufMutexLeft.lock();
      while((tImRight-tImLeft)>maxTimeDiff && imgLeftBuf.size()>1)
      {
        imgLeftBuf.pop();
        tImLeft = imgLeftBuf.front()->header.stamp.toSec();
      }
      this->mBufMutexLeft.unlock();

      if((tImLeft-tImRight)>maxTimeDiff || (tImRight-tImLeft)>maxTimeDiff)
      {
        // std::cout << "big time difference" << std::endl;
        continue;
      }
      if(tImLeft>mpImuGb->imuBuf.back()->header.stamp.toSec())
          continue;

      this->mBufMutexLeft.lock();
      imLeft = GetImage(imgLeftBuf.front());
      imgLeftBuf.pop();
      this->mBufMutexLeft.unlock();

      this->mBufMutexRight.lock();
      imRight = GetImage(imgRightBuf.front());
      imgRightBuf.pop();
      this->mBufMutexRight.unlock();

      vector<ORB_SLAM3::IMU::Point> vImuMeas;
      mpImuGb->mBufMutex.lock();
      if(!mpImuGb->imuBuf.empty())
      {
        // Load imu measurements from buffer
        vImuMeas.clear();
        while(!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec()<=tImLeft)
        {
          double t = mpImuGb->imuBuf.front()->header.stamp.toSec();
          cv::Point3f acc(mpImuGb->imuBuf.front()->linear_acceleration.x, mpImuGb->imuBuf.front()->linear_acceleration.y, mpImuGb->imuBuf.front()->linear_acceleration.z);
          cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x, mpImuGb->imuBuf.front()->angular_velocity.y, mpImuGb->imuBuf.front()->angular_velocity.z);
          vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc,gyr,t));
          mpImuGb->imuBuf.pop();
        }
      }
      mpImuGb->mBufMutex.unlock();
      if(mbClahe)
      {
        mClahe->apply(imLeft,imLeft);
        mClahe->apply(imRight,imRight);
      }

      if(do_rectify)
      {
        cv::remap(imLeft,imLeft,M1l,M2l,cv::INTER_LINEAR);
        cv::remap(imRight,imRight,M1r,M2r,cv::INTER_LINEAR);
      }

      mpSLAM->TrackStereo(imLeft,imRight,tImLeft,vImuMeas);

      std::chrono::milliseconds tSleep(1);
      std::this_thread::sleep_for(tSleep);
    }
  }
}

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
  mBufMutex.lock();
  imuBuf.push(imu_msg);
  mBufMutex.unlock();
  return;
}
```

Add the following information in zed2_stereo_inertial.yaml (here are my camera parameters, you can modify it according to your own situation)

I directly use the DKRP data in the camera_info topic of the original image of the ZED left and right eyes, you can find it in the rostopic list , In addition, the zed2 camera baseline length is 120. 


```bash

%YAML:1.0

#--------------------------------------------------------------------------------------------
# Camera Parameters. Adjust them!
#--------------------------------------------------------------------------------------------
Camera.type: "PinHole"

# Camera calibration and distortion parameters (OpenCV) (equal for both cameras after stereo rectification)
Camera.fx: 256.5277384633465
Camera.fy: 258.08249705047217
Camera.cx: 325.50319459226085
Camera.cy: 180.96517806223522


Camera.k1: -0.020457937535071292
Camera.k2: 0.01104746035697357
Camera.p1: 0.00020521550183980535
Camera.p2: -0.0015638016748186173

Camera.width: 640
Camera.height: 360

# Camera frames per second
Camera.fps: 15.0

# stereo baseline times fx 
Camera.bf: 30.7824

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

# Close/Far threshold. Baseline times.
ThDepth: 40.0 # 35

# Transformation from camera 0 to body-frame (imu)

Tbc: !!opencv-matrix
        rows: 4
        cols: 4
        dt: f
        data: [ 0.011189492088057723, -0.005170435177808852, 0.999924028047573, -0.030171769239718378,
                -0.9999089154286149, -0.0076051555206256005, 0.011149998021449559, +0.006834916768468505,
                0.007546927400109649, -0.9999577132107034, -0.0052550620584632945, -0.018995636408175094,
                0.0, 0.0, 0.0, 1.0]

# IMU noise
# get it from Project of **zed-examples/tutorials/tutorial 7 - sensor data/**.
IMU.NoiseGyro: 8.6e-05 # 1.6968e-04
IMU.NoiseAcc:  0.0014 # 2.0000e-3
IMU.GyroWalk:  2.2e-06
IMU.AccWalk:   8.0e-05 # 3.0000e-3
IMU.Frequency: 200

#--------------------------------------------------------------------------------------------
# Stereo Rectification. Only if you need to pre-rectify the images.
# Camera.fx, .fy, etc must be the same as in LEFT.P
#--------------------------------------------------------------------------------------------
LEFT.height: 360
LEFT.width: 640
LEFT.D: !!opencv-matrix
        rows: 1
        cols: 5
        dt: d
        data: [-0.040750399231910706, 0.009019049815833569, -0.004655580036342144, -0.0006361529813148081, 0.0003129479882773012]
LEFT.K: !!opencv-matrix
        rows: 3
        cols: 3
        dt: d
        data: [264.1424865722656, 0.0, 328.0299987792969, 0.0, 263.9525146484375, 180.45175170898438, 0.0, 0.0, 1.0]
LEFT.R:  !!opencv-matrix
        rows: 3
        cols: 3
        dt: d
        data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
LEFT.Rf:  !!opencv-matrix
        rows: 3
        cols: 3
        dt: f
        data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
LEFT.P:  !!opencv-matrix
        rows: 3
        cols: 4
        dt: d
        data: [264.1424865722656, 0.0, 328.0299987792969, 0.0, 0.0, 263.9525146484375, 180.45175170898438, 0.0, 0.0, 0.0, 1.0, 0.0]

RIGHT.height: 360
RIGHT.width: 640
RIGHT.D: !!opencv-matrix
        rows: 1
        cols: 5
        dt: d
        data: [-0.03843430057168007, 0.005912320222705603, -0.0034095800947397947, 6.041819870006293e-05, -0.00011238799925195053]
RIGHT.K: !!opencv-matrix
        rows: 3
        cols: 3
        dt: d
        data: [263.0425109863281, 0.0, 326.2799987792969, 0.0, 262.93499755859375, 180.3209991455078, 0.0, 0.0, 1.0]
RIGHT.R:  !!opencv-matrix
        rows: 3
        cols: 3
        dt: d
        data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
RIGHT.P:  !!opencv-matrix
        rows: 3
        cols: 4
        dt: d
        data: [263.0425109863281, 0.0, 326.2799987792969, -31.668317794799805, 0.0, 262.93499755859375, 180.3209991455078, 0.0, 0.0, 0.0, 1.0, 0.0]

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 2200

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
# Image is divided in a grid. At each cell FAST are extracted imposing a minimum response.
# Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST
# You can lower these values if your images have low contrast
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1
Viewer.GraphLineWidth: 0.9
Viewer.PointSize:2
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500
```

Edit the CMakeLists.txt file

```bash
cd ORB_SLAM3-master/Examples_old/ROS/ORB_SLAM3
gedit CMakeLists.txt
```

Add at the bottom

```bash
rosbuild_add_executable(zed2_stereo_inertial
    src/zed2_stereo_inertial.cc
)

target_link_libraries(zed2_stereo_inertial
    ${LIBS}
)
```

Just recompile, if the executable file cannot be found, run

```bash
rospack profile

```

## Algorithm operation results display


## Discussion
