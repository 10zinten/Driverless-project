# ABSTRACT for NT-WoC

This project is part of driverless project initiated by CamberRacing Team intended for Student Formula Driverless competition. As a driverless domain lead, i have been working on Cone Detection for driverless vehicle on race track using SSD-Mobilenet.

So we have worked on a prototype with Raspberry Pi installed and using donkey framework which has steering  and throttle value prediction CNN model and we have successfully trained this basic model and achieved good  performance in testing environment. After that we planned to extend the feature of prototype to have object detection to detect cone and semantic segmentation to segment out the race track from surrounding to have  region of interest over race track, which will help to improve prediction accuracy on steering angle and  throttle value. So I have been working on SSD-Mobilenet for cone detection, since it is the most suitable  for prototype running on Raspberry Pi. We have already collected and annotated enough dataset to training  the SSD-Mobilenet and almost all the pipeline of model completed except two things:
  1. Implementing [Data Augmentation and Image Transformation](http://www.telesens.co/2018/06/28/data-augmentation-in-ssd/)
  1. Implementing [mAP (mean Average Precision) metric](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173).

For this Winter of Code I would like be mentor for any interested  student who would like to contribute to above two tasks and the source code is in pure tensorflow. If student could contribute to the project then we will consider hiring the respective student in CamberRacing under Driverless project.


## Selection procedure:
Complete any of the given project:
  1. Implement at least 3 Data augmentation techniques for SSD
  1. Implement simple Precision Vs Recall Graph for Image classification
