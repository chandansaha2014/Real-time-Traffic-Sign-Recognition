# Real-Time Traffic Sign Recognition
Traffic Sign classification requires a high degree of accuracy such that it is viable to use in autonomous vehicles.

![intro](/img/traffic_sign.png)

we try to explore the different features of traffic sign images and use state of the art machine learning
algorithms to classify traffic signs for autonomous cars. Our model is a variation of VGG-16 which uses depth of the image to accurately classify image.
  
 *[Final Report](ML_Report_Traffic_sign_classification.pdf) can be found here.* 
## Dataset 
[Lisa dataset](http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html) is a collection of annotated im-
ages and videos containing traffic signs and videos containing traffic signs. It is comprised of over 6,000 frames that con-
tain over 7,000 signs of 47 different types.

## Architecture 

The model is based on VGGNet. Overall High level architecture of our model is given below, 

![Architecture](/img/architecture.png)

We have used [VGGNet pre- trained model](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3) to speed up training process
## Results
Data Set | Number of Classes | Accuracy | Testing time | Train Time(per epoch)
------------ | ------------- | ------------- | ------------- | -------------
GTSRB  | 43 | 99.3% | 15.3 ms | 210 sec
GTSRB(GPU)  | 43 | 99.31% | 3.5 ms | 45 sec
LISA  | 16 | 98.7% | 14.6 ms | 33 sec
LISA(GPU)  | 16 | 98.76% | 3.5 ms | 3 sec



We see a considerable difference between the training and testing time of CPU and GPU Convolutional Neural Net-
work. With the German Traffic Sign dataset we got an accuracy of 99.31% and with LISA dataset 98.7%. These
differences are due to the imaging conditions like weather, speed of the vehicle, occlusions and light conditions.

## Conclusion

Some of the images are not clear and to improve our model , we plan to integrate object segmentation with classification to accurately identify traffic signs on real-time.


## References

[1] Simonyan, Karen, and Andrew Zisserman. ”Very deep convolutional networks for large-scale image
recognition.” arXiv preprint arXiv:1409.1556 (2014).

[2] P Sermanet , Y LeCun . Traffic Sign Recognition with Multi-Scale Convolutional Networks .

[3] VGGNet - VggNet Github (https://github.com/machrisaa/tensorflow-vgg)

