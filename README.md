# People-detection-tracking-and-recognition-using-OpenCV
Usage: This project contains a python script called people_detection.py, a folder named haarcascades and two test videos. The download links for the two test videos are: 
Test_video_1.mp4: https://github.com/intel-iot-devkit/sample-videos/blob/master/worker-zone-detection.mp4 
Test_video_2.mp4: https://www.pexels.com/video/man-transporting-garden-sand-using-a-cart-7714807/  

You need execute the following statement to run the python script.  > python3 people_detection.py -v ./test_video_1.mp4 -face_detec False  The python script needs to pass in two parameters, namely “-v” and “-face_detec”, where “-v” refers to the video path that needs to be tested, and “-face_detec” refers to the need for face detection(If there is no face in the video, set to False). This python script first modifies the shape of each frame in the video, then performs corner detection, and then performs optical flow tracking on the detected corners, then execute human detection and face detection in turn, and draws frames and ellipses on the image respectively in. 
