import argparse
import cv2 as cv
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression

subject_label = 1
font = cv.FONT_HERSHEY_SIMPLEX
list_of_videos = []
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

output_image_width = 500


def detect_people(frame):
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(12, 12), scale=1.04)
    for i in range(len(rects)):
        rects[i][2] += rects[i][0]
        rects[i][3] += rects[i][1]
    rectsd = non_max_suppression(rects, probs=None, overlapThresh=0.3)
    for (x, y, w, h) in rectsd:
        center = (x + (w-x) // 2, y + (h-y) // 2)
        cv.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
        frame = cv.circle(frame, center, 3, (255, 0, 0), 4)
    return frame, rects


def ShiTomasi_corner_detection(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    feature_params = dict(maxCorners=80, qualityLevel=0.01,
                          minDistance=10, blockSize=7)
    corners = cv.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
    return corners


def optical_flow(pre_frame, current_frame, hsv):
    pre_frame_grey = cv.cvtColor(pre_frame, cv.COLOR_BGR2GRAY)
    current_frame_grey = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(pre_frame_grey, current_frame_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def detect_face(frame):
    face_cascade = cv.CascadeClassifier()
    face_cascade.load(cv.samples.findFile("../haarcascades/haarcascade_frontalface_alt.xml"))
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
    return frame


if __name__ == '__main__':
    # Parameters passed in from outside
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", required=True, help="path to video", default="../test_videos/test_video.mp4")
    ap.add_argument("-face_detec", required=False, help="face detection or not", default=False)
    args = vars(ap.parse_args())
    path = args["v"]
    face_detec = args["face_detec"]
    print ("\033[32m Opening video: \033[37m", path)
    print ("\033[32m Opening video: \033[37m", face_detec)
    # open test video
    camera = cv.VideoCapture(path)

    grabbed, frame = camera.read()
    print("\033[32m Origin video shape: \033[37m", frame.shape)
    # resize image
    frame_resized = imutils.resize(frame, width=min(output_image_width, frame.shape[1]))
    # convert from RGB to GRAY
    frame_resized_grayscale = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)
    print("\033[32m Resized video shape: \033[37m", frame_resized.shape)
    current_frame = 0
    hsv = np.zeros_like(frame_resized)
    hsv[..., 1] = 255

    color = np.random.randint(0, 255, (100, 3))
    # save video
    # fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    # video_saver = cv.VideoWriter('../result_videos/output.mp4', fourcc, 20, (frame_resized.shape[1], frame_resized.shape[0]))
    
    # ShiTomasi corner detection function
    p0 = ShiTomasi_corner_detection(frame_resized)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame_resized)
    previous_frame = frame_resized_grayscale.copy()
    while True:
        read, frame = camera.read()
        if not read:
            break
        # resize image frame
        frame_resized = imutils.resize(frame, width=min(output_image_width, frame.shape[1]))
        frame_resized_grayscale = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)
        # people detection function
        frame_processed, rects = detect_people(frame_resized)

        if face_detec == "True":
            frame_processed = detect_face(frame_processed)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(previous_frame, frame_resized_grayscale, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            draw = False
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            # print (rects)
            for (x, y, w, h) in rects:
                if x < x_new < w and y < y_new < h:
                    draw = True
            if draw:
                mask = cv.line(mask, (int(x_new), int(y_new)), (int(x_old), int(y_old)), color[i].tolist(), 2)
                # frame_resized = cv.circle(frame_resized, (int(x_new), int(y_new)), 5, color[i].tolist(), -1)
        img = cv.add(frame_resized, mask)
        cv.imshow('frame', img)
        # video Writer
        # if current_frame >= 0:
        #     if 690 >= current_frame:
        #         video_saver.write(img)
        #     else:
        #         video_saver.release()
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv.imwrite("result_"+str(current_frame)+".png", img)

        # Now update the previous frame and previous points
        p0 = good_new.reshape(-1, 1, 2)
        previous_frame = frame_resized_grayscale.copy()
        current_frame += 1
    camera.release()
    cv.destroyAllWindows()