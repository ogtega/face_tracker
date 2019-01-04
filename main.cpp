#include <stdio.h>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;
using namespace dnn;

bool detect(Mat &frame, double conf, Rect2d &bbox, Net &net);

int main(int, char **)
{
    Mat frame;
    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap;
    // select any API backend
    int deviceID = 0;        // 0 = open default camera
    int apiID = cv::CAP_ANY; // 0 = autodetect default API
    // open selected camera using selected API
    cap.open(deviceID + apiID);
    // check if we succeeded
    if (!cap.isOpened())
    {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    Ptr<Tracker> tracker;
    Rect2d bbox = Rect2d(0, 0, 0, 0);

    //--- LOAD HUMAN FACE CLASSIFYING NETWORK
    Net net = readNetFromCaffe("caffe/deploy.prototxt", "caffe/model.caffemodel");

    //--- GRAB AND WRITE LOOP
    cout << "Start grabbing" << endl
         << "Press 'esc' key to terminate" << endl;

    for (;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);
        // check if we succeeded
        if (frame.empty())
        {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }

        if (bbox.area() <= 0)
        {
            if (detect(frame, .5, bbox, net))
            {
                tracker = TrackerKCF::create();
                tracker->init(frame, bbox);
            }
        }
        else if (!tracker->update(frame, bbox))
        {
            bbox = Rect2d(0, 0, 0, 0);
        }

        rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
        // show live and wait for 'esc' key with timeout long enough to show images
        imshow("Live Feed", frame);
        if (waitKey(5) == 27)
            break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

bool detect(Mat &frame, double conf, Rect2d &bbox, Net &net)
{
    double confidence;
    Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 117, 123), false, false);

    net.setInput(blob, "data");
    Mat output = net.forward("detection_out");
    Mat detections(output.size[2], output.size[3], CV_32F, output.ptr<float>());

    for (int i = 0; i < detections.rows; i++)
    {
        float confidence = detections.at<float>(i, 2);

        if (confidence > conf)
        {
            int x1 = static_cast<int>(detections.at<float>(i, 3) * frame.cols);
            int y1 = static_cast<int>(detections.at<float>(i, 4) * frame.rows);
            int x2 = static_cast<int>(detections.at<float>(i, 5) * frame.cols);
            int y2 = static_cast<int>(detections.at<float>(i, 6) * frame.rows);

            bbox = Rect2d(x1, y1, x2 - x1, y2 - y1);
            return true;
        }
    }

    return false;
}
