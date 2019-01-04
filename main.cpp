#include <stdio.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;

bool detect(Mat &frame, double conf, Rect2d &bbox, CascadeClassifier &face_model);

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

    //--- LOAD HUMAN FACE CLASSIFIERS
    cv::CascadeClassifier face_model("/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml");

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
            if (detect(frame, 1.24, bbox, face_model))
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

bool detect(Mat &frame, double conf, Rect2d &bbox, CascadeClassifier &face_model)
{
    Mat gray;
    int indx = 0;
    double max = 0.0;
    vector<int> levels;
    vector<double> weights;
    vector<Rect_<int>> faces;
    cv::cvtColor(frame, gray, COLOR_BGR2GRAY);
    face_model.detectMultiScale(gray, faces, levels, weights, 1.100000000000000089, 3, 0, Size(), Size(), true);

    if (faces.size() > 1)
    {
        for (int i = 0; i < faces.size(); i++)
        {
            indx = (max >= weights[i]) ? indx : i;
            max = (max >= weights[i]) ? max : weights[i];
        }
    }

    if (max >= conf)
    {
        bbox = Rect2d(faces[indx].x, faces[indx].y, faces[indx].width, faces[indx].height);
        return true;
    }

    return false;
}