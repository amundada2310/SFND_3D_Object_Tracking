
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)//go through all Lidar POints
    {
        // assemble vector for matrix-vector-multiplication//convert x,y,z lidar coordinates to homogenous coordinates X,Y
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;//convert the X,Y ccordinated back to euclidean pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)//if != 1 then do not add that lidra point to avoid confusion 
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));//white background

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)//go through all boxes and displaying the boxes enclosed lidar point in top view with unique colors
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);//unique random number
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector<cv::DMatch> kptroi;
        
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end(); ++it1)
    {
        cv::KeyPoint kpcurr = kptsCurr.at(it1->trainIdx);
        auto kpcurr1 = kpcurr.pt;
        cv::KeyPoint kpprev = kptsPrev.at(it1->queryIdx);
        auto kpprev1 = kpprev.pt;
        if (boundingBox.roi.contains(kpcurr1))
        {
            kptroi.push_back(*it1);//all keypoints in the bounding boxes ROI- pushed back in kptroi
        }

    }

    //now removing outliers - whihc is done by calculating mean of all the matches
    double dist = 0.0;
    int m = 0;
    for (auto it2 = kptroi.begin(); it2 != kptroi.end(); ++it2)
    {
        cv::KeyPoint kpcurr2 = kptsCurr.at(it2->trainIdx);
        cv::KeyPoint kpprev2 = kptsPrev.at(it2->queryIdx);
        dist = cv::norm(kpcurr2.pt - kpprev2.pt) + dist;
        m++;
    }
    double mean = dist / m;

    //discarding the mayches beyond 2* mean
    double threshold = mean * 2, dist1 = 0;
    for (auto it3 = kptroi.begin(); it3 != kptroi.end(); ++it3)
    {
        cv::KeyPoint kpcurr3 = kptsCurr.at(it3->trainIdx);
        cv::KeyPoint kpprev3 = kptsPrev.at(it3->queryIdx);
        dist1 = cv::norm(kpcurr3.pt - kpprev3.pt);
        if (dist1 < threshold)
            boundingBox.kptMatches.push_back(*it3);
    }

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    //double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();

    //double dT = 1 / frameRate;
    //TTC = -dT / (1 - meanDistRatio);

    double medianDistRatio;
    std::sort(distRatios.begin(), distRatios.end());

    auto index = (distRatios.size() / 2);
    auto index1 = ((distRatios.size() - 1) / 2);

    if (distRatios.size() % 2 != 0)//check for even case
        medianDistRatio = distRatios[index];

    medianDistRatio = (distRatios[index1] + distRatios[index]) / 2;

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medianDistRatio);
    cout << TTC << " time for camera compute"<< endl ;
}

void sorttheLidarpoints(std::vector<LidarPoint> &lidarPoints)
{
    //sorting the lidarpoints only for value of x in ascending order
    std::sort(lidarPoints.begin(), lidarPoints.end(), [](LidarPoint a, LidarPoint b) {
        return a.x < b.x;  
        });
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dt = 1 / frameRate; //calculating time between 2 measurements
    //double lanewidth = 4.0;//assuming width of ego lane (already done in the main file)
    //double minXprev = 1e9, minXcurr = 1e9; //min distance to closet lidra point in prev and curr frames

    //remeber we only comparing x axis along the length in direction of the vehicle (ego lane)

    sorttheLidarpoints(lidarPointsPrev);//need to create a diff function to address the x component
    sorttheLidarpoints(lidarPointsCurr);

    auto index1 = (lidarPointsPrev.size()/2);//taking the median size prev
    auto index11 = ((lidarPointsPrev.size() - 1)/2);
    auto index2 = (lidarPointsCurr.size()/2);//taking the median size curr
    auto index22 = ((lidarPointsCurr.size() - 1)/2);


    double minXprev = lidarPointsPrev[index1].x;//odd
    double minXcurr = lidarPointsCurr[index2].x;//odd


    //calculate median values
    /*if (lidarPointsPrev.size() % 2 != 0)//check for even number of points
    {
        double minXprev = lidarPointsPrev[index1].x;//odd
    }
    else
    {
        double minXprev = (lidarPointsPrev[index1].x + lidarPointsPrev[index11].x) / 2.0;//even
    }

    
    if (lidarPointsCurr.size() % 2 != 0)//check for even number of points
    {
        double minXcurr = lidarPointsCurr[index2].x;//odd
    }
    else
    {
        double minXcurr = (lidarPointsCurr[index2].x + lidarPointsCurr[index22].x) / 2.0;//even
    }*/
        
    
    
    TTC = ((dt * minXcurr) / (minXprev - minXcurr));
    //TTC = -dt / (1 - medianDistRatio);

    cout << TTC << " time for lidar compute" <<endl;
    
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{

    const int size_p = prevFrame.boundingBoxes.size();
    const int size_c = currFrame.boundingBoxes.size();
    //int count[size_p][size_c] = {};//initialize a null matrix with all values "0" of bounding boxes size prev x curr
    cv::Mat count = cv::Mat::zeros(size_p, size_c, CV_32S);
    for (auto matchpair : matches)
    {
        //take one matched keypoint at a time find the corresponsing point in current and prev frame
        //once done check to which bounding box in prev and curr frame the point belong too
        //once found store the value and increment the count
        //cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);
        cv::KeyPoint prevkp1 = prevFrame.keypoints.at(matchpair.queryIdx);
        auto prevkp = prevkp1.pt;//previous frame take keypint

        cv::KeyPoint currkp1 = currFrame.keypoints.at(matchpair.trainIdx);
        auto currkp = currkp1.pt;//current frame take keypint

        for (size_t prevbb = 0; prevbb < size_p; prevbb++)//loop through all the prev frame bb
        {
            if (prevFrame.boundingBoxes[prevbb].roi.contains(prevkp))//check if the "previous frame take keypint" belongs to this box
            {//if it does
                for (size_t currbb = 0; currbb < size_c; currbb++)//loop thrpugh all the curr frame bb
                {
                    if (currFrame.boundingBoxes[currbb].roi.contains(currkp))//check if the "current frame take keypint" belongs to this box
                    {//if it does
                        //count[prevbb][currbb] = count[prevbb][currbb] + 1;//do a +1 if match is found
                        count.at<int>(prevbb, currbb) = count.at<int>(prevbb, currbb) + 1;
                    }
                }
            }
        }
    }
    //for each prev bb find and compare the max count of corresponding curr bb.
    //the curr bb with max no. of matches (max count) is the bbestmatch

        for (size_t i = 0; i < size_p; i++)//loop through prev bounding box
        {
            int id = -1;//initialize id as the matrix starts from 0 x 0 we do not want to take 0 as the initializing value
            int maxvalue = 0;//initialize max value
            for (size_t j = 0; j < size_c; j++)//loop through all curr bounding boxes to see which prev + curr bb pair has maximum count
            {
                if (count.at<int>(i,j) > maxvalue)
                {
                    maxvalue = count.at<int>(i,j);//input value for comparison
                    id = j;//id
                }

            }
            bbBestMatches[i] = id;//once found for 1 prev bounding box; input the matched pair in bbBestMatches
            //bbBestMatches.insert({i, id});
        }                     
}

            







        
 

