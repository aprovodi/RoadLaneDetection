#ifndef LINEFINDER_H
#define LINEFINDER_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class LineFinder {
    
    private:
    
    // original image
    cv::Mat img;
    
    // vector containing the end points
    // of the detected lines
    std::vector<cv::Vec4i> lines;
    cv::Vec4i left;
    cv::Vec4i right;
	// lines groups
    std::vector<int> labels;
    std::vector<cv::Vec4i> mean_lines;
    
    // accumulator resolution parameters
    double deltaRho;
    double deltaTheta;
    
    // minimum number of votes that a line
    // must receive before being considered
    int minVote;
    
    // min length for a line
    double minLength;
    
    // max allowed gap along the line
    double maxGap;
    
    // distance to shift the drawn lines down when using a ROI
    int shift;
    
    // predicate function to split lines into groups
    static bool isEqual(const cv::Vec4i& _l1, const cv::Vec4i& _l2)
    {
        cv::Vec4i l1(_l1), l2(_l2);
        
        float length1 = sqrtf((l1[2] - l1[0])*(l1[2] - l1[0]) + (l1[3] - l1[1])*(l1[3] - l1[1]));
        float length2 = sqrtf((l2[2] - l2[0])*(l2[2] - l2[0]) + (l2[3] - l2[1])*(l2[3] - l2[1]));
        
        float product = (l1[2] - l1[0])*(l2[2] - l2[0]) + (l1[3] - l1[1])*(l2[3] - l2[1]);
        
        if (fabs(product / (length1 * length2)) < cos(CV_PI / 30))
            return false;
        
        float mx1 = (l1[0] + l1[2]) * 0.5f;
        float mx2 = (l2[0] + l2[2]) * 0.5f;
        
        float my1 = (l1[1] + l1[3]) * 0.5f;
        float my2 = (l2[1] + l2[3]) * 0.5f;
        float dist = sqrtf((mx1 - mx2)*(mx1 - mx2) + (my1 - my2)*(my1 - my2));
        
        if (dist > std::max(length1, length2) * 0.2f)
            return false;
        
        return true;
    }
    
    
    public:
    
    // Default accumulator resolution is 1 pixel by 1 degree
    // no gap, no mimimum length
    LineFinder() : deltaRho(1), deltaTheta(CV_PI/180), minVote(10), minLength(0.), maxGap(0.) {
		// initial left and right lines are out of image (if we knew image size, they would be left(0,0,0,image.rows) and right(image.cols, 0, image.cols, image.rows))
        left = cv::Vec4i(0,0,-10000,10000);
        right = cv::Vec4i(10000, 0, 10000, 10000);
    }
    
    // Set the resolution of the accumulator
    void setAccResolution(double dRho, double dTheta) {
        
        deltaRho= dRho;
        deltaTheta= dTheta;
    }
    
    // Set the minimum number of votes
    void setMinVote(int minv) {
        
        minVote= minv;
    }
    
    // Set line length and gap
    void setLineLengthAndGap(double length, double gap) {
        
        minLength= length;
        maxGap= gap;
    }
    
    // set image shift
    void setShift(int imgShift) {
        
        shift = imgShift;
    }
    
    // apply probabilistic Hough Transform
    std::vector<cv::Vec4i> findLines(cv::Mat& binary) {
        
        lines.clear();
        cv::HoughLinesP(binary,lines,deltaRho,deltaTheta,minVote, minLength, maxGap);
        
        return lines;
    }
    
	// combine lines into groups, create left and right line
    std::vector<cv::Vec4i> mergeLines(cv::Mat& binary) {
        
        int numparts = cv::partition(lines, labels, isEqual);
        mean_lines.clear();
        
        for (int p = 0; p < numparts; p++) {
            std::vector<cv::Point2i> upperPoints;
            std::vector<cv::Point2i> bottomPoints;
            for (int i = 0; i < lines.size(); i++) {
                // combine points for each class (partition)
                if ( labels[i] == p )
                {
                    upperPoints.push_back(cv::Point2i(lines[i][2], lines[i][3]));
                    bottomPoints.push_back(cv::Point2i(lines[i][0], lines[i][1]));
                }
            }
            // find mean point for each class (partition)
            cv::Point2i zero(0, 0);
            cv::Point2i usum = std::accumulate(upperPoints.begin(), upperPoints.end(), zero);
            cv::Point2i bsum = std::accumulate(bottomPoints.begin(), bottomPoints.end(), zero);
            cv::Point2i umean_point(usum * (1.0f / upperPoints.size()));
            cv::Point2i bmean_point(bsum * (1.0f / bottomPoints.size()));
            mean_lines.push_back(cv::Vec4i(umean_point.x,umean_point.y, bmean_point.x, bmean_point.y));
            
            // find interception for each merged line
            cv::Point2i intercept;
            if (intersection(bmean_point, umean_point , cv::Point(0, binary.rows), cv::Point(binary.cols, binary.rows), intercept))
            {
                // TODO: currently left and right lines are defined as the first x-intersected line, with respect to botom center point in both directions
                if (intercept.x > binary.cols/2 && intercept.x < right[2])
                { right[3] = intercept.y; right[2] = intercept.x; right[0] = bmean_point.x; right[1] = bmean_point.y; }
                else if (intercept.x < binary.cols/2 && intercept.x > left[2])
                { left[3] = intercept.y; left[2] = intercept.x; left[0] = umean_point.x; left[1] = umean_point.y; }
            }
        }
        return mean_lines;
    }

    // find intersection of two lines
    bool intersection(cv::Point2i o1, cv::Point2i p1, cv::Point2i o2, cv::Point2i p2, cv::Point2i &r)
    {
        cv::Point2i x = o2 - o1;
        cv::Point2i d1 = p1 - o1;
        cv::Point2i d2 = p2 - o2;
        
        float cross = d1.x*d2.y - d1.y*d2.x;
        if (std::abs(cross) < /*EPS*/1e-8)
        return false;
        
        double t1 = (x.x * d2.y - x.y * d2.x)/cross;
        r = o1 + d1 * t1;
        return true;
    }
    
    // draw all detected lines on an image
    void drawDetectedLines(cv::Mat &image, cv::Scalar color=cv::Scalar(255)) {
        
        // Draw the lines
        std::vector<cv::Vec4i>::const_iterator it2= lines.begin(); // or it could be mean_lines
        
        while (it2!=lines.end()) {
            double angle = atan2((*it2)[1] - (*it2)[3], (*it2)[2] - (*it2)[0]) * 180.0 / CV_PI;
            // filter by angle            
            if ((angle > 10. && angle < 90.) || (angle < -10. && angle > -90.) || (angle < 150. && angle > 90.))
            {
                cv::Point pt1((*it2)[0],(*it2)[1]+shift);
                cv::Point pt2((*it2)[2],(*it2)[3]+shift);
                
                cv::line( image, pt1, pt2, color, 6 );
            }
            ++it2;
        }
    }
    
    // draw the merged lines on an image
    void drawLeftAndRightLines(cv::Mat &image, cv::Scalar color=cv::Scalar(0, 255, 255)) {
        
        cv::Point pt1(left[0],left[1]+shift);
        cv::Point pt2(left[2],left[3]+shift);
        cv::line( image, pt1, pt2, color, 6 );
        pt1 = cv::Point(right[0],right[1]+shift);
        pt2 = cv::Point(right[2],right[3]+shift);
        cv::line( image, pt1, pt2, color, 6 );
    }

    // draw all the lines by classes
    void drawMergedLines(cv::Mat &image, cv::Scalar color=cv::Scalar(0, 255, 255)) {
        
        std::vector<cv::Vec4i>::const_iterator it2= lines.begin();
        int count = 0;
        while (it2!=lines.end()) {
            if ( labels[count] == 0 )
            color = cv::Scalar(255,0,0);
            else if ( labels[count] == 1 )
            color = cv::Scalar(0,255,0);
            else if ( labels[count] == 2 )
            color = cv::Scalar(0,0,255);
            else
            color = cv::Scalar(0,0,0);
            
            cv::Point pt1((*it2)[0],(*it2)[1]+shift);
            cv::Point pt2((*it2)[2],(*it2)[3]+shift);
            cv::line( image, pt1, pt2, color, 6 );
            ++it2;
            count++;
        }
    }

    cv::Vec2i getLeaftAndRightInterceptX()
    {
        return cv::Vec2i(left[2], right[2]);
    }
};


#endif
