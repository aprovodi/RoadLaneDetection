
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include "linefinder.h"
#include "dirent.h"

int
main (int argc, char *argv[])
{
	if (!argv[1])
	{
        printf ("Please, specify input folder, containing images, as an argument!\n");
        return 1;
	}
    std::string inputDirectory = argv[1];
    struct dirent **namelist;
    int n = scandir(inputDirectory.c_str(), &namelist, 0, alphasort);
    if (n < 0)
        perror("scandir");

    std::string window_name = "Processed Frame";
    cv::namedWindow (window_name, CV_WINDOW_KEEPRATIO); //resizable window;
    cv::Mat image;
    std::ofstream intercepts_file;
    intercepts_file.open ("intercepts.csv");
    for (int count = 0; count < n; count++)
    {
        std::string filepath =
          inputDirectory + "/" + std::string (namelist[count]->d_name);
        std::string filename = std::string (namelist[count]->d_name);

        image = cv::imread (filepath.c_str ());
        
        //capture >> image;
        if (image.empty ())
	        continue;
        cv::GaussianBlur (image, image, cv::Size (17, 17), 0, 0);
        
        cv::Mat gray;
        cv::cvtColor (image, gray, CV_RGB2GRAY);
        std::vector < std::string > codes;
        cv::Mat corners;
        findDataMatrix (gray, codes, corners);
        drawDataMatrixCodes (image, codes, corners);
        
        int interesting_height = 5 * image.cols / 12;
        cv::Rect roi (0, interesting_height, image.cols - 1, image.rows - interesting_height); // set the ROI for the image
        cv::Mat imgROI = image (roi);
        
        // Canny algorithm
        cv::Mat contours;
        cv::Canny (imgROI, contours, 50, 20);
        cv::Mat contoursInv;
        cv::threshold (contours, contoursInv, 128, 255, cv::THRESH_BINARY_INV);
        
        /*
        Hough tranform for line detection with feedback
        Increase by 25 for the next frame if we found some lines.
        This is so we don't miss other lines that may crop up in the next frame
        but at the same time we don't want to start the feed back loop from scratch.
        */
        int houghVote = 20;
        std::vector < cv::Vec2f > lines;
        if (houghVote < 1 or lines.size () > 2)
        {  // we lost all lines. reset
            houghVote = 100;
        }
        else
        {
            houghVote += 25;
        }
        while (lines.size () < 5 && houghVote > 0)
        {
            cv::HoughLines (contours, lines, 1, CV_PI / 180, houghVote);
            houghVote -= 5;
        }
        
        cv::Mat result (imgROI.size (), CV_8U, cv::Scalar (255));
        imgROI.copyTo (result);
        
        // Draw the lines
        std::vector < cv::Vec2f >::const_iterator it = lines.begin ();
        cv::Mat hough (imgROI.size (), CV_8U, cv::Scalar (0));
        
        while (it != lines.end ())
        {
            float rho = (*it)[0];
            float theta = (*it)[1];
            
            if ((theta > CV_PI / 4. && theta < 9. * CV_PI / 20.)
            || (theta < 3. * CV_PI / 4. && theta > 11. * CV_PI / 20.))
            {
                // point of intersection of the line with first row
                cv::Point pt1 (rho / cos (theta), 0);
                // point of intersection of the line with last row
                cv::Point pt2 ((rho - result.rows * sin (theta)) / cos (theta),
                result.rows);
                // draw a white line
                cv::line (result, pt1, pt2, cv::Scalar (255), 8);
                cv::line (hough, pt1, pt2, cv::Scalar (255), 8);
            }
            ++it;
        }
        
        // Create LineFinder instance
        LineFinder ld;
        
        // Set probabilistic Hough parameters
        ld.setLineLengthAndGap (250, 120);
        ld.setMinVote (90);
        
        // Detect lines
        std::vector < cv::Vec4i > li = ld.findLines (contours);
        cv::Mat houghP (imgROI.size (), CV_8U, cv::Scalar (0));
        ld.setShift (0);
        ld.drawDetectedLines (houghP);
        
        // bitwise AND of the two hough images
        cv::bitwise_and (houghP, hough, houghP);
        cv::Mat houghPinv (imgROI.size (), CV_8U, cv::Scalar (0));
        cv::Mat dst (imgROI.size (), CV_8U, cv::Scalar (0));
        cv::threshold (houghP, houghPinv, 80, 255, cv::THRESH_BINARY_INV); // threshold and invert to black lines
        
        cv::Canny (houghPinv, contours, 30, 65);
		// Find lines again, split into groups and identify left and right
        li = ld.findLines (contours);
        std::vector < cv::Vec4i > mli = ld.mergeLines (contours);
        ld.setShift (interesting_height);
        ld.drawLeftAndRightLines (image);
        
        cv::imshow (window_name, image);
        cv::imwrite("processed_" + std::string(filename), image);
        
        cv::Vec2i xintercept = ld.getLeaftAndRightInterceptX ();
        intercepts_file << filename << ",";
        (xintercept[0] ==
        -10000) ? intercepts_file << "None," : intercepts_file << xintercept[0]
        << ",";
        (xintercept[1] ==
        10000) ? intercepts_file << "None" : intercepts_file << xintercept[1];
        intercepts_file << std::endl;
        
        char key = (char) cv::waitKey (10);
        lines.clear ();
        free(namelist[count]);
    }
    intercepts_file.close ();
    free(namelist);
}
