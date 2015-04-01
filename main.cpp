#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

#include "gc_min.hpp"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

int main( int argc, char** argv ) {

    cv::VideoCapture cap(0);





    cv::Mat img, gray;
    cap >> img; cap >> img; cap >> img;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);


    int vtxCount = img.cols*img.rows;
    int edgeCount = 2*(4*img.cols*img.rows - 3*(img.cols + img.rows) + 2);



    cv::Point p,q;
    double covariance = 5.0;
    double weight_diff = 0.02;
    double max_diff = 20;

    cv::Mat denoised(gray.size(), gray.type(), cv::Scalar(0));

    for (uchar intensity = 1; intensity<255; ++intensity) {

        GC_Solver<double> graph;
        graph.create(vtxCount, edgeCount);
        std::cout<<" value # "<<(int)intensity<<std::endl;

        for( p.y = 0; p.y < img.rows; p.y++ ) {
            for( p.x = 0; p.x < img.cols; p.x++) {

                // add node
                int vtxIdx = graph.addVtx();

                double image_data_p = (double)gray.at<uchar>(p);

                double color_diff_p_0 = pow( double(intensity)-image_data_p , 2.0 )/(2*covariance);
                double color_diff_p_1 = pow( double(denoised.at<uchar>(p))-image_data_p , 2.0 )/(2*covariance);


                // set t-weights
                graph.addTermWeights( vtxIdx, color_diff_p_0, color_diff_p_1 );

                // set n-weights
                if( p.x>0 ) { // left
                    q = cv::Point(p.x-1,p.y);
                    double image_data_q = (double)gray.at<uchar>(q);

                    double w = weight_diff * MIN( pow( image_data_p - image_data_q, 2.0 ), max_diff );
                    graph.addEdges( vtxIdx, vtxIdx-1, w, w );
                }

                if( p.x>0 && p.y>0 ) { // upleft
                    q = cv::Point(p.x-1,p.y-1);
                    double image_data_q = (double)gray.at<uchar>(q);

                    double w = weight_diff * MIN( pow( image_data_p - image_data_q, 2.0 ), max_diff );
                    graph.addEdges( vtxIdx, vtxIdx-img.cols-1, w, w );
                }

                if( p.y>0 ) { // up
                    q = cv::Point(p.x,p.y-1);
                    double image_data_q = (double)gray.at<uchar>(q);

                    double w = weight_diff * MIN( pow( image_data_p - image_data_q, 2.0 ), max_diff );
                    graph.addEdges( vtxIdx, vtxIdx-img.cols, w, w );
                }
                if( p.x<img.cols-1 && p.y>0 ) { // upright
                    q = cv::Point(p.x+1,p.y-1);
                    double image_data_q = (double)gray.at<uchar>(q);

                    double w = weight_diff * MIN( pow( image_data_p - image_data_q, 2.0 ), max_diff );
                    graph.addEdges( vtxIdx, vtxIdx-img.cols+1, w, w );
                }
            }
        }
        graph.maxFlow();

        for( p.y = 0; p.y < denoised.rows; p.y++ ) {
            for( p.x = 0; p.x < denoised.cols; p.x++ ) {
                if( !graph.inSourceSegment( p.y*denoised.cols+p.x /*vertex index*/ ) )
                    denoised.at<uchar>(p) = intensity;
            }
        }

        cv::imshow("DEN", denoised);
        cv::waitKey(30);
    }


    cv::imshow("DEN", denoised);
    cv::imshow("IMG", gray);

    cv::waitKey(0);

    return 0;
}
