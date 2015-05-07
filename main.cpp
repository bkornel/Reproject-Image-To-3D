#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

void customReproject(const cv::Mat& disparity, const cv::Mat& Q, cv::Mat& out3D)
{
	CV_Assert(disparity.type() == CV_32F && !disparity.empty());
	CV_Assert(Q.type() == CV_32F && Q.cols == 4 && Q.rows == 4);

	out3D = cv::Mat::zeros(disparity.size(), CV_32FC3);

	float Q03 = Q.at<float>(0, 3);
	float Q13 = Q.at<float>(1, 3);
	float Q23 = Q.at<float>(2, 3);
	float Q32 = Q.at<float>(3, 2);
	float Q33 = Q.at<float>(3, 3);

	for (int i = 0; i < disparity.rows; i++)
	{
		const float* disp_ptr = disparity.ptr<float>(i);
		cv::Vec3f* out3D_ptr = out3D.ptr<cv::Vec3f>(i);

		for (int j = 0; j < disparity.cols; j++)
		{
			const float pw = 1.0f / (disp_ptr[j] * Q32 + Q33);

			cv::Vec3f& point = out3D_ptr[j];
			point[0] = (static_cast<float>(j)+Q03) * pw;
			point[1] = (static_cast<float>(i)+Q13) * pw;
			point[2] = Q23 * pw;
		}
	}
}

void save(const cv::Mat& image3D, const std::string& fileName)
{
	CV_Assert(image3D.type() == CV_32FC3 && !image3D.empty());
	CV_Assert(!fileName.empty());

	std::ofstream outFile(fileName);

	if (!outFile.is_open())
	{
		std::cerr << "ERROR: Could not open " << fileName << std::endl;
		return;
	}

	for (int i = 0; i < image3D.rows; i++)
	{
		const cv::Vec3f* image3D_ptr = image3D.ptr<cv::Vec3f>(i);

		for (int j = 0; j < image3D.cols; j++)
		{
			outFile << image3D_ptr[j][0] << " " << image3D_ptr[j][1] << " " << image3D_ptr[j][2] << std::endl;
		}
	}

	outFile.close();
}

int main(int argc, char* argv[])
{
	// Loading Q Matrix
	cv::FileStorage fs("Q.xml", cv::FileStorage::READ);

	if (!fs.isOpened())
	{
		std::cerr << "ERROR: Could not read Q.xml" << std::endl;
		return 1;
	}

	cv::Mat Q;

	fs["Q"] >> Q;
	Q.convertTo(Q, CV_32F);
	fs.release();

	// If size of Q is not 4x4 exit
	if (Q.cols != 4 || Q.rows != 4)
	{
		std::cerr << "ERROR: Q is not 4x4)" << std::endl;
		return 1;
	}

	// Loading disparity image
	cv::Mat disparity = cv::imread("disparity-image.pgm", cv::IMREAD_GRAYSCALE);
	if (disparity.empty())
	{
		std::cerr << "ERROR: Could not read disparity-image.pgm" << std::endl;
		return 1;
	}

	// Conversion of the disparity map to 32F before reprojecting to 3D
	// NOTE: also take care to do not scale twice the disparity
	disparity.convertTo(disparity, CV_32F, 1.0 / 16.0);

	// Reproject image to 3D by OpenCV
	cv::Mat image3DOCV;
	cv::reprojectImageTo3D(disparity, image3DOCV, Q, false, CV_32F);

	// Reproject image to 3D by our own method
	cv::Mat image3D;
	customReproject(disparity, Q, image3D);

	save(image3D, "custom.xyz");
	save(image3D, "opencv.xyz");

	return 0;
}
