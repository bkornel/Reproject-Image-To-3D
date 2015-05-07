# Reproject-Image-To-3D
Comparing a OpenCV's [`reprojectImageTo3D()`][1] to my own

	// Reproject image to 3D
	void customReproject(const cv::Mat& disparity, const cv::Mat& Q, cv::Mat& out3D)
	{
		CV_Assert(disparity.type() == CV_32F && !disparity.empty());
		CV_Assert(Q.type() == CV_32F && Q.cols == 4 && Q.rows == 4);

		// 3-channel matrix for containing the reprojected 3D world coordinates
		out3D = cv::Mat::zeros(disparity.size(), CV_32FC3);

		// Getting the interesting parameters from Q, everything else is zero or one
		float Q03 = Q.at<float>(0, 3);
		float Q13 = Q.at<float>(1, 3);
		float Q23 = Q.at<float>(2, 3);
		float Q32 = Q.at<float>(3, 2);
		float Q33 = Q.at<float>(3, 3);

		// Transforming a single-channel disparity map to a 3-channel image representing a 3D surface
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
	
  [1]: http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#reprojectimageto3d
