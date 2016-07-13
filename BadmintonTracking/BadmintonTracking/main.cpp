// main.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
using namespace std;
using namespace cv;

//General Configuration
int hMin = 0;
int hMax = 180;
int sMin = 0;
int sMax = 80;
int vMin = 80;
int vMax = 255;


//滑动条回调
void on_trackbar(int, void*) {
	//HSV二值化
	//inRange(frame1HSV, Scalar(hMin, sMin, vMin), Scalar(hMax, sMax, vMax), thresholded);
	//imshow("frame", thresholded);

	//灰度二值化
	//threshold(fgMaskMOG2, fgMaskMOG2, thresholdMin, thresholdMax, THRESH_BINARY);
	//imshow("frame", fgMaskMOG2);
}

double GetDistance(Point a, Point b) {
	double xx = (a.x - b.x)*(a.x - b.x);
	double yy = (a.y - b.y)*(a.y - b.y);
	return sqrt(xx + yy);
}

int main() {
	//载入视频
	VideoCapture cap("C:/新建文件夹/0960.mp4");
	if (!cap.isOpened())
		return -1;

	int frameCount = 0;//帧计数
	int ballCount = 0;
	Point curPos, lastPos;

	Mat frame1, frame2, diff, overlay, fgMaskMOG2;


	namedWindow("trackbar", WINDOW_NORMAL);
	namedWindow("frame");

	createTrackbar("hMin", "trackbar", &hMin, 180, on_trackbar);
	createTrackbar("hMax", "trackbar", &hMax, 180, on_trackbar);
	createTrackbar("sMin", "trackbar", &sMin, 255, on_trackbar);
	createTrackbar("sMax", "trackbar", &sMax, 255, on_trackbar);
	createTrackbar("vMin", "trackbar", &vMin, 255, on_trackbar);
	createTrackbar("vMax", "trackbar", &vMax, 255, on_trackbar);

	//构造自适应混合高斯背景提取器
	Ptr<BackgroundSubtractor> pMOG2;
	pMOG2 = createBackgroundSubtractorMOG2();
	//轨迹白板图像
	//Mat track(Size(1280, 720), CV_8UC3, cv::Scalar(0, 0, 0));
	Mat track(Size(960, 540), CV_8UC3, cv::Scalar(0, 0, 0));

	double aveTime = 0;
	bool ballExist = false;
	int pixelAveValThre = 500;
	//帧循环开始
	while (cap.get(CAP_PROP_POS_FRAMES) < cap.get(CAP_PROP_FRAME_COUNT) - 1) {
		double t = (double)getTickCount();
		//读取视频帧
		cap >> frame1;
		cap >> frame2;
		++frameCount;

		//更新背景并输出前景图像
		pMOG2->apply(frame1, fgMaskMOG2);
		medianBlur(fgMaskMOG2, fgMaskMOG2, 5);
		threshold(fgMaskMOG2, fgMaskMOG2, 180, 255, THRESH_BINARY);
		imshow("mog", fgMaskMOG2);

		//进行差分
		diff = frame1 - frame2;
		blur(diff, diff, Size(7, 7));
		threshold(diff, diff, 30, 255, THRESH_BINARY);
		morphologyEx(diff, diff, MORPH_CLOSE, Mat());
		imshow("diff", diff);

		//创建轮廓数组
		vector<vector<cv::Point>> contours;
		//获取轮廓
		findContours(fgMaskMOG2,
			contours,
			CV_RETR_EXTERNAL, //只取外部轮廓
			CV_CHAIN_APPROX_NONE);//存储所有轮廓点

		Mat contoursResult(Size(960, 540), CV_8UC3, cv::Scalar(255, 255, 10));
		drawContours(contoursResult, contours, -1, Scalar(0, 0, 0), 2);
		imshow("contoursResult", contoursResult);

		int maxPixelAveVal = 0;
		double minDistance = 1280;
		RotatedRect rrs[10];
		Point2f center;
		float radius;
		int contoursNum = 0;
		//创建轮廓的迭代器
		vector<vector<cv::Point>>::const_iterator itc = contours.begin();
		while (itc != contours.end()) {
			//按size()过滤
			if (itc->size() < 6 || itc->size() > 100)
				itc = contours.erase(itc);
			else {
				RotatedRect rr = minAreaRect(Mat(*itc));//计算最小包围可旋转矩形
														//求最短边长（最小包围可旋转矩形）
				int rrMinSize = rr.size.height;
				if (rr.size.width < rrMinSize)
					rrMinSize = rr.size.width;
				//按最小包围矩形的宽度过滤 太大的去掉
				if (rrMinSize > 15)
					itc = contours.erase(itc);
				else {
					double pointDistance = 0;
					if (ballExist)
						pointDistance = GetDistance(rr.center, lastPos);
					if (ballExist && pointDistance > 210)
						itc = contours.erase(itc);
					else {
						//计算该轮廓在diff中包含区域内的像素均值来过滤轮廓
						int pointNumber = 0;	//轮廓中的点数
						int pixelValueSum = 0;	//轮廓范围中像素值之和
						int pixelAveVal = 0;	//平均的像素值
						Rect r0 = boundingRect(Mat(*itc));//计算最小包围矩形
														  //以下四行为包围矩形的位置的尺寸
						int xROI = r0.x;
						int yROI = r0.y;
						int widthROI = r0.width;
						int heightROI = r0.height;

						for (int i = xROI - 1; i < xROI + widthROI; i++)
							for (int j = yROI - 1; j < yROI + heightROI; j++) {
								//如果该点在轮廓内
								if (pointPolygonTest(*itc, Point2f(i + 1, j + 1), false) >0) {
									++pointNumber;
									pixelValueSum += diff.at<Vec3b>(j, i)[0];
									pixelValueSum += diff.at<Vec3b>(j, i)[1];
									pixelValueSum += diff.at<Vec3b>(j, i)[2];
								}
							}
						if (pointNumber != 0)
							pixelAveVal = pixelValueSum / pointNumber;
						if (pixelAveVal < 450)
							itc = contours.erase(itc);
						else {
							//下面四行 绘制该旋转矩形
							Point2f vertices[4];
							rr.points(vertices);
							for (int i = 0; i < 4; i++)
								line(frame1, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 255), 2);
							/*
							if ((ballCount<10 && pixelAveVal > maxPixelAveVal) {
							maxPixelAveVal = pixelAveVal;
							curPos = rr.center;
							*/
							if (ballCount<10)
								curPos = rr.center;
							if (ballCount >= 10 && pointDistance < minDistance) {
								minDistance = pointDistance;
								curPos = rr.center;
							}
							cout << pixelAveVal << endl;
							curPos = rr.center;
							ballExist = true;
							++itc;
						}
					}
				}
			}
		}
		if (ballExist) {
			++ballCount;
			circle(track, curPos, 2, Scalar(255, 252, 60), 2);
			if (ballCount >= 2) {
				line(track, curPos, lastPos, Scalar(40, 255, 255), 2);
			}
			lastPos = curPos;
		}
		overlay = frame1 + track;
		cv::imshow("Frame", overlay);
		double time = ((double)getTickCount() - t) / getTickFrequency();
		if (frameCount>10)
			aveTime = (aveTime*(frameCount - 11) + time) / frameCount - 10;
		cout << "------------------------当前帧序：" << frameCount << "耗时" << aveTime << endl;
		//按空格暂停 再按继续
		if (waitKey(30) == 32) {
			if (waitKey() == 32) {
				continue;
			}
		}
	}
	return 0;
}
