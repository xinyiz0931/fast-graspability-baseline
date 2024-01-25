#pragma once
#include <math.h>
#include <cmath>
#include <opencv2/opencv.hpp>

struct GraspPose{
	int x;
	int y;
	double theta;
};  

class Graspability
{
public:
	Graspability(int FingerH, int FingerW, int OpenWidth, 
						   double RotationStep, int DepthStep, int HandDownDepth);

	void Show(cv::Mat);
	void ShowHandModel();
	void GenHandTplt(cv::Mat &tplt, int, int, int, int, double);
	// void SetDismissArea(cv::Mat &);
	
	void GraspPlanning(cv::Mat, int);

	void GraspabilityMap(cv::Mat, cv::Mat, cv::Mat);
	void GraspRanking(int, int, int, int, int);
	// void TargetOrientedMap(Mat touchMat, Mat ConflictMat, Point &GraspPoint, double &GraspAngle);
	// void PointOrientedMap(Mat SrcMat, Point GraspPoint, double &GraspAngle);
	
	void DrawGrasp(cv::Mat &, std::vector<GraspPose>, cv::Scalar);

	// Grasp candidates
	std::vector<GraspPose> cGraspPoses;
	std::vector<int> cScores;

	std::vector<GraspPose> graspPoses;
	
private:

	int m_TpltSize;

	cv::Mat m_Ht;
	cv::Mat m_Hc;
	int m_Center;
	int m_HalfOpenW;
	int m_HalfFingerH;
	int m_FingerW;
    int m_OpenWidth;

	double m_RotationStep;
	int m_DepthStep;
	int m_HandDownDepth;

	int m_KernelSize;
	int m_Sigma;
};