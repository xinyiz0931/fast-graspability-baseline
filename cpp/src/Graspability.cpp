#include "Graspability.h"
#include <string>
#include <numeric>
using namespace cv;
using namespace std;

void MatType(Mat SrcMat)
{
	int inttype = SrcMat.type();

	string r, a;
	uchar depth = inttype & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (inttype >> CV_CN_SHIFT);
	switch (depth) {
	case CV_8U:  r = "8U";   a = "Mat.at<uchar>(y,x)"; break;
	case CV_8S:  r = "8S";   a = "Mat.at<schar>(y,x)"; break;
	case CV_16U: r = "16U";  a = "Mat.at<ushort>(y,x)"; break;
	case CV_16S: r = "16S";  a = "Mat.at<short>(y,x)"; break;
	case CV_32S: r = "32S";  a = "Mat.at<int>(y,x)"; break;
	case CV_32F: r = "32F";  a = "Mat.at<float>(y,x)"; break;
	case CV_64F: r = "64F";  a = "Mat.at<double>(y,x)"; break;
	default:     r = "User"; a = "Mat.at<UKNOWN>(y,x)"; break;
	}
	r += "C";
	r += (chans + '0');
	cout << "Mat is of type " << r << " and should be accessed with " << a << endl;
}

Graspability::Graspability(int FingerH, int FingerW, int OpenWidth, 
						   double RotationStep, int DepthStep, int HandDownDepth)
{

	m_TpltSize = 500;
    m_OpenWidth = OpenWidth;
    m_FingerW = FingerW;

	// Graspability param
	m_RotationStep = RotationStep;
	m_DepthStep = DepthStep;
	m_HandDownDepth = HandDownDepth;

	m_KernelSize = 75;
	m_Sigma = 25;
	
	m_Ht = Mat::zeros(m_TpltSize, m_TpltSize, CV_8UC1);
	m_Hc = Mat::zeros(m_TpltSize, m_TpltSize, CV_8UC1);

	// Hand param
	m_Center = int(m_TpltSize/ 2); // template center
	m_HalfOpenW = int(m_OpenWidth / 2); // half of gripper open width
	m_HalfFingerH = int(FingerH / 2); // half of finger height
	m_FingerW = FingerW; // finger width

	m_Hc(Rect((m_Center - m_HalfOpenW - m_FingerW), (m_Center - m_HalfFingerH), m_FingerW, m_HalfFingerH * 2)).setTo(255);
	m_Hc(Rect((m_Center + m_HalfOpenW), (m_Center - m_HalfFingerH), m_FingerW, m_HalfFingerH * 2)).setTo(255);
	m_Ht(Rect((m_Center - m_HalfOpenW), (m_Center - m_HalfFingerH), m_HalfOpenW * 2, m_HalfFingerH * 2)).setTo(255);
	
}
void Graspability::Show(Mat src)
{
	namedWindow("Display",WINDOW_AUTOSIZE);
	imshow("Display", src);
	waitKey(0);
	destroyWindow("Display");
}

void Graspability::ShowHandModel()
{
	Mat m_HandModel;
	hconcat(this->m_Hc, this->m_Ht, m_HandModel);
	namedWindow("Opening | Closing", WINDOW_AUTOSIZE | WINDOW_FREERATIO);

	imshow("Opening | Closing", m_HandModel);
	waitKey(0);
	destroyWindow("Opening | Closing");
}

void Graspability::GenHandTplt(Mat &tplt, int h, int w, int x, int y, double theta) 
{
	cv::Mat draw = Mat::zeros(h, w, CV_8UC1);
	tplt = Mat::zeros(h, w, CV_8UC1);
	draw(Rect((x - m_HalfOpenW - m_FingerW), (y - m_HalfFingerH), m_FingerW, m_HalfFingerH * 2)).setTo(255);
	draw(Rect((x + m_HalfOpenW), (y - m_HalfFingerH), m_FingerW, m_HalfFingerH * 2)).setTo(255);
	draw(Rect((x - m_HalfOpenW), (y - 1), m_HalfFingerH*4, 2)).setTo(255);
	draw(Rect((x - 3), (y - 3), 6, 6)).setTo(255);
	Mat affine_matrix = getRotationMatrix2D(Point(x, y), theta, 1.0);
	warpAffine(draw, tplt, affine_matrix, draw.size());
}

void Graspability::GraspabilityMap(Mat src, Mat maskOpen, Mat maskClose)
{
	double MaxGraspability = 0;
	int HandDepth;
	double HandRotation;

	std::vector<Mat> Ht_rot;
	std::vector<Mat> Hc_rot;

	// Mat Wt, Wc, T, C, Cbar, T_and_Cbar, G, Thresh;
	for (HandRotation == 0.0; HandRotation < 180.0; HandRotation += m_RotationStep) {
		Mat Ht_, Hc_; 
		Point2f center(m_TpltSize/2, m_TpltSize/2);
		Mat affine_matrix = getRotationMatrix2D(center, HandRotation, 1.0);
		warpAffine(maskClose, Ht_, affine_matrix, maskClose.size());
		warpAffine(maskOpen, Hc_, affine_matrix, maskOpen.size());
		Ht_rot.push_back(Ht_);
		Hc_rot.push_back(Hc_);
	}

	// set dismiss area first
	// SetDismissArea(SrcMat);
	// this->show(SrcMat);
	for (HandDepth = 0.0; HandDepth <= 200; HandDepth += m_DepthStep)
	{
		Mat Wc, Wt;
		threshold(src, Wc, HandDepth, 255, THRESH_BINARY);
		threshold(src, Wt, HandDepth + m_HandDownDepth, 255, THRESH_BINARY);
		int j = 0;
		for (HandRotation = 0.0; HandRotation < 180.0; HandRotation += m_RotationStep)
		{
			double tmpMaxG;
			Point tmpMaxGLoc;

			Mat T, C, Cbar, T_and_Cbar, G, thresh;
			Mat Ht = Ht_rot[j];
			Mat Hc = Hc_rot[j];
			
			filter2D(Wc, C, -1, Hc);
			filter2D(Wt, T, -1, Ht);
			
			bitwise_not(C, Cbar);
			T_and_Cbar = T & Cbar;

			GaussianBlur(T_and_Cbar, G, Size(m_KernelSize, m_KernelSize), m_Sigma, m_Sigma);
			Mat TC_bool, labels, stats, centroids;
			threshold(T_and_Cbar, thresh, 122, 255, THRESH_BINARY);
			
			int n = connectedComponentsWithStats(thresh, labels, stats, centroids);
			for (int i=0; i<n; i++) {
				int x = static_cast<int>(centroids.at<double>(i,0));
				int y = static_cast<int>(centroids.at<double>(i,1));
				GraspPose gp {x, y, HandRotation};
				int score = (int)G.at<uchar>(y,x);
				if (score > 0) {
					auto pos = std::find_if(cScores.begin(), cScores.end(), [score](auto s) {
						return s < score;
					});
					int index = std::distance(std::begin(cScores), pos);
					this->cScores.insert(cScores.begin() + index, score);
					this->cGraspPoses.insert(cGraspPoses.begin() + index, gp);
				}
			}
			j++;
		}
	}
}

void Graspability::GraspRanking(int n, int h, int w, int _dismiss, int _distance)
{
	int i = 0;
	int k = 0;
	if (cScores.size() < n) 
		n = cScores.size();

	while (k < n && i < cScores.size()) {
		GraspPose cg = cGraspPoses[i];
		if (_dismiss < cg.x && cg.x < w-_dismiss && _dismiss < cg.y && cg.y < h-_dismiss) {
			if (graspPoses.size() == 0) {
				graspPoses.push_back(cg);
				k++;
			}
			else {
				bool addGrasp =true;
				for (int j=0; j<graspPoses.size(); j++) {
					GraspPose g = graspPoses[j];
					if (pow((cg.x - g.x),2) + pow((cg.y - g.y),2) < pow(_distance,2)) {
						addGrasp = false;
					}
				}
				if (addGrasp) {
					graspPoses.push_back(cg);
					k++;
					}
			}
		}
		i++;
	}
	if (graspPoses.size() == 0) 
		std::cout << "No valid grasps after ranking! " << std::endl;
}

void Graspability::GraspPlanning(Mat src, int n)
{
	int h, w; 
	h = src.rows;
	w = src.cols;
	GraspabilityMap(src, m_Hc, m_Ht);
	int _dm = (m_OpenWidth + m_FingerW*2) * atan(45);
	int _dt = 25;
	GraspRanking(n, h, w, _dm, _dt);
}

void Graspability::DrawGrasp(Mat &src, std::vector<GraspPose> gs, Scalar BGR)
{
// TODO
	int h = src.rows;
	int w = src.cols;

	for (int i=gs.size()-1; i>=0; i--) {
		GraspPose g = gs[i];
		Mat mask;
		Scalar c = int(gs.size()-i)*BGR/int(gs.size());
		// GenHandTplt(tplt, h, w, g.x, g.y, g.theta, c);
		GenHandTplt(mask, h, w, g.x, g.y, g.theta);
		Mat tplt = Mat(h, w, CV_8UC3, c);
		Mat mask_inv, fg, bg, newmat;
		bitwise_not(mask, mask_inv);  
		bitwise_and(tplt, tplt, fg, mask);
		bitwise_and(src, src, bg, mask_inv);
		add(bg, fg, src);
	}
}

