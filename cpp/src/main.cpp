#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "Graspability.h"
using namespace cv;

int main(int argc, char** argv )
{	

	int finger_h = 40;
	int finger_w = 16;
	int open_w = 80;

	double rotation_step = 45;
	int depth_step = 20;
	int down_depth = 25;

	int n_grasp = 5;

	Graspability *g = new Graspability(finger_h, finger_w, open_w, rotation_step, depth_step, down_depth);
	// g->ShowHandModel();

	std::string data_dir = "../../data/";
	std::string img_path = data_dir + "depth.png";
	std::string ret_path = data_dir + "result.png";
	Mat img = imread(img_path, 0);
	Mat ret = imread(img_path);
	
	g->GraspPlanning(img, 5);
		
	printf("Success! %lu grasps detected!\n", g->graspPoses.size());
	
	g->DrawGrasp(ret, g->graspPoses, Scalar(0,255,0));
	imwrite(ret_path, ret);
	g->Show(ret);
	for (int i=0; i<n_grasp; i++)
		printf("grasp #%d: (%d, %d, %.1f)\n",i, g->graspPoses[i].x, g->graspPoses[i].y, g->graspPoses[i].theta);

    return 0;
}

