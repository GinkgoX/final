//SVM多分类训练测试  
#include<opencv2/contrib/contrib.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/ml/ml.hpp>
#include<opencv2/objdetect/objdetect.hpp>
#include <iostream>  
#include <fstream>  

using namespace cv;
using namespace std;
Directory dir;

string img_tr = "FPGA\\Trina\\";
string img_tt = "FPGA\\Test\\";
Size imageSize = Size(64, 36);

void coumputeHog(const Mat& src, vector<float> &descriptors)
{
	HOGDescriptor myHog = HOGDescriptor(imageSize, Size(32, 18), cvSize(16, 9), cvSize(4, 3), 9);
	myHog.compute(src.clone(), descriptors, Size(1, 1), Size(0, 0));

}
string num2str(double i)
{
	stringstream ss;
	ss << i;
	return ss.str();
}
int main() {
	string imageName;
	//signed imageLabel;
	vector<Mat> vecImages;
	vector<int> vecLabels;
	Mat Tr_data;
	Mat Tr_lab;

	CvSVM *mySVM = new CvSVM();
	CvSVMParams params = CvSVMParams();
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 10000, 1e-10);

	/////////////////////////读取视频////////////////////

	vector<string>trclassename = dir.GetListFolders(img_tr, "*.", true);
	for (int i = 0;i < trclassename.size();i++) {
		vector<string>imgfile = dir.GetListFiles(trclassename[i], "*.png", true);
		for (int j = 0;j < imgfile.size();j++) {
			Mat img = cv::imread(imgfile[j], 0);
			resize(img, img, imageSize);
			vector<float> vecDescriptors;
			coumputeHog(img, vecDescriptors);
			Mat tempRow = ((Mat)vecDescriptors).t();
			Tr_data.push_back(tempRow);
			Tr_lab.push_back(i + 1);
		}
	}

	/////////保存向量数据///////////////////////

	mySVM->train(Tr_data, Tr_lab, Mat(), Mat(), params);
	string svmName = "mysvm.xml";
	mySVM->save(svmName.c_str());

	///////读取测试数据，显示效果//////////////

	vector<string>ttclassename = dir.GetListFiles(img_tt, "*.png", true);
	int arrlabel[12];
	int i = 0;
	arrlabel[0] = 0;
	for (int j = 0;j < ttclassename.size();j++) {
		Mat img = cv::imread(ttclassename[j]);
		Mat gray;
		cv::cvtColor(img, gray, CV_BGR2GRAY);
		resize(gray, gray, imageSize);
		vector<float> vecDescriptors;
		coumputeHog(gray, vecDescriptors);
		Mat tempRow = ((Mat)vecDescriptors).t();
		float  label = mySVM->predict(tempRow, false);
		string lab;
		arrlabel[i + 1] = label;
		switch (int(label))
		{
		case 1:
			lab = "Ni_1";
			if (arrlabel[i] != arrlabel[i + 1]) {
				printf("你\n");
				i++;
			}
			break;
		case 2:
			lab = "Hao_2";
			if (arrlabel[i] != arrlabel[i + 1]) {
				printf("好\n");
				i++;
			}
			break;
		case 3:
			lab = "F_3";
			if (arrlabel[i] != arrlabel[i + 1])
			{
				printf("F\n");
				i++;
			}
			break;
		case 4:
			lab = "P_4";
			if (arrlabel[i] != arrlabel[i + 1])
			{
				printf("P\n");
				i++;
			}
			break;
		case 5:
			lab = "G_5";
			if (arrlabel[i] != arrlabel[i + 1])
			{
				printf("G\n");
				i++;
			}
			break;
		case 6:
			lab = "A_6";
			if (arrlabel[i] != arrlabel[i + 1])
			{
				printf("A\n");
				i++;
			}
			break;
		default:
			printf("No matching!!\n");
			break;
		}
		if (arrlabel[i] != arrlabel[i + 1]) {
			imshow(lab, img);
		}
		cv::waitKey(400);
	}
	return 0;
}