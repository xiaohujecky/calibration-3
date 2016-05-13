#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp> 
#include "cvut.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace cvut;
using namespace std;

//#pragma comment(lib,"cxcore.lib")
//#pragma comment(lib,"cv.lib")
//#pragma comment(lib,"highgui.lib")

CvMat *intrinsics;//内参矩阵
CvMat *distortion_coeff;
CvMat *rotation_vectors;
CvMat *translation_vectors;
CvMat *object_points;
CvMat *point_counts;
CvMat *image_points;

 CvSize board_size = cvSize(7,7);    /* 定标板上每行、列的角点数 标定板尺寸*/
 CvPoint2D32f* image_points_buf = new CvPoint2D32f[board_size.width*board_size.height];   /* 缓存每幅图像上检测到的角点 */
 Seq<CvPoint2D32f> image_points_seq;  /* 保存检测到的所有角点 */

IplImage     *img_gray;
CvSize image_size;
int image_count=0;
ifstream fin("image_pic.txt"); /* 定标所用图像文件的路径 */
ofstream fout("cab_result.txt");  /* 保存定标结果的文件 */

void main(int argc,char** argv)
{
	string filename;
	while (getline(fin,filename))
	{
		image_count++;
		Image<uchar> view(filename);
		if (image_count == 1) 
		{
			image_size.width = view.size().width;
			image_size.height = view.size().height;
		}

		int count;
 		 
	 	img_gray = cvCreateImage(cvSize(image_size.width, image_size.height), IPL_DEPTH_8U, 1);
 		cvCvtColor(view.cvimage, img_gray, CV_BGR2GRAY);

		cvFindChessboardCorners( view.cvimage, board_size,
            image_points_buf, &count, CV_CALIB_CB_ADAPTIVE_THRESH ),/*)*/
		cvFindCornerSubPix( img_gray, image_points_buf, count, cvSize(11,11),
					cvSize(-1,-1), cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 20, 0.1 ));
				image_points_seq.push_back(image_points_buf,count);
		cvDrawChessboardCorners( view.cvimage, board_size, image_points_buf, count, 1);
		view.show("标定");
		
// 		cvWaitKey();
// 		view.close();
	}	
	cout<<"照片数量为="<<image_count<<endl;
	delete []image_points_buf;
	cout<<"角点提取完成！"<<endl;
	



	cout<<"开始定标..."<<endl;
 	CvSize square_size = cvSize(7,7);  /* 实际测量得到的定标板上每个棋盘格的大小 */
	Matrix<double> object_points(1,board_size.width*board_size.height*image_count,3); /* 保存定标板上角点的三维坐标 */
 	Matrix<double> image_points(1,image_points_seq.cvseq->total,2); /* 保存提取的所有角点 */
 	Matrix<int> point_counts(1,image_count,1); /* 每幅图像中角点的数量 */
 	Matrix<double> intrinsic_matrix(3,3,1); /* 摄像机内参数矩阵 */
 	Matrix<double> distortion_coeffs(1,4,1); /* 摄像机的4个畸变系数：k1,k2,p1,p2 */
 	Matrix<double> rotation_vectors(1,image_count,3); /* 每幅图像的旋转向量 */
 	Matrix<double> translation_vectors(1,image_count,3); /* 每幅图像的平移向量 */	
 
 	/* 初始化定标板上角点的三维坐标 */
 	int i,j,t;
 	for (t=0;t<image_count;t++) 
	{
 		for (i=0;i<board_size.height;i++) 
		{
 			for (j=0;j<board_size.width;j++) 
			{
 				/* 假设定标板放在世界坐标系中z=0的平面上 */
 				object_points(0,t*board_size.height*board_size.width+i*board_size.width+j,0) = i;
 				object_points(0,t*board_size.height*board_size.width+i*board_size.width+j,1) = j;
 				object_points(0,t*board_size.height*board_size.width+i*board_size.width+j,2) = 0;
 			}
 		}
 	}
 	
 	/* 将角点的存储结构转换成矩阵形式 */
 	for (i=0;i<image_points_seq.cvseq->total;i++) {
 		image_points(0,i,0) = image_points_seq[i].x;
 		image_points(0,i,1) = image_points_seq[i].y;
 	}
	
	 	/* 初始化每幅图像中的角点数量，这里我们假设每幅图像中都可以看到完整的定标板 */
 	for (i=0;i<image_count;i++)
 		point_counts(0,i) = board_size.width*board_size.height;
 	
 	/* 开始定标 */
 	cvCalibrateCamera2(object_points.cvmat,
 					   image_points.cvmat,
                       point_counts.cvmat,
					   image_size,
                       intrinsic_matrix.cvmat,
 					   distortion_coeffs.cvmat,
                       rotation_vectors.cvmat,
 					   translation_vectors.cvmat,
 					   0);
 	cout<<"定标完成！\n";


	Matrix<double> rotation_vector(3,1); /* 保存每幅图像的旋转向量 */
	Matrix<double> rotation_matrix(3,3); /* 保存每幅图像的旋转矩阵 */
	
	fout<<"相机内参数矩阵："<<endl;
	fout<<intrinsic_matrix<<endl;
	cout<<intrinsic_matrix<<endl;
	fout<<"畸变系数："<<endl;
	fout<<distortion_coeffs<<endl;
	cout<<distortion_coeffs<<endl;
	for (i=0;i<image_count;i++) {
		fout<<"第"<<i+1<<"幅图像的旋转向量："<<endl;
		fout<<rotation_vectors(0,i,0)<<","<<rotation_vectors(0,i,1)<<","<<rotation_vectors(0,i,2);
		/* 对旋转向量进行存储格式转换 */
		for (j=0;j<3;j++) {
			rotation_vector(j,0,0) = rotation_vectors(0,i,j);
		}
		/* 将旋转向量转换为相对应的旋转矩阵 */
		cvRodrigues2(rotation_vector.cvmat,rotation_matrix.cvmat);
		fout<<"第"<<i+1<<"幅图像的旋转矩阵："<<endl;
		fout<<rotation_matrix<<endl;
		cout<<rotation_matrix<<endl;
		fout<<"第"<<i+1<<"幅图像的平移向量："<<endl;
		fout<<translation_vectors(0,i,0)<<","<<translation_vectors(0,i,1)<<","<<translation_vectors(0,i,2)<<endl;
	}
	cout<<"保存完毕"<<endl;
}

