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

CvMat *intrinsics;//�ڲξ���
CvMat *distortion_coeff;
CvMat *rotation_vectors;
CvMat *translation_vectors;
CvMat *object_points;
CvMat *point_counts;
CvMat *image_points;

 CvSize board_size = cvSize(7,7);    /* �������ÿ�С��еĽǵ��� �궨��ߴ�*/
 CvPoint2D32f* image_points_buf = new CvPoint2D32f[board_size.width*board_size.height];   /* ����ÿ��ͼ���ϼ�⵽�Ľǵ� */
 Seq<CvPoint2D32f> image_points_seq;  /* �����⵽�����нǵ� */

IplImage     *img_gray;
CvSize image_size;
int image_count=0;
ifstream fin("image_pic.txt"); /* ��������ͼ���ļ���·�� */
ofstream fout("cab_result.txt");  /* ���涨�������ļ� */

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
		view.show("�궨");
		
// 		cvWaitKey();
// 		view.close();
	}	
	cout<<"��Ƭ����Ϊ="<<image_count<<endl;
	delete []image_points_buf;
	cout<<"�ǵ���ȡ��ɣ�"<<endl;
	



	cout<<"��ʼ����..."<<endl;
 	CvSize square_size = cvSize(7,7);  /* ʵ�ʲ����õ��Ķ������ÿ�����̸�Ĵ�С */
	Matrix<double> object_points(1,board_size.width*board_size.height*image_count,3); /* ���涨����Ͻǵ����ά���� */
 	Matrix<double> image_points(1,image_points_seq.cvseq->total,2); /* ������ȡ�����нǵ� */
 	Matrix<int> point_counts(1,image_count,1); /* ÿ��ͼ���нǵ������ */
 	Matrix<double> intrinsic_matrix(3,3,1); /* ������ڲ������� */
 	Matrix<double> distortion_coeffs(1,4,1); /* �������4������ϵ����k1,k2,p1,p2 */
 	Matrix<double> rotation_vectors(1,image_count,3); /* ÿ��ͼ�����ת���� */
 	Matrix<double> translation_vectors(1,image_count,3); /* ÿ��ͼ���ƽ������ */	
 
 	/* ��ʼ��������Ͻǵ����ά���� */
 	int i,j,t;
 	for (t=0;t<image_count;t++) 
	{
 		for (i=0;i<board_size.height;i++) 
		{
 			for (j=0;j<board_size.width;j++) 
			{
 				/* ���趨��������������ϵ��z=0��ƽ���� */
 				object_points(0,t*board_size.height*board_size.width+i*board_size.width+j,0) = i;
 				object_points(0,t*board_size.height*board_size.width+i*board_size.width+j,1) = j;
 				object_points(0,t*board_size.height*board_size.width+i*board_size.width+j,2) = 0;
 			}
 		}
 	}
 	
 	/* ���ǵ�Ĵ洢�ṹת���ɾ�����ʽ */
 	for (i=0;i<image_points_seq.cvseq->total;i++) {
 		image_points(0,i,0) = image_points_seq[i].x;
 		image_points(0,i,1) = image_points_seq[i].y;
 	}
	
	 	/* ��ʼ��ÿ��ͼ���еĽǵ��������������Ǽ���ÿ��ͼ���ж����Կ��������Ķ���� */
 	for (i=0;i<image_count;i++)
 		point_counts(0,i) = board_size.width*board_size.height;
 	
 	/* ��ʼ���� */
 	cvCalibrateCamera2(object_points.cvmat,
 					   image_points.cvmat,
                       point_counts.cvmat,
					   image_size,
                       intrinsic_matrix.cvmat,
 					   distortion_coeffs.cvmat,
                       rotation_vectors.cvmat,
 					   translation_vectors.cvmat,
 					   0);
 	cout<<"������ɣ�\n";


	Matrix<double> rotation_vector(3,1); /* ����ÿ��ͼ�����ת���� */
	Matrix<double> rotation_matrix(3,3); /* ����ÿ��ͼ�����ת���� */
	
	fout<<"����ڲ�������"<<endl;
	fout<<intrinsic_matrix<<endl;
	cout<<intrinsic_matrix<<endl;
	fout<<"����ϵ����"<<endl;
	fout<<distortion_coeffs<<endl;
	cout<<distortion_coeffs<<endl;
	for (i=0;i<image_count;i++) {
		fout<<"��"<<i+1<<"��ͼ�����ת������"<<endl;
		fout<<rotation_vectors(0,i,0)<<","<<rotation_vectors(0,i,1)<<","<<rotation_vectors(0,i,2);
		/* ����ת�������д洢��ʽת�� */
		for (j=0;j<3;j++) {
			rotation_vector(j,0,0) = rotation_vectors(0,i,j);
		}
		/* ����ת����ת��Ϊ���Ӧ����ת���� */
		cvRodrigues2(rotation_vector.cvmat,rotation_matrix.cvmat);
		fout<<"��"<<i+1<<"��ͼ�����ת����"<<endl;
		fout<<rotation_matrix<<endl;
		cout<<rotation_matrix<<endl;
		fout<<"��"<<i+1<<"��ͼ���ƽ��������"<<endl;
		fout<<translation_vectors(0,i,0)<<","<<translation_vectors(0,i,1)<<","<<translation_vectors(0,i,2)<<endl;
	}
	cout<<"�������"<<endl;
}

