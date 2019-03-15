#include <iostream>
#include <fstream>
#include <ctime>
#include "ferns.h"
using namespace std;


ofstream logfile("log", ios::out);  // 以写日志的形式输出结果
int getint(char *str, int len = 4)
{
	// 从char类型转换成double类型
    unsigned char *ustr = (unsigned char *)str;
    int res = 0;
    for(int i = 0; i < len; i++)
        res = res*256 + ustr[i];
    return res;
}

const char *trainfile = "train-images.idx3-ubyte";  //训练数据
const char *labelfile = "train-labels.idx1-ubyte";  //训练label
const char *testfile = "t10k-images.idx3-ubyte"; //测试数据
const char *testlabel = "t10k-labels.idx1-ubyte"; //测试label
struct TData
{
    double *data; // 数据
    int *ys; // 标签
    int N; // 数据量
    int F; // 单个图片的大小
};
int min(int a, int b)
{
	if (a > b)
	{
		return a;
	}
	else
	{
		return b;
	}
}
TData getData(const char *file, const char *labelfile, int maxnum = 100000)
{
	TData res = {0, 0, 0, 0}; //读取文件中的前4个数据
    ifstream tf(file, ios::in | ios::binary);  //以二进制形式读数据
    ifstream lf(labelfile, ios::in | ios::binary);  //以二进制形式读取label数据
    char magic[4] = {0}; 
    tf.read(magic, 4);  //读取数据4个字节
    if(magic[2] != 8 || magic[3] != 3)
    {
		//判断当前文件的有效性
        logfile << "not a valid data file!" << endl;
		cout << "not a valid data file!" << endl;
        return res;
    }
    lf.read(magic, 4); // 读取label4个字节
    if(magic[2] != 8 || magic[3] != 1)
    {
		//判断当前文件的有效性
        logfile << "not a valid label file!" << endl;
		cout << "not a valid data file!" << endl;
        return res;
    }
    int tdnum, nrows, ncols, labelnum;
    tf.read(magic, 4);  // 读取4个数据字节
    tdnum = getint(magic); // 转换为int型，训练数据集元素个数
    lf.read(magic, 4); // 读取4个label字节
    labelnum = getint(magic); // 转换为int型，标签数据元素个数
    if(tdnum != labelnum) 
    {
		// 判断数据量和label数量是否相等
        logfile << "not right label file for this data set!" << endl;
		cout << "not right label file for this data set!" << endl;
        return res;
    }
    tf.read(magic, 4); // 读取数据4个字节 
    nrows = getint(magic); // 获取单幅图片的行数
    tf.read(magic, 4); // 读取数据4个字节
    ncols = getint(magic); // 获取单幅图片的列数
    int tnum = min(maxnum, tdnum);  //至多采用100000个数据单元进行训练
    double *data = new double[tnum*nrows*ncols]; // 开辟一个数据空间，存放所有的图片数据
    int *ys = new int[tnum]; //开辟一个数据空间，存储label数据
    int F = nrows * ncols; // 单幅图片的大小
    unsigned char *imgarr = new unsigned char[nrows*ncols]; // 开辟一个图片的空间
    unsigned char label; // 标签
    logfile << "read data, total " <<tnum<<" entries..." << endl; //读取数据的数量
	cout << "read data, total " << tnum << " entries..." << endl;
    for(int i = 0; i < tnum; i++)
    {
        double *entry = data + i*F; // 每个图片的指针位置变化
        tf.read((char *)imgarr, F); //读取图片
        for (int j = 0; j < F; ++j)
            entry[j] = (double)imgarr[j] / 255.0; // 数据类型转换，char到double型
        lf.read((char *)(&label), 1); // 读取一个label
        ys[i] = label;
    }
    res.data = data;  //图片数据集double型
    res.ys = ys;  //标签数据集char型
    res.N = tnum; // 数据量
    res.F = F; //单幅图片大小
    delete []imgarr;
    return res;
}
int main()
{
    srand(time(0)); // srand函数是随机数发生器的初始化函数
    logfile<<"load train data..."<<endl;
	cout << "load train data..." << endl;
    TData traindata = getData(trainfile, labelfile); //加载训练需要的数据
    if(traindata.N == 0)
        return 1;
    RandomFerns rf(200, 12); // 建立随机蕨（蕨的数量为200，蕨的深度为12）
    Diff_Binary_feature dbf(2400, traindata.F, 0, 1); // 随机得到二进制点对数据的index位置和相应的阈值大小
    logfile << "read train data over, "<<traindata.N<<" entries" << endl; // 输出数据量大小
	cout << "read train data over, " << traindata.N << " entries" << endl;
    logfile <<"begin train..."<<endl;
	cout << "begin train..." << endl;
    logfile << "train over, correct rate is "<<rf.train(traindata.data, traindata.ys, traindata.N, traindata.F, 10, &dbf)<<endl; // 训练
	cout<< "train over, correct rate is " << rf.train(traindata.data, traindata.ys, traindata.N, traindata.F, 10, &dbf) << endl;
    delete []traindata.data; // 释放空间
    delete []traindata.ys; // 释放空间
    logfile<<"load test data..."<<endl;
	cout << "load test data..." << endl;
    TData testdata = getData(testfile, testlabel); // 获取测试数据集
    logfile << "read test data over, "<<testdata.N<<" entries" << endl; // 测试数据集的数量
	cout << "read test data over, " << testdata.N << " entries" << endl;
    int *ysout = new int[testdata.N]; // 开辟空间，用于输出预测结果
    logfile <<"begin test..."<<endl;
    logfile <<"test over, correct rate is "<<rf.evaluate(testdata.data, testdata.ys, testdata.N, testdata.F, ysout)<<endl; // 对测试数据集进行预测
	cout << "begin test..." << endl;
	cout << "test over, correct rate is " << rf.evaluate(testdata.data, testdata.ys, testdata.N, testdata.F, ysout) << endl;
	// 对预测结果进行统计
    logfile << "compare ...\nlabel\tout" << endl;
	cout << "compare ...\nlabel\tout" << endl;
    int err = 0;
    for(int i = 0; i < testdata.N; i++)
    {
        logfile<<testdata.ys[i]<<'\t'<<ysout[i];
        if(testdata.ys[i] != ysout[i])
        {
            logfile<<"\te";
            err++;
        }
        logfile<<endl;
    }
    logfile<<"err is "<<err<<endl;
	cout << "err is " << err << endl;
    delete []testdata.data;
    delete []testdata.ys;
    delete []ysout;
    return 0;
}
