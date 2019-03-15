#include <iostream>
#include <fstream>
#include <ctime>
#include "ferns.h"
using namespace std;


ofstream logfile("log", ios::out);  // ��д��־����ʽ������
int getint(char *str, int len = 4)
{
	// ��char����ת����double����
    unsigned char *ustr = (unsigned char *)str;
    int res = 0;
    for(int i = 0; i < len; i++)
        res = res*256 + ustr[i];
    return res;
}

const char *trainfile = "train-images.idx3-ubyte";  //ѵ������
const char *labelfile = "train-labels.idx1-ubyte";  //ѵ��label
const char *testfile = "t10k-images.idx3-ubyte"; //��������
const char *testlabel = "t10k-labels.idx1-ubyte"; //����label
struct TData
{
    double *data; // ����
    int *ys; // ��ǩ
    int N; // ������
    int F; // ����ͼƬ�Ĵ�С
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
	TData res = {0, 0, 0, 0}; //��ȡ�ļ��е�ǰ4������
    ifstream tf(file, ios::in | ios::binary);  //�Զ�������ʽ������
    ifstream lf(labelfile, ios::in | ios::binary);  //�Զ�������ʽ��ȡlabel����
    char magic[4] = {0}; 
    tf.read(magic, 4);  //��ȡ����4���ֽ�
    if(magic[2] != 8 || magic[3] != 3)
    {
		//�жϵ�ǰ�ļ�����Ч��
        logfile << "not a valid data file!" << endl;
		cout << "not a valid data file!" << endl;
        return res;
    }
    lf.read(magic, 4); // ��ȡlabel4���ֽ�
    if(magic[2] != 8 || magic[3] != 1)
    {
		//�жϵ�ǰ�ļ�����Ч��
        logfile << "not a valid label file!" << endl;
		cout << "not a valid data file!" << endl;
        return res;
    }
    int tdnum, nrows, ncols, labelnum;
    tf.read(magic, 4);  // ��ȡ4�������ֽ�
    tdnum = getint(magic); // ת��Ϊint�ͣ�ѵ�����ݼ�Ԫ�ظ���
    lf.read(magic, 4); // ��ȡ4��label�ֽ�
    labelnum = getint(magic); // ת��Ϊint�ͣ���ǩ����Ԫ�ظ���
    if(tdnum != labelnum) 
    {
		// �ж���������label�����Ƿ����
        logfile << "not right label file for this data set!" << endl;
		cout << "not right label file for this data set!" << endl;
        return res;
    }
    tf.read(magic, 4); // ��ȡ����4���ֽ� 
    nrows = getint(magic); // ��ȡ����ͼƬ������
    tf.read(magic, 4); // ��ȡ����4���ֽ�
    ncols = getint(magic); // ��ȡ����ͼƬ������
    int tnum = min(maxnum, tdnum);  //�������100000�����ݵ�Ԫ����ѵ��
    double *data = new double[tnum*nrows*ncols]; // ����һ�����ݿռ䣬������е�ͼƬ����
    int *ys = new int[tnum]; //����һ�����ݿռ䣬�洢label����
    int F = nrows * ncols; // ����ͼƬ�Ĵ�С
    unsigned char *imgarr = new unsigned char[nrows*ncols]; // ����һ��ͼƬ�Ŀռ�
    unsigned char label; // ��ǩ
    logfile << "read data, total " <<tnum<<" entries..." << endl; //��ȡ���ݵ�����
	cout << "read data, total " << tnum << " entries..." << endl;
    for(int i = 0; i < tnum; i++)
    {
        double *entry = data + i*F; // ÿ��ͼƬ��ָ��λ�ñ仯
        tf.read((char *)imgarr, F); //��ȡͼƬ
        for (int j = 0; j < F; ++j)
            entry[j] = (double)imgarr[j] / 255.0; // ��������ת����char��double��
        lf.read((char *)(&label), 1); // ��ȡһ��label
        ys[i] = label;
    }
    res.data = data;  //ͼƬ���ݼ�double��
    res.ys = ys;  //��ǩ���ݼ�char��
    res.N = tnum; // ������
    res.F = F; //����ͼƬ��С
    delete []imgarr;
    return res;
}
int main()
{
    srand(time(0)); // srand������������������ĳ�ʼ������
    logfile<<"load train data..."<<endl;
	cout << "load train data..." << endl;
    TData traindata = getData(trainfile, labelfile); //����ѵ����Ҫ������
    if(traindata.N == 0)
        return 1;
    RandomFerns rf(200, 12); // �������ާ��ާ������Ϊ200��ާ�����Ϊ12��
    Diff_Binary_feature dbf(2400, traindata.F, 0, 1); // ����õ������Ƶ�����ݵ�indexλ�ú���Ӧ����ֵ��С
    logfile << "read train data over, "<<traindata.N<<" entries" << endl; // �����������С
	cout << "read train data over, " << traindata.N << " entries" << endl;
    logfile <<"begin train..."<<endl;
	cout << "begin train..." << endl;
    logfile << "train over, correct rate is "<<rf.train(traindata.data, traindata.ys, traindata.N, traindata.F, 10, &dbf)<<endl; // ѵ��
	cout<< "train over, correct rate is " << rf.train(traindata.data, traindata.ys, traindata.N, traindata.F, 10, &dbf) << endl;
    delete []traindata.data; // �ͷſռ�
    delete []traindata.ys; // �ͷſռ�
    logfile<<"load test data..."<<endl;
	cout << "load test data..." << endl;
    TData testdata = getData(testfile, testlabel); // ��ȡ�������ݼ�
    logfile << "read test data over, "<<testdata.N<<" entries" << endl; // �������ݼ�������
	cout << "read test data over, " << testdata.N << " entries" << endl;
    int *ysout = new int[testdata.N]; // ���ٿռ䣬�������Ԥ����
    logfile <<"begin test..."<<endl;
    logfile <<"test over, correct rate is "<<rf.evaluate(testdata.data, testdata.ys, testdata.N, testdata.F, ysout)<<endl; // �Բ������ݼ�����Ԥ��
	cout << "begin test..." << endl;
	cout << "test over, correct rate is " << rf.evaluate(testdata.data, testdata.ys, testdata.N, testdata.F, ysout) << endl;
	// ��Ԥ��������ͳ��
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
