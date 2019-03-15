#ifndef FERNS_H
#define FERNS_H

#include <cstdlib>
#include <cmath>

#ifndef RAND_MAX
#define RAND_MAX 65535
#endif // RAND_MAX
// 二进制类
class Binary_feature
{
protected:
    int fea_num; // 单个特征长度
    unsigned int *fea; // 记录单个特征内容
    Binary_feature() : fea_num(0), fea(0) {}  // 构造函数
public:
    Binary_feature(int num) : fea_num(num)
    {
        const int nbits = sizeof(unsigned int)<<3; // 数据左移三位，unsigned int 占4个字节
        int len = (num + nbits-1)/nbits; // 特征长度
        fea = new unsigned int[len]; //开辟空间
        memset(fea, 0, sizeof(unsigned int)*len); // 初始化特征
    }
    Binary_feature(const Binary_feature &bf) : fea_num(bf.fea_num)
    {
        const int nbits = sizeof(unsigned int)<<3; // 数据左移三位，unsigned int 占4个字节
        int len = (fea_num + nbits-1)/nbits; // 特征长度
        fea = new unsigned int[len];  //开辟单个特征空间
        memcpy(fea, bf.fea, sizeof(unsigned int)*len); // 初始化特征空间
    }
    virtual ~Binary_feature()
    {
		// 析构函数
        if (fea)
            delete []fea; // 释放开辟的特征空间
    }
    unsigned int get_binary(int from, int to)
    {
		// 获取二进制特征，用fea数组存储
        const int nbits = sizeof(unsigned int)<<3; // 数据左移三位，unsigned int 占4个字节
		// 开始位置
        int fp = from/(nbits); // 完整的nbits个数
        int foff = from%(nbits); // 不完整的nbits的长度
		// 结束位置
        int tp = to/(nbits); // 完整的nbits个数
        int toff = to%(nbits); // 不完整的nbits的长度
        if (fp == tp)
        {
			// 需要的特征长度在nbits大小以内
            return (fea[fp]>>(nbits-1 - toff))&((1u<<(toff-foff+1))-1); // 截取一定长度的特征大小(十进制)
        }
        else
        {
            unsigned int res = fea[fp]&((1u<<(nbits-foff))-1);
            for (int i = fp+1; i < tp; ++i)
            {
                res <<= nbits;
                res += fea[i]; 
            }
            res <<= (toff+1); 
            res += (fea[tp]>>(nbits-1-toff)); 
            return res; // 返回获取一定长度的特征大小(十进制)
        }
    }
    unsigned int get_binary()
    {
        return get_binary(0, fea_num-1); // 缺省参数的情况下，获取数据长度的二进制特征
    }
    void set_bit(int pos)
    {
		// 将pos位置置为1
        const int nbits = sizeof(unsigned int)<<3;  // 32
        int p = pos / nbits;  // 完整nbits的个数
        int off = pos % nbits;  // 不完整nbits的长度
        fea[p] |= (1u<<(nbits-1-off));  // 更新指定位置的feature大小，1u,表示当前1是无符号整型
    }
    void reset_bit(int pos)
    {
        const int nbits = sizeof(unsigned int)<<3; // 32
        int p = pos / nbits;   // 完整nbits的个数
        int off = pos % nbits;  // 不完整nbits的长度
        fea[p] &= ~(1u<<(nbits-1-off)); // 将pos位置的特征值置反
    }
    virtual Binary_feature *copy_self() const = 0;
    virtual void get_feature(double *vec, int veclen) = 0;
};
class Diff_Binary_feature : public Binary_feature
{
private:
    int *id; // 二进制特征对应的id
    double *thrs; // 两个像素点差的阈值
public:
    Diff_Binary_feature(int num) : Binary_feature(num)
    {
        id = new int[fea_num<<1]; // 开辟扩大一倍的空间，id是点对
        thrs = new double[fea_num]; //阈值
    }
    Diff_Binary_feature(int num, int veclen, double (*range)[2]) : Binary_feature(num)
    {
        id = new int[fea_num<<1]; // 开辟扩大一倍的空间
        thrs = new double[fea_num]; //阈值
        set_random(veclen, range); // 对id和thrs进行赋值
    }
    Diff_Binary_feature(int num, int veclen, double inf, double sup) : Binary_feature(num)
    {
        id = new int[fea_num<<1];  // 开辟扩大一倍的空间
		thrs = new double[fea_num]; //阈值
        set_random(veclen, inf, sup); // 在指定区域内生成一个随机阈值
    }
    Diff_Binary_feature(int num, int *aid, double *athrs) : Binary_feature(num)
    {
        id = new int[fea_num<<1]; // 开辟扩大一倍的空间
        thrs = new double[fea_num]; // 阈值
        memcpy(id, aid, sizeof(int)*(fea_num<<1)); // 用已有的id赋值到新生成的id数组
        memcpy(thrs, athrs, sizeof(double)*fea_num); // 用已有的thrs赋值到新开辟的thrs数组
    }
    Diff_Binary_feature(const Diff_Binary_feature &dbf) : Binary_feature(dbf)
    {
        id = new int[fea_num<<1]; // 开辟扩大一倍的空间
        thrs = new double[fea_num]; // 阈值
        memcpy(id, dbf.id, sizeof(int)*(fea_num<<1)); // 用已有的二进制特征对当前id和thrs阈值进行初始化
        memcpy(thrs, dbf.thrs, sizeof(double)*fea_num);
    }
    virtual ~Diff_Binary_feature()
    {
		// 析构函数
        delete []id;
        delete []thrs;
        id = 0;
        thrs = 0;
    }
    void set_param(int *aid, double *athrs)
    {
		// 用aid和athrs数据初始化id和thrs
        memcpy(id, aid, sizeof(int)*(fea_num<<1));
        memcpy(thrs, athrs, sizeof(double)*fea_num);
    }
    void set_random(int veclen, double (*range)[2])
    {
        for (int i = 0; i < fea_num; ++i)
        {
            id[i<<1] = rand()%veclen; // veclen向量长度，随机取一个数，大小在veclen向量长度内，第一个点的id
			id[i << 1 | 1] = rand() % veclen; // veclen向量长度，随机取一个数，大小在veclen向量长度内，第二个点的id
            thrs[i] = range[i][0] + (range[i][1]-range[i][0])*(rand()%RAND_MAX) / (double)(RAND_MAX-1); // 随机生成一个阈值
        }
    }
    void set_random(int veclen, double inf, double sup)
    {
        for (int i = 0; i < fea_num; ++i)
        {
            id[i<<1] = rand()%veclen; // 随机生成一个数，大小在veclen以内，第一个点的id
            id[i<<1|1] = rand()%veclen; // 随机生成一个数，大小在veclen以内随机生成一个数，大小在veclen以内，第二个点的id
            thrs[i] = inf + (sup-inf)*(rand()%RAND_MAX) / (double)(RAND_MAX-1); // 在指定区域内生成一个阈值
        }
    }
    virtual Binary_feature *copy_self() const
    {
        Diff_Binary_feature *dbf = new Diff_Binary_feature(*this); // 对二进制特征进行复制
        return dbf;
    }
    virtual void get_feature(double *vec, int veclen)
    {   // 获取特征
        for (int i = 0; i < fea_num; ++i)
        {
			// 任意两个值算差
            if ((vec[id[i<<1]] - vec[id[i<<1|1]]) < thrs[i])
                set_bit(i);  
            else
                reset_bit(i);
        }
    }
};
// 单个蕨类
class SingleFern
{
private:
    int depth; // 蕨的深度(每个蕨拥有特征的数量)
    int class_num;  // 类别数
    double *prob;  //  先验条件概率prob[i][j] = P(F=i|C=j),大小为(1<<depth) * classnum
    Binary_feature *bf;  // 二进制特征集
    void preproc(int H)
    {
		// 对先验条件概率数组进行开辟空间
        if (bf)
            delete bf;
        bf = 0;
        int len = H*(1<<depth); // 总共需要的空间大小
        if (prob)
        {
            if (H != class_num)
            {
                delete []prob;
                prob = new double[len]; // 总共开辟 类别数*蕨深度 大小的 概率数组
            }
        }
        else
            prob = new double[len];
        class_num = H;
        for (int i = 0; i < len; prob[i++]=0);  // 初始化先验条件概率数组
    }
public:
    SingleFern(int fern_depth) : depth(fern_depth), class_num(0), prob(0), bf(0)
    {
		//构造函数，用于属性的初始化
    }
    ~SingleFern()
    {
		// 析构函数
        if (bf)
            delete bf;
        if (prob)
            delete []prob;
    }
    double train(double *X, int *C, int N, int K, int H, Binary_feature *getfea, int reg = 1)
    {
		// 训练函数，其中X表示数据,C表示标签,N表示数据量,K表示单幅图片大小, H表示类别数, getfea表示二进制点对的index信息和对应的阈值信息，reg平滑参数
        preproc(H);  // 对先验条件概率数组进行初始化
        int n = 1<<depth;  // 蕨深度
        bf = getfea->copy_self();  //对二进制点对信息进行拷贝
        double *vec = X; 
        int *cnt = new int[H];  // 用来记录每个类别出现的次数
        memset(cnt, 0, sizeof(int)*H); //初始化cnt
        for (int i = 0; i < N; ++i)
        {
			// 迭代
            bf->get_feature(vec, K); // 获取一幅图片的二进制点对
            unsigned int id = bf->get_binary(0, depth-1); // 获取特征对应的id(index) 信息
            ++prob[C[i]+id*H]; // 更新先验条件概率模型
            ++cnt[C[i]]; //更新相应类别出现的次数
            vec += K; // 移动训练数据集的指针
        }
		// 更新先验条件概率数组
        int idx = 0;
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < H; ++j)
            {
                prob[idx] = (prob[idx] + reg)/ ((double)cnt[j] + n*reg); // 概率归一化处理
                ++idx;
            }
        }
        delete []cnt; // 释放投票数组
        return evaluate(X, C, N, K); // 评估训练结果
    }
    int classify(double *vec, int veclen, double *cprob = 0)
    {
		// 分类函数
        bf->get_feature(vec, veclen); // 获取一幅图片的二进制特征
        unsigned int id = bf->get_binary(0, depth-1); // 单个蕨对应的点对id(十进制表示)
        int idx = id * class_num;  // 获得概率数组中对应的index值
        double tprob = 0; //tprob = sigma(P(F=id|C=[0...class_num)) 用于记录先验条件概率的和
        double maxprob; //maxprob = max(P(F=id|C)) 用于记录归一化后的最大概率值
        int c;
        for (int i = 0; i < class_num; ++i)
        {
            if (i == 0 || prob[idx+i] > maxprob)
                maxprob = prob[idx+i], c = i;
            tprob += prob[idx+i];
        }
        if (cprob) //cprob = max(P(C|F=id))
            *cprob = maxprob / tprob; // 归一化操作
        return c; // 返回类别
    }
    double evaluate(double *X, int *C, int N, int K, int *predict = 0)
    {
        double *vec = X; // 数据
        int rn = 0; // 临时变量，用于计数
        for (int i = 0; i < N; ++i)
        {
            int pred = classify(vec, K); //预测
            if (predict)
                predict[i] = pred;
            rn += (pred == C[i]); // 统计正确分类数
            vec += K; // 移动数据指针
        }
        return (double)rn / (double)N; // 正确率
    }
};
// 随机蕨类
class RandomFerns
{
private:
    int m; // 蕨的数量
    int depth; // 蕨的深度
    int class_num; // 类别数
    double *prob;  // 大小为 m * (1<<depth) * class_num，prob[i][j][k] = log(P(F_k = i|C = j)) 先验条件概率
    Binary_feature *bf;  // 特征集变量
    void preproc(int H)
    {
		//对先验条件概率模型进行开辟空间和初始化
        if (bf)
            delete bf;
        bf = 0;
        int len = H*(1<<depth)*m;  //总共长度
        if (prob)
        {
            if (H != class_num)
            {
                delete []prob;
                prob = new double[len];  //开辟空间
            }
        }
        else
            prob = new double[len];
        class_num = H;
        for (int i = 0; i < len; prob[i++]=0); // 初始化
    }
public:
    RandomFerns(int fern_num, int fern_depth) : m(fern_num), depth(fern_depth), class_num(0), prob(0), bf(0)
    {
		// 构造函数，用于对属性的初始化
    }
    double train(double *X, int *C, int N, int K, int H, Binary_feature *getfea, int reg = 1)
    {
		// 训练函数，其中X表示数据,C表示标签,N表示数据量,K表示单幅图片大小, H表示类别数, getfea表示二进制点对的index信息和对应的阈值信息，reg平滑参数
        preproc(H); //预处理操作
        int n = 1<<depth; // 特征长度
        bf = getfea->copy_self(); // 将所有二进制点对信息以及相应的阈值信息拷贝到bf
        double *vec = X; 
        int *cnt = new int[H];  // 开辟一个大小为类别数的数组，用于计算先验条件概率
        memset(cnt, 0, sizeof(int)*H); //cnt变量的初始化
        for (int i = 0; i < N; ++i)
        {
            bf->get_feature(vec, K); // 获取长度大小为一副图片的点对
            for (int j = 0, k = 0; j < m; ++j,k+=depth)
            {
				// 每个蕨的数据更新
                unsigned id = bf->get_binary(k, k+depth-1);  // 特征的二进制化 
                ++prob[j*n*H + id*H + C[i]];  // 更新联合概率
            }
            ++cnt[C[i]];  // 更新先验概率
            vec += K; //改变指针位置
        }
        int idx = 0;
		// 更新先验条件概率数组
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                for (int k = 0; k < H; ++k)
                {
                    prob[idx] = (prob[idx] + reg)/ ((double)cnt[k] + n*reg);  // 统计得到先验条件概率
                    prob[idx] = log(prob[idx]);
                    ++idx;
                }
            }
        }
        delete []cnt; // 释放空间
        return evaluate(X, C, N, K);
    }
    int classify(double *vec, int veclen, double *cprob = 0)
    {
		// 分类函数
        bf->get_feature(vec, veclen);  // 获取一幅图片的特征
        int n = 1<<depth; // 蕨的深度(特征的长度)
        double tprob = 0; //tprob = sigma(logp(F=feature|C=[0...class_num)) 先验条件概率的求和
        double maxprob; //maxprob = max(logp(F=feature|C)) 最大的先验条件概率
        int c; // 临时变量，计数使用
        for (int i = 0; i < class_num; ++i)
        {
            double iprob = 0; //先验条件概率iprob = log(P(F=feature|C = i))
            for (int j = 0; j < m; ++j)
            {
                unsigned id = bf->get_binary(j*depth, j*depth+depth-1); //得到二进制特征中1对应的id
                iprob += prob[j*n*class_num + id*class_num + i];
            }
            if (i == 0 || iprob > maxprob)
                maxprob = iprob, c = i;
            tprob += iprob;
        }
        if (cprob)
            *cprob = maxprob/tprob; // 归一化处理
        return c;
    }
    double evaluate(double *X, int *C, int N, int K, int *predict = 0)
    {
		// 评估函数
        double *vec = X; //数据
        int rn = 0;
        for (int i = 0; i < N; ++i)
        {
            int pred = classify(vec, K);
            if (predict)
                predict[i] = pred;
            rn += (pred == C[i]);
            vec += K;
        }
        return (double)rn / (double)N; //概率大小
    }
};
#endif // FERNS_H
