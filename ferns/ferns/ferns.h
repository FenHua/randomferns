#ifndef FERNS_H
#define FERNS_H

#include <cstdlib>
#include <cmath>

#ifndef RAND_MAX
#define RAND_MAX 65535
#endif // RAND_MAX
// ��������
class Binary_feature
{
protected:
    int fea_num; // ������������
    unsigned int *fea; // ��¼������������
    Binary_feature() : fea_num(0), fea(0) {}  // ���캯��
public:
    Binary_feature(int num) : fea_num(num)
    {
        const int nbits = sizeof(unsigned int)<<3; // ����������λ��unsigned int ռ4���ֽ�
        int len = (num + nbits-1)/nbits; // ��������
        fea = new unsigned int[len]; //���ٿռ�
        memset(fea, 0, sizeof(unsigned int)*len); // ��ʼ������
    }
    Binary_feature(const Binary_feature &bf) : fea_num(bf.fea_num)
    {
        const int nbits = sizeof(unsigned int)<<3; // ����������λ��unsigned int ռ4���ֽ�
        int len = (fea_num + nbits-1)/nbits; // ��������
        fea = new unsigned int[len];  //���ٵ��������ռ�
        memcpy(fea, bf.fea, sizeof(unsigned int)*len); // ��ʼ�������ռ�
    }
    virtual ~Binary_feature()
    {
		// ��������
        if (fea)
            delete []fea; // �ͷſ��ٵ������ռ�
    }
    unsigned int get_binary(int from, int to)
    {
		// ��ȡ��������������fea����洢
        const int nbits = sizeof(unsigned int)<<3; // ����������λ��unsigned int ռ4���ֽ�
		// ��ʼλ��
        int fp = from/(nbits); // ������nbits����
        int foff = from%(nbits); // ��������nbits�ĳ���
		// ����λ��
        int tp = to/(nbits); // ������nbits����
        int toff = to%(nbits); // ��������nbits�ĳ���
        if (fp == tp)
        {
			// ��Ҫ������������nbits��С����
            return (fea[fp]>>(nbits-1 - toff))&((1u<<(toff-foff+1))-1); // ��ȡһ�����ȵ�������С(ʮ����)
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
            return res; // ���ػ�ȡһ�����ȵ�������С(ʮ����)
        }
    }
    unsigned int get_binary()
    {
        return get_binary(0, fea_num-1); // ȱʡ����������£���ȡ���ݳ��ȵĶ���������
    }
    void set_bit(int pos)
    {
		// ��posλ����Ϊ1
        const int nbits = sizeof(unsigned int)<<3;  // 32
        int p = pos / nbits;  // ����nbits�ĸ���
        int off = pos % nbits;  // ������nbits�ĳ���
        fea[p] |= (1u<<(nbits-1-off));  // ����ָ��λ�õ�feature��С��1u,��ʾ��ǰ1���޷�������
    }
    void reset_bit(int pos)
    {
        const int nbits = sizeof(unsigned int)<<3; // 32
        int p = pos / nbits;   // ����nbits�ĸ���
        int off = pos % nbits;  // ������nbits�ĳ���
        fea[p] &= ~(1u<<(nbits-1-off)); // ��posλ�õ�����ֵ�÷�
    }
    virtual Binary_feature *copy_self() const = 0;
    virtual void get_feature(double *vec, int veclen) = 0;
};
class Diff_Binary_feature : public Binary_feature
{
private:
    int *id; // ������������Ӧ��id
    double *thrs; // �������ص�����ֵ
public:
    Diff_Binary_feature(int num) : Binary_feature(num)
    {
        id = new int[fea_num<<1]; // ��������һ���Ŀռ䣬id�ǵ��
        thrs = new double[fea_num]; //��ֵ
    }
    Diff_Binary_feature(int num, int veclen, double (*range)[2]) : Binary_feature(num)
    {
        id = new int[fea_num<<1]; // ��������һ���Ŀռ�
        thrs = new double[fea_num]; //��ֵ
        set_random(veclen, range); // ��id��thrs���и�ֵ
    }
    Diff_Binary_feature(int num, int veclen, double inf, double sup) : Binary_feature(num)
    {
        id = new int[fea_num<<1];  // ��������һ���Ŀռ�
		thrs = new double[fea_num]; //��ֵ
        set_random(veclen, inf, sup); // ��ָ������������һ�������ֵ
    }
    Diff_Binary_feature(int num, int *aid, double *athrs) : Binary_feature(num)
    {
        id = new int[fea_num<<1]; // ��������һ���Ŀռ�
        thrs = new double[fea_num]; // ��ֵ
        memcpy(id, aid, sizeof(int)*(fea_num<<1)); // �����е�id��ֵ�������ɵ�id����
        memcpy(thrs, athrs, sizeof(double)*fea_num); // �����е�thrs��ֵ���¿��ٵ�thrs����
    }
    Diff_Binary_feature(const Diff_Binary_feature &dbf) : Binary_feature(dbf)
    {
        id = new int[fea_num<<1]; // ��������һ���Ŀռ�
        thrs = new double[fea_num]; // ��ֵ
        memcpy(id, dbf.id, sizeof(int)*(fea_num<<1)); // �����еĶ����������Ե�ǰid��thrs��ֵ���г�ʼ��
        memcpy(thrs, dbf.thrs, sizeof(double)*fea_num);
    }
    virtual ~Diff_Binary_feature()
    {
		// ��������
        delete []id;
        delete []thrs;
        id = 0;
        thrs = 0;
    }
    void set_param(int *aid, double *athrs)
    {
		// ��aid��athrs���ݳ�ʼ��id��thrs
        memcpy(id, aid, sizeof(int)*(fea_num<<1));
        memcpy(thrs, athrs, sizeof(double)*fea_num);
    }
    void set_random(int veclen, double (*range)[2])
    {
        for (int i = 0; i < fea_num; ++i)
        {
            id[i<<1] = rand()%veclen; // veclen�������ȣ����ȡһ��������С��veclen���������ڣ���һ�����id
			id[i << 1 | 1] = rand() % veclen; // veclen�������ȣ����ȡһ��������С��veclen���������ڣ��ڶ������id
            thrs[i] = range[i][0] + (range[i][1]-range[i][0])*(rand()%RAND_MAX) / (double)(RAND_MAX-1); // �������һ����ֵ
        }
    }
    void set_random(int veclen, double inf, double sup)
    {
        for (int i = 0; i < fea_num; ++i)
        {
            id[i<<1] = rand()%veclen; // �������һ��������С��veclen���ڣ���һ�����id
            id[i<<1|1] = rand()%veclen; // �������һ��������С��veclen�����������һ��������С��veclen���ڣ��ڶ������id
            thrs[i] = inf + (sup-inf)*(rand()%RAND_MAX) / (double)(RAND_MAX-1); // ��ָ������������һ����ֵ
        }
    }
    virtual Binary_feature *copy_self() const
    {
        Diff_Binary_feature *dbf = new Diff_Binary_feature(*this); // �Զ������������и���
        return dbf;
    }
    virtual void get_feature(double *vec, int veclen)
    {   // ��ȡ����
        for (int i = 0; i < fea_num; ++i)
        {
			// ��������ֵ���
            if ((vec[id[i<<1]] - vec[id[i<<1|1]]) < thrs[i])
                set_bit(i);  
            else
                reset_bit(i);
        }
    }
};
// ����ާ��
class SingleFern
{
private:
    int depth; // ާ�����(ÿ��ާӵ������������)
    int class_num;  // �����
    double *prob;  //  ������������prob[i][j] = P(F=i|C=j),��СΪ(1<<depth) * classnum
    Binary_feature *bf;  // ������������
    void preproc(int H)
    {
		// ��������������������п��ٿռ�
        if (bf)
            delete bf;
        bf = 0;
        int len = H*(1<<depth); // �ܹ���Ҫ�Ŀռ��С
        if (prob)
        {
            if (H != class_num)
            {
                delete []prob;
                prob = new double[len]; // �ܹ����� �����*ާ��� ��С�� ��������
            }
        }
        else
            prob = new double[len];
        class_num = H;
        for (int i = 0; i < len; prob[i++]=0);  // ��ʼ������������������
    }
public:
    SingleFern(int fern_depth) : depth(fern_depth), class_num(0), prob(0), bf(0)
    {
		//���캯�����������Եĳ�ʼ��
    }
    ~SingleFern()
    {
		// ��������
        if (bf)
            delete bf;
        if (prob)
            delete []prob;
    }
    double train(double *X, int *C, int N, int K, int H, Binary_feature *getfea, int reg = 1)
    {
		// ѵ������������X��ʾ����,C��ʾ��ǩ,N��ʾ������,K��ʾ����ͼƬ��С, H��ʾ�����, getfea��ʾ�����Ƶ�Ե�index��Ϣ�Ͷ�Ӧ����ֵ��Ϣ��regƽ������
        preproc(H);  // ��������������������г�ʼ��
        int n = 1<<depth;  // ާ���
        bf = getfea->copy_self();  //�Զ����Ƶ����Ϣ���п���
        double *vec = X; 
        int *cnt = new int[H];  // ������¼ÿ�������ֵĴ���
        memset(cnt, 0, sizeof(int)*H); //��ʼ��cnt
        for (int i = 0; i < N; ++i)
        {
			// ����
            bf->get_feature(vec, K); // ��ȡһ��ͼƬ�Ķ����Ƶ��
            unsigned int id = bf->get_binary(0, depth-1); // ��ȡ������Ӧ��id(index) ��Ϣ
            ++prob[C[i]+id*H]; // ����������������ģ��
            ++cnt[C[i]]; //������Ӧ�����ֵĴ���
            vec += K; // �ƶ�ѵ�����ݼ���ָ��
        }
		// ��������������������
        int idx = 0;
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < H; ++j)
            {
                prob[idx] = (prob[idx] + reg)/ ((double)cnt[j] + n*reg); // ���ʹ�һ������
                ++idx;
            }
        }
        delete []cnt; // �ͷ�ͶƱ����
        return evaluate(X, C, N, K); // ����ѵ�����
    }
    int classify(double *vec, int veclen, double *cprob = 0)
    {
		// ���ຯ��
        bf->get_feature(vec, veclen); // ��ȡһ��ͼƬ�Ķ���������
        unsigned int id = bf->get_binary(0, depth-1); // ����ާ��Ӧ�ĵ��id(ʮ���Ʊ�ʾ)
        int idx = id * class_num;  // ��ø��������ж�Ӧ��indexֵ
        double tprob = 0; //tprob = sigma(P(F=id|C=[0...class_num)) ���ڼ�¼�����������ʵĺ�
        double maxprob; //maxprob = max(P(F=id|C)) ���ڼ�¼��һ�����������ֵ
        int c;
        for (int i = 0; i < class_num; ++i)
        {
            if (i == 0 || prob[idx+i] > maxprob)
                maxprob = prob[idx+i], c = i;
            tprob += prob[idx+i];
        }
        if (cprob) //cprob = max(P(C|F=id))
            *cprob = maxprob / tprob; // ��һ������
        return c; // �������
    }
    double evaluate(double *X, int *C, int N, int K, int *predict = 0)
    {
        double *vec = X; // ����
        int rn = 0; // ��ʱ���������ڼ���
        for (int i = 0; i < N; ++i)
        {
            int pred = classify(vec, K); //Ԥ��
            if (predict)
                predict[i] = pred;
            rn += (pred == C[i]); // ͳ����ȷ������
            vec += K; // �ƶ�����ָ��
        }
        return (double)rn / (double)N; // ��ȷ��
    }
};
// ���ާ��
class RandomFerns
{
private:
    int m; // ާ������
    int depth; // ާ�����
    int class_num; // �����
    double *prob;  // ��СΪ m * (1<<depth) * class_num��prob[i][j][k] = log(P(F_k = i|C = j)) ������������
    Binary_feature *bf;  // ����������
    void preproc(int H)
    {
		//��������������ģ�ͽ��п��ٿռ�ͳ�ʼ��
        if (bf)
            delete bf;
        bf = 0;
        int len = H*(1<<depth)*m;  //�ܹ�����
        if (prob)
        {
            if (H != class_num)
            {
                delete []prob;
                prob = new double[len];  //���ٿռ�
            }
        }
        else
            prob = new double[len];
        class_num = H;
        for (int i = 0; i < len; prob[i++]=0); // ��ʼ��
    }
public:
    RandomFerns(int fern_num, int fern_depth) : m(fern_num), depth(fern_depth), class_num(0), prob(0), bf(0)
    {
		// ���캯�������ڶ����Եĳ�ʼ��
    }
    double train(double *X, int *C, int N, int K, int H, Binary_feature *getfea, int reg = 1)
    {
		// ѵ������������X��ʾ����,C��ʾ��ǩ,N��ʾ������,K��ʾ����ͼƬ��С, H��ʾ�����, getfea��ʾ�����Ƶ�Ե�index��Ϣ�Ͷ�Ӧ����ֵ��Ϣ��regƽ������
        preproc(H); //Ԥ�������
        int n = 1<<depth; // ��������
        bf = getfea->copy_self(); // �����ж����Ƶ����Ϣ�Լ���Ӧ����ֵ��Ϣ������bf
        double *vec = X; 
        int *cnt = new int[H];  // ����һ����СΪ����������飬���ڼ���������������
        memset(cnt, 0, sizeof(int)*H); //cnt�����ĳ�ʼ��
        for (int i = 0; i < N; ++i)
        {
            bf->get_feature(vec, K); // ��ȡ���ȴ�СΪһ��ͼƬ�ĵ��
            for (int j = 0, k = 0; j < m; ++j,k+=depth)
            {
				// ÿ��ާ�����ݸ���
                unsigned id = bf->get_binary(k, k+depth-1);  // �����Ķ����ƻ� 
                ++prob[j*n*H + id*H + C[i]];  // �������ϸ���
            }
            ++cnt[C[i]];  // �����������
            vec += K; //�ı�ָ��λ��
        }
        int idx = 0;
		// ��������������������
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                for (int k = 0; k < H; ++k)
                {
                    prob[idx] = (prob[idx] + reg)/ ((double)cnt[k] + n*reg);  // ͳ�Ƶõ�������������
                    prob[idx] = log(prob[idx]);
                    ++idx;
                }
            }
        }
        delete []cnt; // �ͷſռ�
        return evaluate(X, C, N, K);
    }
    int classify(double *vec, int veclen, double *cprob = 0)
    {
		// ���ຯ��
        bf->get_feature(vec, veclen);  // ��ȡһ��ͼƬ������
        int n = 1<<depth; // ާ�����(�����ĳ���)
        double tprob = 0; //tprob = sigma(logp(F=feature|C=[0...class_num)) �����������ʵ����
        double maxprob; //maxprob = max(logp(F=feature|C)) ����������������
        int c; // ��ʱ����������ʹ��
        for (int i = 0; i < class_num; ++i)
        {
            double iprob = 0; //������������iprob = log(P(F=feature|C = i))
            for (int j = 0; j < m; ++j)
            {
                unsigned id = bf->get_binary(j*depth, j*depth+depth-1); //�õ�������������1��Ӧ��id
                iprob += prob[j*n*class_num + id*class_num + i];
            }
            if (i == 0 || iprob > maxprob)
                maxprob = iprob, c = i;
            tprob += iprob;
        }
        if (cprob)
            *cprob = maxprob/tprob; // ��һ������
        return c;
    }
    double evaluate(double *X, int *C, int N, int K, int *predict = 0)
    {
		// ��������
        double *vec = X; //����
        int rn = 0;
        for (int i = 0; i < N; ++i)
        {
            int pred = classify(vec, K);
            if (predict)
                predict[i] = pred;
            rn += (pred == C[i]);
            vec += K;
        }
        return (double)rn / (double)N; //���ʴ�С
    }
};
#endif // FERNS_H
