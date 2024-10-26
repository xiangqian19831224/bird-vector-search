
    
# 研发目的
    1.方便基于java栈开发高性能、高精度和低成本的向量检索服务
    2.方便工程领域利用java生态的优势，更容易研发高性能工程服务
    3.方便搜索领域的文本检索和向量检索的融合查询
# 使用说明
    1.VectorGen可以用来生成测试向量文件
        int centerCount = 16;               生成的向量中心数
        int vecCountPerCenter = 1000;       每个向量中心对应的向量数
        int vecDim = 1024;                  向量的维数
    2. PQTest用来生成pq模型（量化模型）
        int pqSegmentCount = 16;            每个向量划分段数
        int clusterCount = 16;              每个向量段的聚类数
        int maxIterCount = 100;             训练的最大迭代数
        int vectorDimension = 1024;         向量的维数
        说明：
            clusterCount并不需要太多，一个向量命中的概率是
    3. EmIndexTest用来创建、存储、加载和访问索引
        支持精确距离计算
    4. KmeansTest是用来聚类的，被PQTest调用
        聚类的关键在于选择初始中心
        4.1 基于距离值选择top2初始聚类中心   基本没有任何问题
        4.2 基于备选初始中心与各个初始中心的距离和距离的比值心情进行选择，这是一个技巧
            基于距离可以大致区分不同的聚类中心
            基于距离的比值可以降低新增备选中心属于已有聚类中心的风险
#  使用案例
    参考Test，代码比较简洁，可以工程中直接运行；
    如果有疑问可以联系。
#  向量化模型下载
    链接：https://pan.baidu.com/s/1j_kyUnjWnh7fiM67cTyH4w?pwd=1234 
    提取码：1234