package com.bird.vector;

import com.bird.vector.common.VectorTools;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created with IntelliJ IDEA.
 *
 * @description：
 * @author： liuxiangqian
 * @date： 2024/10/14
 */
@Data
@Slf4j
public class Kmeans {
    private static String COMMA = ",";

    /**
     * MIN_CLUSTER_COUNT:   最小聚类数
     * MIN_ITER_COUNT:      最小迭代数
     * MINI_POSITIVE_DECIMAL:   用来判断0的最小正小数
     */
    private static int MIN_CLUSTER_COUNT = 2;
    private static int MIN_ITER_COUNT = 5;
    private static float MINI_POSITIVE_DECIMAL = 0.000001f;

    /**
     * clusterCount: 聚类数目
     * maxIterCount: 最大迭代次数
     * dataCount:    数据集中数据的个数
     * vectorDimension:      数据的向量维度
     * vectors:         数据集
     * centers:       聚类中心，结构与各数据点相同
     * cluster:      聚类形成的簇
     * clusterSSEs:          聚类中系统整体误差平方和
     * iterCount:    用于记录最终迭代次数
     * initCenter:   初始聚类中心
     */
    private int clusterCount;
    private int maxIterCount;
    private int dataCount;
    private int vectorDimension;
    private List<float[]> vectors;
    private List<float[]> centers;
    private List<List<float[]>> cluster;
    private List<Float> clusterSSEs = new ArrayList<>();
    ;
    private int iterCount;
    private List<float[]> initCenter;

    /**
     * @param clusterCount    聚类数
     * @param maxIterCount    训练最大迭代数
     * @param vectorDimension 训练数据的向量的维数
     */
    public Kmeans(int clusterCount, int maxIterCount, int vectorDimension) {
        clusterCount = clusterCount > MIN_CLUSTER_COUNT ? clusterCount : MIN_CLUSTER_COUNT;
        maxIterCount = maxIterCount > MIN_ITER_COUNT ? maxIterCount : MIN_ITER_COUNT;

        this.clusterCount = clusterCount;
        this.maxIterCount = maxIterCount;
        this.vectorDimension = vectorDimension;
    }

    /**
     * 聚类算法训练
     *
     * @param csvFilePath 向量文件，文件结构: id#vector  vector格式： d1,d2,....,dn
     */
    public void train(String csvFilePath) {
        Pair<List<Integer>, List<float[]>> idsAndVectors = VectorTools.laodIdsAndVectors(csvFilePath);
        this.vectors = idsAndVectors.getValue();
        init();
        train();
    }

    /**
     * 聚类算法训练
     *
     * @param vectors 向量集合
     */
    public void train(List<float[]> vectors) {
        this.vectors = vectors;
        vectorDimension = vectorDimension < 0 ? vectors.get(0).length : vectorDimension;
        vectors.forEach(vector -> {
            if (vector.length != vectorDimension) {
                log.error("数据维度不一致 startVectorLength:{} curVectorLength:{} ", vectorDimension, vector.length);
                log.error("当前的向量为:{}", vector);
            }
        });

        init();
        train();
    }

    private void train() {
        iterCount = 1;
        while (true) {
            dataClassification();
            computeSSE();
            if (iterCount > maxIterCount) {
                break;
            }

            if (clusterSSEs.size() > 1) {
                float diff = Math.abs(clusterSSEs.get(iterCount - 2) - clusterSSEs.get(iterCount - 1));
                if (diff <= MINI_POSITIVE_DECIMAL) {
                    break;
                }
            }

            updateCenter();
            log.info("第{}次迭代完成", iterCount);

            cluster.clear();
            initCluster();
            iterCount++;
        }
    }

    /**
     * 聚类数调整
     * 聚类中心初始化
     * 聚类初始化
     */
    private void init() {
        dataCount = vectors.size();
        clusterCount = clusterCount > dataCount ? dataCount : clusterCount;

        initCenter();
        initCluster();
    }

    /**
     * 显示聚类中心
     *
     * @param center
     */
    public void printCenter(List<float[]> center) {
        log.info("聚类中心为:");
        for (int i = 0; i < center.size(); i++) {
            StringBuilder sb = new StringBuilder();
            sb.append("[");
            for (int j = 0; j < vectorDimension; j++) {
                sb.append(center.get(i)[j]).append(COMMA);
            }
            sb.deleteCharAt(sb.length() - 1);
            sb.append("]");

            log.info("第{}个类中心:{}", i, sb.toString());
        }
    }


    /**
     * 随机生成类中心，初始化类中心
     */
    public void initCenter() {
        centers = new ArrayList<float[]>();

        //随机选择一个点作为中心
        Random random = new Random();
        int index = random.nextInt(dataCount);
        index = 1;
        float[] rdCenter = vectors.get(index);
        centers.add(rdCenter);
        log.info("第{}个初始聚类中心：{}", centers.size(), rdCenter);
        for (int i = 1; i < clusterCount; i++) {
            //步骤二： 选择与centers中所有的中心距离的最大归一乘积作为新的中心  diffMultiply/diffSum
            float maxUniformDiff = 0.0f;
            float maxDiffSum = 0.0f;
            float[] newCenter = null;
            for (int j = 0; j < dataCount; j++) {
                float[] vector = vectors.get(j);
                float tmpUniformDiff = 0.0f;
                float tmpDiffSum = 0.0f;
                float tmpMultiply = 1.0f;

                //要确保与各个中心的距离都有一定距离
                List<Float> centerDiffs = new ArrayList<>(8);
                for (float[] center : centers) {
                    float diff = VectorTools.continentalDistance(vector, center);
                    tmpDiffSum += diff;
                    tmpMultiply += Math.log(diff);

                    centerDiffs.add(diff);
                }

                if (i == 1) {
                    if (tmpDiffSum > maxDiffSum) {
                        maxDiffSum = tmpDiffSum;
                        newCenter = vector;
                    }

                    continue;
                }

                tmpUniformDiff = tmpMultiply - (float) Math.log(tmpDiffSum);
                tmpUniformDiff = weight(centerDiffs) * tmpUniformDiff;
                if (tmpUniformDiff > maxUniformDiff) {
                    maxUniformDiff = tmpUniformDiff;
                    newCenter = vector;
                }
            }

            centers.add(newCenter);
        }

        //保存初始聚类中心的副本
        initCenter = new ArrayList<>();
        initCenter.addAll(centers);

        printCenter(initCenter);

        log.info("初始化完毕");
    }

    /**
     * 对k个类进行数据初始化
     */
    public void initCluster() {
        cluster = new ArrayList<>();
        for (int i = 0; i < clusterCount; i++) {
            cluster.add(Collections.synchronizedList(new ArrayList<>()));
        }
    }

    /**
     * 打印每个类的向量成员
     *
     * @param clusters
     */
    public void printClusterVector(List<List<float[]>> clusters) {
        for (int i = 0; i < clusters.size(); i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < clusters.get(i).size(); j++) {

                sb.append("[");
                for (int index = 0; index < vectorDimension; index++) {
                    sb.append(clusters.get(i).get(j)[index]).append(COMMA);
                }

                sb.deleteCharAt(sb.length() - 1);
                sb.append("]").append(COMMA);
            }

            log.debug("第{}个类包含的向量为: {}", i, sb.toString());
        }
    }

    /**
     * 获取最小的距离
     *
     * @param distances
     * @return 最小距离对应的下标
     */
    private int minContinentalDistance(float[] distances) {
        float minDistance = distances[0];
        int minIndex = 0;
        for (int i = 1; i < distances.length; i++) {
            if (distances[i] <= minDistance) {
                minDistance = distances[i];
                minIndex = i;
            }
        }

        return minIndex;
    }

    /**
     * 根据欧式距离，将数据归类到各个类中
     */
    private void dataClassification() {
//        printCenter(centers);

        vectors.parallelStream().forEach(vector -> {
            float[] dis = new float[clusterCount];
            for (int j = 0; j < clusterCount; j++) {
                dis[j] = VectorTools.continentalDistance(vector, centers.get(j));
            }

            int location = minContinentalDistance(dis);
            cluster.get(location).add(vector);
        });

        //显示此时簇中包含的元素
        // printClusterVector(cluster);
    }

    /**
     * 计算当前分类中所有簇中误差平方和
     */
    private void computeSSE() {
        float oneSSE = 0;
        for (int i = 0; i < cluster.size(); i++) {
            for (int j = 0; j < cluster.get(i).size(); j++) {
                //计算当前簇中的所有数据到该簇聚类中心的距离
                oneSSE += VectorTools.diffSquare(cluster.get(i).get(j), centers.get(i));
            }
        }

        clusterSSEs.add(oneSSE);
    }

    /**
     * 更新聚类中心
     */
    private void updateCenter() {
        int centerCount = cluster.size();
        for (int i = 0; i < centerCount; i++) {
            float[] newCenter = new float[vectorDimension];
            //聚类的处理
            List<float[]> vectors = cluster.get(i);
            int vectorCount = vectors.size();
            if (0 == vectorCount) {
                continue;
            }

            for (int j = 0; j < vectorCount; j++) {
                //向量的处理
                float[] vector = vectors.get(j);
                for (int k = 0; k < vectorDimension; k++) {
                    //向量维度粒度处理
                    newCenter[k] += vector[k];
                }
            }

            for (int k = 0; k < vectorDimension; k++) {
                newCenter[k] = newCenter[k] / vectorCount;
            }
            centers.set(i, newCenter);
        }
    }

    /**
     * 对小距离进行降权
     *
     * @param centerDiffs
     * @return
     */
    private float weight(List<Float> centerDiffs) {
        if (centerDiffs.size() < 2) {
            //权重不需要改变
            return 1.0f;
        }

        Collections.sort(centerDiffs);
        float minWeight = 1.0f;
        for (int i = 1; i < centerDiffs.size(); i++) {
            float tmpWeight = centerDiffs.get(i - 1) / centerDiffs.get(i);
            if (tmpWeight < minWeight) {
                minWeight = tmpWeight;
            }
        }

        return minWeight;
    }

    /**
     * 显示聚类最终信息
     */
    public void show() {
        log.info("训练开始的聚类中心：");
        printCenter(initCenter);

        log.info("训练结束的聚类中心：");
        printCenter(centers);

        log.info("迭代执行的次数为：{}", iterCount);
        log.info("各阶段系统误差平方和");
        for (int i = 0; i < clusterSSEs.size(); i++) {
            log.info("{}:{}", i, clusterSSEs.get(i));
        }

        //显示最后系统中各簇中的元素
        log.info("各个聚类的向量集合");
//        printClusterVector(cluster);
    }
}
