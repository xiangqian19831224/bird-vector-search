package com.bird.vector;

import com.bird.vector.common.VectorTools;
import com.bird.vector.utils.FolderTools;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.tuple.Pair;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * 量化训练
 * 1.对向量进行均匀分段
 * 2.对分段后的向量集合进行聚类
 * 3.对聚类中心进行编号和存储
 * 量化访问
 * 1.初始化对向量中心和编号进行加载
 * 2.对访问的向量进行分段
 * 3.计算分段向量的最近topn向量中心及其编号
 *
 * @description：
 * @author： liuxiangqian
 * @date： 2024/10/15
 */
@Slf4j
@Data
public class EmPQ {
    private static String MODEL_PARAM = "param.data";
    private static String MODEL_CENTERS = "center.data";
    private static int MAX_TRAIN_COUNT = 500000;


    /**
     * pqSegmentCount：  向量分段数
     * clusterCount：    聚类数
     * maxIterCount：    聚类最大迭代数
     * dim：             向量维数
     * vectors：            数据集
     * centersList：     聚类中心   [段1的聚类，段2的聚类，...]
     * *    格式： [[<id1, center1><id2,center2>,....],[],[]...]
     */
    private int pqSegmentCount;
    private int clusterCount;
    private int maxIterCount;
    private int vectorDimension;
    private List<List<Pair<Integer, float[]>>> centersList;

    EmPQ(int pqSegmentCount, int clusterCount, int maxIterCount, int vectorDimension) {
        this.pqSegmentCount = pqSegmentCount;
        this.clusterCount = clusterCount;
        this.maxIterCount = maxIterCount;
        this.vectorDimension = vectorDimension;
    }

    public void train(String csvFilePath) {
        Pair<List<Integer>, List<float[]>> idsAndvectors = VectorTools.laodIdsAndVectors(csvFilePath);
        List<float[]> vectors = idsAndvectors.getValue();
        resizeList(vectors, MAX_TRAIN_COUNT);
        System.gc();

        cluster(vectors);
    }

    public void train(List<float[]> vectors) {
        Collections.shuffle(vectors);
        resizeList(vectors, MAX_TRAIN_COUNT);
        System.gc();

        cluster(vectors);
    }

    /**
     * 对向量集合进行分段和聚类，
     * * 每个向量段获得一组聚类中心
     *
     * @param vectors 向量集合
     * @return
     */
    public List<List<float[]>> cluster(List<float[]> vectors) {
        log.info("开始聚类训练...");
        long start = System.currentTimeMillis();

        //步骤一： 每个向量段进行聚类
        List<List<float[]>> vectorsList = vectorSegment(vectors);
        List<List<float[]>> centersList = Collections.synchronizedList(new ArrayList<>(vectorsList.size()));
        vectorsList.parallelStream().forEach(segVectors -> {
            List<float[]> centers = segCluster(segVectors);
            centersList.add(centers);
        });

        //步骤二： 收集聚类中心
        this.centersList = new ArrayList<>(centersList.size());
        centersList.forEach(centers -> {
            List<Pair<Integer, float[]>> centerPairs = new ArrayList<>(centers.size());
            for (int i = 0; i < centers.size(); i++) {
                Pair<Integer, float[]> pair = Pair.of(i, centers.get(i));
                centerPairs.add(pair);
            }

            this.centersList.add(centerPairs);
        });

        long took = (System.currentTimeMillis() - start) / 1000;
        log.info("聚类训练完毕，训练向量数：{} 训练耗时：{}秒", vectors.size(), took);
        return centersList;
    }

    /**
     * 将pg模型存储到磁盘
     *
     * @param modelDir 模型目录
     */
    public void store(String modelDir) {
        modelDir = FolderTools.folderAppendSlash(modelDir);
        FolderTools.createFolder(modelDir);

        String centerPath = modelDir + MODEL_CENTERS;
        deleteFile(centerPath);
        try (FileOutputStream fos = new FileOutputStream(centerPath);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(this.centersList);
        } catch (Exception e) {
            e.printStackTrace();
        }

        List<Integer> params = new ArrayList<>(4);
        params.add(pqSegmentCount);
        params.add(clusterCount);
        params.add(maxIterCount);
        params.add(vectorDimension);
        String paramPath = modelDir + MODEL_PARAM;
        deleteFile(paramPath);
        try (FileOutputStream fos = new FileOutputStream(paramPath);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(params);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 加载pg模型
     *
     * @param modelDir
     */
    public void load(String modelDir) {
        modelDir = FolderTools.folderAppendSlash(modelDir);
        String centerPath = modelDir + MODEL_CENTERS;
        try (FileInputStream fis = new FileInputStream(centerPath);
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            this.centersList = (List<List<Pair<Integer, float[]>>>) ois.readObject();
        } catch (Exception e) {
            e.printStackTrace();
        }

        List<Integer> params = new ArrayList<>(4);
        String paramPath = modelDir + MODEL_PARAM;
        try (FileInputStream fis = new FileInputStream(paramPath);
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            params = (List<Integer>) ois.readObject();
        } catch (Exception e) {
            e.printStackTrace();
        }

        assert params.size() == 4;
        pqSegmentCount = params.get(0);
        clusterCount = params.get(1);
        maxIterCount = params.get(2);
        vectorDimension = params.get(3);
    }

    /**
     * 对向量进行最近n个中心查询
     *
     * @param vector 查询向量
     * @param topn   查询topn聚类中心
     * @return 各个子段最相近的topn个聚类中心
     */
    public List<List<Pair<Integer, Float>>> search(float[] vector, int topn) {
        assert vector.length == vectorDimension;
        assert topn > 0;

        List<List<Pair<Integer, Float>>> segDisListList = new ArrayList<>(pqSegmentCount);
        int pgSegmentLength = vectorDimension / pqSegmentCount;
        List<float[]> segVectors = vectorSegment(vector, pgSegmentLength);
        for (int i = 0; i < segVectors.size(); i++) {
            List<Pair<Integer, Float>> segDisList = similary(i, segVectors.get(i), topn);
            segDisListList.add(segDisList);
        }

        return segDisListList;
    }

    /**
     * 对向量进行量化
     *
     * @param vector 查询向量
     * @return 各个子段编号
     */
    public List<Integer> pq(float[] vector) {
        assert vector.length == vectorDimension;
        List<Integer> pgIds = new ArrayList<>(pqSegmentCount);

        int pgSegmentLength = vectorDimension / pqSegmentCount;
        List<float[]> segVectors = vectorSegment(vector, pgSegmentLength);
        for (int i = 0; i < segVectors.size(); i++) {
            List<Pair<Integer, Float>> segDisList = similary(i, segVectors.get(i), 1);
            pgIds.add(segDisList.get(0).getKey());
        }

        return pgIds;
    }

    /**
     * 获取最近的聚类id及距离
     *
     * @param segNum    查询的段号
     * @param segVector 查询向量的子段
     * @param topn      获取最近topn个聚类中心
     * @return 获取最近的topn聚类id和距离
     */
    private List<Pair<Integer, Float>> similary(int segNum, float[] segVector, int topn) {
        List<Pair<Integer, float[]>> vectorPairs = centersList.get(segNum);
        assert topn >= vectorPairs.size();

        List<Pair<Integer, Float>> vectorDistances = new ArrayList<>(vectorPairs.size());
        vectorPairs.forEach(vectorPair -> {
            float conDis = VectorTools.continentalDistance(segVector, vectorPair.getRight());
            Pair<Integer, Float> vectorDistance = Pair.of(vectorPair.getLeft(), conDis);
            vectorDistances.add(vectorDistance);
        });
        vectorDistances.sort(Comparator.comparing(Pair::getValue));

        return vectorDistances.subList(0, topn);
    }

    /**
     * 对向量进行切分
     *
     * @return
     */
    private List<List<float[]>> vectorSegment(List<float[]> vectors) {
        //确保向量可以均匀分段
        assert vectorDimension % pqSegmentCount == 0;

        int pgSegmentLength = vectorDimension / pqSegmentCount;
        List<List<float[]>> segmentVectorsList = new ArrayList<>(pgSegmentLength);
        for (int i = 0; i < pqSegmentCount; i++) {
            segmentVectorsList.add(new ArrayList<>());
        }

        vectors.forEach(vector -> {
            if (vector.length != vectorDimension) {
                log.error("向量长度有误:{}", vector);
                return;
            }

            List<float[]> vectorSegs = vectorSegment(vector, pgSegmentLength);
            for (int i = 0; i < vectorSegs.size(); i++) {
                segmentVectorsList.get(i).add(vectorSegs.get(i));
            }
        });

        return segmentVectorsList;
    }

    /**
     * 对单个向量进行分段
     *
     * @param vector          向量
     * @param pgSegmentLength 向量段长度
     * @return 向量段集合
     */
    private List<float[]> vectorSegment(float[] vector, int pgSegmentLength) {
        List<float[]> vectorSegs = new ArrayList<>();
        for (int i = 0, j = 0; i < vector.length; i += pgSegmentLength, j++) {
            int start = i;
            float[] segmentVector = new float[pgSegmentLength];
            System.arraycopy(vector, start, segmentVector, 0, pgSegmentLength);
            vectorSegs.add(segmentVector);
        }

        return vectorSegs;
    }

    /**
     * 根据向量集合训练获取聚类中心
     *
     * @param vectors 向量集合
     * @return
     */
    private List<float[]> segCluster(List<float[]> vectors) {
        Kmeans kmeans = new Kmeans(clusterCount, maxIterCount, vectorDimension / pqSegmentCount);
        kmeans.train(vectors);
        List<float[]> centers = kmeans.getCenters();

        return centers;
    }


    /**
     * 删除指定文件
     *
     * @param filePath 文件路径
     */
    private void deleteFile(String filePath) {
        File file = new File(filePath);
        if (file.isDirectory()) {
            log.error("删除的是目录,filePath:{}", filePath);
            return;
        }

        if (file.exists()) {
            file.delete();
        }
    }

    /**
     * 将list的数据量限制到maxCount
     *
     * @param vectors 向量列表
     */
    private void resizeList(List<float[]> vectors, int maxCount) {
        Collections.shuffle(vectors);
        int end = vectors.size() > maxCount ? maxCount : vectors.size();
        for (int i = vectors.size() - 1; i >= end; i--) {
            vectors.remove(i);
        }
    }
}


