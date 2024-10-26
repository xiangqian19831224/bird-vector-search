package com.bird.vector;

import ai.onnxruntime.OrtException;
import com.bird.vector.common.TextTools;
import com.bird.vector.utils.Separators;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created with IntelliJ IDEA.
 *
 * @description：
 * @author： liuxiangqian
 * @date： 2024/10/26
 */
@Slf4j
public class VectorSearchTest {
    private static int pqSegmentCount = 16;
    private static int clusterCount = 16;
    private static int maxIterCount = 100;
    //vectorDimention 要跟embedding的维度一致
    private static int vectorDimention = 768;
    private static int clusterTopn = 10;
    private static int topn = 10;
    private static String dataPath = "data/title.csv";
    private static String embeddingModelDir = "bgemodel/bge-base-zh-v1.5/";
    private static Embedding embedding = new Embedding(embeddingModelDir, false);

    private static String pqModelDir = "pqmodel/";
    private static String searchModelDir = "model/";

    /**
     * 创建索引并存储
     */
    public static void createIndexAndStore() {
        Pair<List<Integer>, List<String>> docs = TextTools.loadIdsAndTexts(dataPath);
        List<float[]> vectors = embedding(docs);

        EmPQ pq = new EmPQ(pqSegmentCount, clusterCount, maxIterCount, vectorDimention);
        pq.train(vectors);
        pq.store(pqModelDir);

        EmIndex emIndex = new EmIndex(pq);
        VectorSearch vectorSearch = new VectorSearch(emIndex, embedding);
        vectorSearch.addTexts(docs);

        vectorSearch.store(searchModelDir);
    }

    /**
     * 加载索引并访问
     */
    public static void loadIndexAndSearch() {
        EmPQ pq = new EmPQ(pqSegmentCount, clusterCount, maxIterCount, vectorDimention);
        pq.load(pqModelDir);

        EmIndex emIndex = new EmIndex(pq);
        VectorSearch vectorSearch = new VectorSearch(emIndex, embedding);
        vectorSearch.load(searchModelDir);

        String query = "青岛虹桥之光文化传媒有限公司";
        List<Pair<Integer, Pair<Float, List<String>>>> recallDocs = vectorSearch.searchText(query, clusterTopn, topn);
        printRecallDocs(recallDocs);
    }


    private static List<float[]> embedding(Pair<List<Integer>, List<String>> docs) {
        List<float[]> vectors = Collections.synchronizedList(new ArrayList<>(docs.getKey().size()));
        log.info("开始向量化....");
        long start = System.currentTimeMillis();
        docs.getRight().parallelStream().forEach(content -> {
            float[] vector = new float[0];
            try {
                vector = embedding.encode(content);
            } catch (OrtException e) {
                e.printStackTrace();
            }
            vectors.add(vector);
        });
        log.info("嵌入文档数：{} 嵌入耗时：{}ms", vectors.size(), (System.currentTimeMillis() - start));
        return vectors;
    }

    private static void printRecallDocs(List<Pair<Integer, Pair<Float, List<String>>>> recallDocs) {
        for (int i = 0; i < recallDocs.size(); i++) {
            Pair<Integer, Pair<Float, List<String>>> doc = recallDocs.get(i);
            int id = doc.getKey();
            float dis = doc.getValue().getKey();
            List<String> contents = doc.getValue().getValue();

            StringBuilder sb = new StringBuilder();
            sb.append(Separators.NEW_LINE).append("id:").append(id).append(Separators.COMMA)
                    .append("mindis:").append(dis).append("[");
            contents.forEach(content -> {
                sb.append(content).append(Separators.COMMA);
            });

            sb.deleteCharAt(sb.length() - 1);
            sb.append("]");

            log.info(sb.toString());
        }
    }

    public static void main(String[] args) {
        createIndexAndStore();
        loadIndexAndSearch();
    }
}
