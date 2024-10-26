package com.bird.vector;

import ai.onnxruntime.OrtException;
import com.bird.vector.common.TextTools;
import com.bird.vector.utils.FolderTools;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.tuple.Pair;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created with IntelliJ IDEA.
 *
 * @description：
 * @author： liuxiangqian
 * @date： 2024/10/25
 */
@Slf4j
public class VectorSearch {
    /**
     * IVT_PARENT_DIR： 索引父目录
     * INDEX_DOCS：       参与索引的文档信息存储
     */
    private static String IVT_PARENT_DIR = "ivt";
    private static String INDEX_DOCS = "docs.data";
    private Map<Integer, List<String>> idToTextsMap = new HashMap<>(1024);
    private Embedding embedding;
    private EmIndex emIndex;

    public VectorSearch(EmIndex emIndex, Embedding embedding) {
        this.emIndex = emIndex;
        this.embedding = embedding;
    }

    public void addTexts(String dataPath) throws OrtException {
        Pair<List<Integer>, List<String>> idsAndVectorsPair = TextTools.loadIdsAndTexts(dataPath);
        List<String> texts = idsAndVectorsPair.getValue();
        List<Integer> ids = idsAndVectorsPair.getKey();

        for (int i = 0; i < texts.size(); i++) {
            int id = ids.get(i);
            String text = texts.get(i);
            addText(id, text);
        }
    }

    public void addTexts(List<Pair<Integer, String>> idToTextList) {
        idToTextList.forEach(idToText -> {
            int id = idToText.getKey();
            String text = idToText.getValue();
            try {
                addText(id, text);
            } catch (OrtException e) {
                e.printStackTrace();
            }
        });
    }

    public void addTexts(Pair<List<Integer>, List<String>> idsToTextsPair) {
        List<Integer> ids = idsToTextsPair.getKey();
        List<String> texts = idsToTextsPair.getValue();
        for (int i = 0; i < ids.size(); i++) {
            int id = ids.get(i);
            String text = texts.get(i);
            try {
                addText(id, text);
            } catch (OrtException e) {
                e.printStackTrace();
            }
        }
    }

    public void addText(int id, String text) throws OrtException {
        List<String> texts = idToTextsMap.getOrDefault(id, new ArrayList<>(1));
        texts.add(text);
        idToTextsMap.put(id, texts);

        float[] vector = embedding.encode(text);
        emIndex.addVector(vector, id);
    }

    public void store(String indexDir) {
        indexDir = FolderTools.folderAppendSlash(indexDir);
        String ivtPrefix = "full";
        String ivtDir = indexDir + IVT_PARENT_DIR;
        emIndex.store(ivtDir, ivtPrefix);

        String docsPath = indexDir + INDEX_DOCS;
        storeDocs(docsPath);
    }

    /**
     * 加载检索模型
     *
     * @param indexDir
     */
    public void load(String indexDir) {
        indexDir = FolderTools.folderAppendSlash(indexDir);
        String ivtPrefix = "full";
        String ivtDir = indexDir + IVT_PARENT_DIR;
        emIndex.load(ivtDir, ivtPrefix);

        String docsPath = indexDir + INDEX_DOCS;
        loadDocs(docsPath);
    }

    /**
     * @param query       查询query
     * @param clusterTopn 向量每个字段获取的类别数
     * @param topn        获取最相近向量个数
     * @return 获取id列表和最小相似距离(1个id可能有多个向量 ）
     */
    public List<Pair<Integer, Pair<Float, List<String>>>> searchText(String query, int clusterTopn, int topn) {
        List<Pair<Integer, Pair<Float, List<String>>>> idContentsPairs = new ArrayList<>(topn);
        long start = System.currentTimeMillis();
        try {
            float[] vector = embedding.encode(query);
            List<Pair<Integer, Float>> idDisPairs = emIndex.searchDocs(vector, clusterTopn, topn);
            idDisPairs.forEach(pair -> {
                int id = pair.getKey();
                float dis = pair.getValue();

                List<String> contents = this.idToTextsMap.get(id);
                idContentsPairs.add(Pair.of(id, Pair.of(dis, contents)));
            });


        } catch (OrtException e) {
            e.printStackTrace();
        }
        log.info("搜索耗时:{}毫秒", (System.currentTimeMillis() - start));
        return idContentsPairs;
    }

    /**
     * 存储文本信息
     *
     * @param filePath 文本路径
     */
    private void storeDocs(String filePath) {
        try (FileOutputStream fos = new FileOutputStream(filePath);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(this.idToTextsMap);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 加载文本信息
     *
     * @param filePath 文本路径
     */
    private void loadDocs(String filePath) {
        try (FileInputStream fis = new FileInputStream(filePath);
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            this.idToTextsMap = (Map<Integer, List<String>>) ois.readObject();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

