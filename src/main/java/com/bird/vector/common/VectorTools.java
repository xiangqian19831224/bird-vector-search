package com.bird.vector.common;

import com.bird.vector.utils.Separators;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.tuple.Pair;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Created with IntelliJ IDEA.
 *
 * @description：
 * @author： liuxiangqian
 * @date： 2024/10/17
 */
@Slf4j
public class VectorTools {
    /**
     * 计算两个点之间的欧式距离
     *
     * @param p1 第一个点的坐标
     * @param p2 第二个点的坐标
     * @return
     */
    public static float continentalDistance(float[] p1, float[] p2) {
        float continentalDistance;
        float squareDiff = diffSquare(p1, p2);
        continentalDistance = (float) Math.sqrt(squareDiff);

        return continentalDistance;
    }

    /**
     * 向量之间欧式距离的平方
     *
     * @param p1 向量1
     * @param p2 向量2
     * @return
     */
    public static float diffSquare(float[] p1, float[] p2) {
        assert p1.length == p2.length;

        float squareDiff = 0.0f;
        for (int i = 0; i < p1.length; i++) {
            squareDiff += (p1[i] - p2[i]) * (p1[i] - p2[i]);
        }

        return squareDiff;
    }

    /**
     * 加载数据  数据格式如下：
     * id#value1,value2,....,value1024
     *
     * @param csvFilePath
     */
    public static Pair<List<Integer>, List<float[]>> laodIdsAndVectors(String csvFilePath) {
        List<float[]> vectors = new ArrayList<>();
        List<Integer> ids = new ArrayList<>();
        int lineCount = 0;
        try (BufferedReader bufferedReader = new BufferedReader(new FileReader(new File(csvFilePath)))) {
            String line = null;
            while ((line = bufferedReader.readLine()) != null) {
                if (StringUtils.isAllBlank(line)) {
                    continue;
                }

                String[] idAndVector = line.split(Separators.POUND);
                if (idAndVector.length != 2) {
                    continue;
                }

                int id = Integer.parseInt(idAndVector[0].trim());
                line = idAndVector[1];
                String[] splits = line.split(Separators.COMMA);
                int len = splits.length;
                float[] vector = new float[len];
                for (int i = 0; i < len; i++) {
                    vector[i] = Float.parseFloat(splits[i]);
                }

                ids.add(id);
                vectors.add(vector);

                lineCount++;
                if (lineCount % 10000 == 0) {
                    log.info("当前读取到第{}个向量", lineCount);
                }
            }
        } catch (Exception e) {
            log.error("加载数据文件失败，ERROR:{}", e);
        }

        log.info("数据加载完毕... ...");

        return Pair.of(ids, vectors);
    }
}
