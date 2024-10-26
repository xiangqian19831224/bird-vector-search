package com.bird.vector;

import com.bird.vector.utils.Separators;
import lombok.extern.slf4j.Slf4j;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

/**
 * Created with IntelliJ IDEA.
 *
 * @description：
 * @author： liuxiangqian
 * @date： 2024/10/15
 */
@Slf4j
public class VectorGen {

    public static void main(String[] args) {
        long start = System.currentTimeMillis();

        String filePath = "data/test.txt";
        File file = new File(filePath);
        if (file.exists()) {
            file.delete();
        }

        int curCount = 0;
        int centerCount = 16;
        int vecCountPerCenter = 10000;
        int vecDim = 1024;
        try (FileWriter fw = new FileWriter(filePath, true);
             BufferedWriter bw = new BufferedWriter(fw)
        ) {
            int id = 0;
            long seed = 2 * 1000;
            Random rd = new Random(seed);
            for (int m = 0; m < centerCount; m++) {
                int base = 100 * m;
                //一个类：生成10,000个向量
                for (int i = 0; i < vecCountPerCenter; i++) {
                    StringBuilder sb = new StringBuilder();
                    sb.append(id).append(Separators.POUND);
                    id++;

                    //生成一个向量
                    for (int j = 0; j < vecDim; j++) {
                        float dim = rd.nextFloat()  + base;
                        sb.append(dim).append(",");
                    }
                    sb.deleteCharAt(sb.length() - 1);
                    sb.append("\n");

                    bw.write(sb.toString());

                    curCount++;
                    if (curCount % 10000 == 0) {
                        log.info("当前生成第{}个向量", curCount);
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        long end = System.currentTimeMillis();
        long took = (end - start) / 60000;
        log.info("生成向量数:{} 生成耗时:{}分钟", centerCount * vecCountPerCenter, took);
    }
}

