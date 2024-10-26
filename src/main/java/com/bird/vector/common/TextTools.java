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
 * @date： 2024/10/25
 */
@Slf4j
public class TextTools {
    public static Pair<List<Integer>, List<String>> loadIdsAndTexts(String csvFilePath) {
        List<String> texts = new ArrayList<>();
        List<Integer> ids = new ArrayList<>();
        int lineCount = 0;
        try (BufferedReader bufferedReader = new BufferedReader(new FileReader(new File(csvFilePath)))) {
            String line = null;
            bufferedReader.readLine();
            while ((line = bufferedReader.readLine()) != null) {
                if (StringUtils.isAllBlank(line)) {
                    continue;
                }

                String[] idAndText = line.split(Separators.COMMA);
                if (idAndText.length != 2) {
                    continue;
                }

                int id = Integer.parseInt(idAndText[0].trim());
                line = idAndText[1];

                ids.add(id);
                texts.add(line);

                lineCount++;
                if (lineCount % 10000 == 0) {
                    log.info("当前读取到第{}个向量", lineCount);
                }
            }
        } catch (Exception e) {
            log.error("加载数据文件失败，ERROR:{}", e);
        }

        log.info("数据加载完毕... ...");

        return Pair.of(ids, texts);
    }
}
