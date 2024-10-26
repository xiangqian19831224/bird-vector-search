package com.bird.vector;

/**
 * Created with IntelliJ IDEA.
 *
 * @description：
 * @author： liuxiangqian
 * @date： 2024/10/25
 */

import ai.onnxruntime.OrtException;
import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;

@Slf4j
public class EmbeddingTest {
    public static void main(String[] args) throws OrtException {
        String modelDir = "bgemodel/bge-base-zh-v1.5/";
        Embedding embedding = new Embedding(modelDir, false);

        String[] texts = {
                "This is a sample text for embedding calculation.",
                "Another example text to test the model.",
                "Yet another text to ensure consistency.",
                "Testing with different lengths and contents.",
                "Final text to verify the ONNX model accuracy.",
                "中文数据测试",
                "随便说点什么吧！反正也只是测试用！",
                "你好！jone！"
        };

        for (String text : texts) {
            float[] vector = embedding.encode(text);
            log.info("dim:{}", vector.length);
            String vectorStr = Arrays.toString(vector);
            log.info(vectorStr);
        }
    }
}