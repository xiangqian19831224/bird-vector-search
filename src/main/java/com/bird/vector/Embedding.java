package com.bird.vector;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.bird.vector.utils.FolderTools;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.nio.LongBuffer;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Created with IntelliJ IDEA.
 *
 * @description：
 * @author： liuxiangqian
 * @date： 2024/10/25
 */
@Slf4j
public class Embedding {
    /**
     * ONNX_MODDEL_NAME： onnx格式模型名称
     * TOKENIZER_NAME：  分词模型名称
     * poolingMethod：   ？
     * env:                 环境变量
     */
    private static String ONNX_MODDEL_NAME = "flag_model.onnx";
    private static String TOKENIZER_NAME = "tokenizer.json";
    private static String poolingMethod = "cls";
    private static OrtEnvironment env = OrtEnvironment.getEnvironment();
    /**
     * session：  编码会话
     * tokenizer：分词器
     * normalizeEmbeddings： 向量化结果是否规范化（平方和为1）
     */
    private OrtSession session;
    private HuggingFaceTokenizer tokenizer;
    private boolean normalizeEmbeddings = true;

    /**
     * 创建嵌入模型
     *
     * @param modelDir 模型目录
     */
    public Embedding(String modelDir, boolean normalizeEmbeddings) {
        this.normalizeEmbeddings = normalizeEmbeddings;

        modelDir = FolderTools.folderAppendSlash(modelDir);
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        try {
            String modelPath = modelDir + ONNX_MODDEL_NAME;
            this.session = env.createSession(modelPath, opts);

            String tokenizerPath = modelDir + TOKENIZER_NAME;
            tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(tokenizerPath));
        } catch (OrtException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * 对输入文档进行向量化
     *
     * @param text 向量化文档
     * @return
     * @throws OrtException
     */
    public float[] encode(String text) throws OrtException {
        Encoding enc = tokenizer.encode(text);
        long[] inputIdsData = enc.getIds();
        long[] attentionMaskData = enc.getAttentionMask();

        int maxLength = 128;
        int batchSize = 1;

        long[] inputIdsShape = new long[]{batchSize, maxLength};
        long[] attentionMaskShape = new long[]{batchSize, maxLength};
        // 确保数组长度为128
        inputIdsData = padArray(inputIdsData, maxLength);
        attentionMaskData = padArray(attentionMaskData, maxLength);

        OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIdsData), inputIdsShape);
        OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMaskData),
                attentionMaskShape);

        // 创建输入的Map
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_ids", inputIdsTensor);
        inputs.put("attention_mask", attentionMaskTensor);

        // 运行推理
        OrtSession.Result result = session.run(inputs);

        // 提取三维数组
        float[][][] lastHiddenState = (float[][][]) result.get(0).getValue();

        float[] embeddings;
        if ("cls".equals(poolingMethod)) {
            embeddings = lastHiddenState[0][0];
        } else if ("mean".equals(poolingMethod)) {
            int sequenceLength = lastHiddenState[0].length;
            int hiddenSize = lastHiddenState[0][0].length;
            float[] sum = new float[hiddenSize];
            int count = 0;

            for (int i = 0; i < sequenceLength; i++) {
                if (attentionMaskData[i] == 1) {
                    for (int j = 0; j < hiddenSize; j++) {
                        sum[j] += lastHiddenState[0][i][j];
                    }
                    count++;
                }
            }

            float[] mean = new float[hiddenSize];
            for (int j = 0; j < hiddenSize; j++) {
                mean[j] = sum[j] / count;
            }
            embeddings = mean;
        } else {
            throw new IllegalArgumentException("Unsupported pooling method: " + poolingMethod);
        }

        if (normalizeEmbeddings) {
            float norm = 0;
            for (float v : embeddings) {
                norm += v * v;
            }
            norm = (float) Math.sqrt(norm);
            for (int i = 0; i < embeddings.length; i++) {
                embeddings[i] /= Math.sqrt(norm);
            }
        }

        // 释放资源
        inputIdsTensor.close();
        attentionMaskTensor.close();

        return embeddings;
    }

    private long[] padArray(long[] array, int length) {
        long[] paddedArray = new long[length];
        System.arraycopy(array, 0, paddedArray, 0, Math.min(array.length, length));
        return paddedArray;
    }

    public static void main(String[] args) throws OrtException {
        String modelDir = "bgemodel/bge-base-zh-v1.5/";
        Embedding embedding = new Embedding(modelDir, false);

        String[] texts = {
                "中华人民共和国是一个伟大的国度",
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
            String vectorStr = Arrays.toString(vector);
            log.info(vectorStr);
        }
    }

}
