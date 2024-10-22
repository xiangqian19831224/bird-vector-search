package com.bird.vector;

/**
 * Created with IntelliJ IDEA.
 *
 * @description：
 * @author： liuxiangqian
 * @date： 2024/10/15
 */
public class KmeansTest {
    public static void main(String[] args) {
        String csvFile = "data/test.txt";
        Kmeans kmeans = new Kmeans(16, 100, 1024 );
        kmeans.train(csvFile);
        kmeans.show();
    }
}
