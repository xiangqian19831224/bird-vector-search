package com.bird.vector.utils;

/**
 * Created with IntelliJ IDEA.
 *
 * @description：
 * @author： liuxiangqian
 * @date： 2024/10/22
 */
public class ArrayTools {
    /**
     * 将double数组，转化为float数组
     *
     * @param doubles
     * @return
     */
    public static float[] toFloatArr(double[] doubles) {
        assert null != doubles;
        float[] floats = new float[doubles.length];
        for (int i = 0; i < doubles.length; i++) {
            floats[i] = (float) doubles[i];
        }

        return floats;
    }
}
