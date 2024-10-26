package com.bird.vector.utils;

import java.io.File;

/**
 * Created with IntelliJ IDEA.
 *
 * @description：
 * @author： liuxiangqian
 * @date： 2024/7/21
 */
public class FolderTools {
    /**
     * 对目录路径进行标准化，去除空格和添加结果的斜杠
     *
     * @param dirPath 目录路径
     * @return
     */
    public static String folderAppendSlash(String dirPath) {
        dirPath = dirPath.trim();

        if (dirPath.endsWith(Separators.SLASH)) {
            return dirPath;
        }

        dirPath += Separators.SLASH;
        return dirPath;
    }

    public static void createFolder(String folder) {
        File file = new File(folder);
        if (!file.exists()) {
            file.mkdirs();
        }
    }

}
