package com.bird.vector.utils;

import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * 文件映射常用函数管理工具
 *
 * @description：
 * @author： liuxiangqian
 * @date： 2024/2/29
 */
@Slf4j
public class MMapTools {
    /**
     * 关闭内存映射文件
     *
     * @param raf
     * @param fc
     * @param mbf
     */
    public static void closeMMapFile(RandomAccessFile raf, FileChannel fc, MappedByteBuffer mbf) throws IOException {
        if (raf != null) {
            raf.close();
        }

        if (fc != null) {
            fc.close();
        }

        if (mbf != null) {
            unMmap(mbf);
        }
    }

    /**
     * 计算内存映射文件的真实大小
     *
     * @param fc
     * @param mbf
     * @return
     */
    public static long mmapFileSize(FileChannel fc, MappedByteBuffer mbf) throws IOException {
        long fsize = fc.size();
        long nullSize = (long) mbf.capacity() - mbf.position();
        long realSize = fsize - nullSize;

        return realSize;
    }

    /**
     * 对文件进行truncate到size
     *
     * @param filePath
     * @param size
     */
    public static void truncateFile(String filePath, long size) throws IOException {
        RandomAccessFile raf = new RandomAccessFile(filePath, "rw");
        FileChannel fc = raf.getChannel();
        fc.truncate(size);
        fc.close();
    }

    public static void unMmap(ByteBuffer bb) {
        if (null != bb && bb.isDirect()) {
            try {
                Method cleaner = bb.getClass().getMethod("cleaner");
                cleaner.setAccessible(true);
                Method clean = Class.forName("sun.misc.Cleaner").getMethod("clean");
                clean.setAccessible(true);
                clean.invoke(cleaner.invoke(bb));
            } catch (Exception ex) {
                log.error("UnMmap unMmap ex", ex);
            }
            bb = null;
        }
    }
}
