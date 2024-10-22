package com.bird.vector.utils;

import lombok.extern.slf4j.Slf4j;

import java.lang.reflect.Method;
import java.nio.ByteBuffer;

/**
 * @author caoyunjia
 * NIO删除文件类
 */
@Slf4j
public class UnMmap {
    public UnMmap() {
    }

    public static void unMmap(ByteBuffer bb) {
        //当ByteBuffer不为空,且为直接内存,同时容量大于0时执行关系，大于0的判断为了解决读取空文件会报错的问题
        if (null != bb && bb.isDirect()&&bb.capacity()!=0) {
            try {
                Method cleaner = bb.getClass().getMethod("cleaner");
                cleaner.setAccessible(true);
                Method clean = Class.forName("sun.misc.Cleaner").getMethod("clean");
                clean.setAccessible(true);
                clean.invoke(cleaner.invoke(bb));
            } catch (Exception ex) {
                log.error("UnMmap unMmap ex",ex);
            }
            bb = null;
        }
    }
}
