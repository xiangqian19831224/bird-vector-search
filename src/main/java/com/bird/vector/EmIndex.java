package com.bird.vector;

import com.bird.vector.common.VectorTools;
import com.bird.vector.utils.FolderTools;
import com.bird.vector.utils.MMapTools;
import com.bird.vector.utils.TermSignature;
import com.bird.vector.utils.UnMmap;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import org.roaringbitmap.IntConsumer;
import org.roaringbitmap.RoaringBitmap;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.*;

/**
 * 给向量构建倒排索引
 *
 * @description：
 * @author： liuxiangqian
 * @date： 2024/10/15
 */
@Data
@Slf4j
public class EmIndex {
    /**
     * MAX_THREAD_NUM:      索引创建最大线程数
     * MIN_INDEX_GOODS_NUM: 每个线程索引时候，尽量满足这么多文档数
     * MAX_RECALL:          最大的召回数目，用来减少距离计算的量
     */
    public static final int MAX_MMAP_SIZE = 500 * 1024 * 1024;
    public static final int WARN_MMAP_SIZE = 250 * 1024 * 1024;
    public static final int MAX_RECALL = 100000;

    /**
     * IVT_KEY_FORMAT：  倒排索引的key格式  向量段编号_聚类编号
     */
    private static String IVT_KEY_FORMAT = "%d_%d";

    /**
     * pq:              量化模型
     * termToDocList：  倒排信息
     * delDocs：        由Index传过来的删除文档
     * indexVectors：   参与索引的向量集合
     */
    private EmPQ pq;
    private Map<Long, RoaringBitmap> termToDocList = new HashMap<>(1024);
    private BitSet delDocs;
    private Map<Integer, List<float[]>> indexVectors = new HashMap<>();

    EmIndex(EmPQ pq) {
        this.pq = pq;
        this.delDocs = new BitSet();
    }

    public void addVectors(List<Pair<Integer, float[]>> pairs) {
        pairs.forEach(pair -> {
            addVector(pair.getRight(), pair.getKey());
        });
    }

    public void addVector(List<float[]> vectors, List<Integer> ids) {
        long start = System.currentTimeMillis();
        assert vectors.size() == ids.size();

        for (int i = 0; i < vectors.size(); i++) {
            addVector(vectors.get(i), ids.get(i));

            if (i % 100000 == 0) {
                log.info("当前向量索引数:{}", i);
            }
        }

        long took = (System.currentTimeMillis() - start) / 1000;
        log.info("索引向量数：{} 索引耗时：{}秒", vectors.size(), took);
    }

    /**
     * 向量存储
     * 步骤一： 向量切割      向量段长度，向量段类数
     * 步骤二： 向量段聚类    聚类数
     * 步骤三： 向量索引      基于rb实现
     *
     * @param vector 向量
     * @param id     向量对应的文档id（一个id可以对应多个向量）
     */
    public void addVector(float[] vector, int id) {
        //步骤一： 量化
        List<Integer> pqIds = pq.pq(vector);

        //步骤二： 加入索引
        for (int i = 0; i < pqIds.size(); i++) {
            int segNum = i;
            int clusterId = pqIds.get(i);
            String ivtKey = String.format(IVT_KEY_FORMAT, segNum, clusterId);
            long ivtKeySig = TermSignature.signatureTerm(ivtKey);
            if (!termToDocList.containsKey(ivtKeySig)) {
                termToDocList.put(ivtKeySig, new RoaringBitmap());
            }
            termToDocList.get(ivtKeySig).add(id);
        }

        //步骤三： 收集向量集合
        List<float[]> vectorList = indexVectors.getOrDefault(id, new ArrayList<>(1));
        vectorList.add(vector);
        indexVectors.put(id, vectorList);
    }

    /**
     * 向量查询
     * 步骤一： 向量分段
     * 步骤二： 向量距离表
     * 步骤三： 向量召回
     * 步骤四： 向量距离计算并排序
     *
     * @param vector      查询向量
     * @param clusterTopn 向量每个字段获取的类别数
     * @param topn        获取最相近向量个数
     * @return 获取id列表和最小相似距离(1个id可能有多个向量 ）
     */
    public List<Pair<Integer, Float>> searchDocs(float[] vector, int clusterTopn, int topn) {
        //步骤一： 聚类查询，获取距离表
        List<List<Pair<Integer, Float>>> segDisListList = pq.search(vector, clusterTopn);

        //步骤二： 生成需要的rb
        long start = System.currentTimeMillis();
        RoaringBitmap resultRb = searchRb(segDisListList);
        log.info("召回结果数:{} 召回耗时:{}毫秒", resultRb.getCardinality(), (System.currentTimeMillis() - start));

        //步骤三： 收集最匹配的topn向量编号与距离
        start = System.currentTimeMillis();
        final List<Pair<Integer, Float>> docList = Collections.synchronizedList(new ArrayList<>(topn));
        int count = 0;
        List<Integer> docIds = new ArrayList<>(topn);
        for (int id : resultRb) {
            docIds.add(id);
            if (count++ >= MAX_RECALL) {
                break;
            }
        }
        log.info("收集文档: 文档数{} 耗时{}毫秒", docIds.size(), (System.currentTimeMillis() - start));

        docIds.parallelStream().forEach(id -> {
            float disSum = computeDis(id, vector);
            Pair<Integer, Float> pair = Pair.of(id, disSum);
            docList.add(pair);
        });
        log.info("距离计算耗时:{}毫秒", (System.currentTimeMillis() - start));

        //步骤四： 排序
        docList.sort(Comparator.comparing(Pair::getValue));

        List<Pair<Integer, Float>> returnDocs = docList.size() > topn ? docList.subList(0, topn) : docList;
        return returnDocs;
    }


    /**
     * 获取满足条件的Rb列表
     * 与基于文本的索引进行倒排链求交
     *
     * @param vector
     * @param clusterTopn
     * @return
     */
    public RoaringBitmap searchRb(float[] vector, int clusterTopn) {
        //步骤一： 聚类查询，获取距离表
        List<List<Pair<Integer, Float>>> segDisListList = pq.search(vector, clusterTopn);

        //步骤二： 生成需要的rb
        RoaringBitmap resultRb = searchRb(segDisListList);

        return resultRb;
    }

    /**
     * 对索引进行合并
     *
     * @param ivt
     */
    public void merge(EmIndex ivt) {
        if (this.pq != ivt.getPq()) {
            log.error("量化模型不一致");
        }

        // 倒排索引合并
        Map<Long, RoaringBitmap> rightTermToDocList = ivt.getTermToDocList();
        for (Map.Entry<Long, RoaringBitmap> entry : rightTermToDocList.entrySet()) {
            long ivtKey = entry.getKey();
            RoaringBitmap rightRb = entry.getValue();
            RoaringBitmap leftRb = rightTermToDocList.getOrDefault(ivtKey, new RoaringBitmap());
            leftRb.or(rightRb);
            termToDocList.put(ivtKey, leftRb);
        }

        //删除文档合并
        delDocs.or(ivt.getDelDocs());
        //向量合并
        indexVectors.putAll(ivt.getIndexVectors());
    }

    /**
     * 索引存储
     * 1.清理删除文档的位图索引并存储
     * 2.清理删除文档的向量并存储
     *
     * @param indexDir    索引目录
     * @param indexPrefix 索引前缀（分片，nas盘存储等进行区分用）
     */
    public void store(String indexDir, String indexPrefix) {
        indexDir = FolderTools.folderAppendSlash(indexDir);
        FolderTools.createFolder(indexDir);

        storeBitmapIndex(indexDir, indexPrefix);
        storeVectors(indexDir, indexPrefix);
    }

    /**
     * 加载索引
     * 1.位图索引加载
     * 2.索引向量加载
     *
     * @param modelDir
     * @param indexPrefix
     */
    public void load(String modelDir, String indexPrefix) {
        long start = System.currentTimeMillis();
        modelDir = FolderTools.folderAppendSlash(modelDir);
        termToDocList = loadBitmapIndex(modelDir, indexPrefix);
        indexVectors = loadIndexVectors(modelDir, indexPrefix);
        log.info("向量索引加载耗时:{}ms", System.currentTimeMillis() - start);
    }

    /**
     * @param segDisListList 向量的各个段对应的聚类编号和距离
     * @return 匹配的rb列表
     */
    private RoaringBitmap searchRb(List<List<Pair<Integer, Float>>> segDisListList) {
        RoaringBitmap resultRb = new RoaringBitmap();
        for (int i = 0; i < segDisListList.size(); i++) {
            RoaringBitmap segRb = new RoaringBitmap();

            int segNum = i;
            List<Pair<Integer, Float>> segDisList = segDisListList.get(i);
            for (int j = 0; j < segDisList.size(); j++) {
                Pair<Integer, Float> pair = segDisList.get(j);
                int clusterId = pair.getLeft();
                String key = String.format(IVT_KEY_FORMAT, segNum, clusterId);
                long ivtKey = TermSignature.signatureTerm(key);
                RoaringBitmap tmpRb = termToDocList.getOrDefault(ivtKey, new RoaringBitmap());
                segRb.or(tmpRb);
            }

            if (0 == i) {
                resultRb = segRb;
                continue;
            }

            resultRb.and(segRb);
        }

        return resultRb;
    }

    /**
     * 根据向量集合和检索向量获取文档相关性
     *
     * @param id           召回的文档id
     * @param searchVector 检索向量
     * @return
     */
    private float computeDis(int id, float[] searchVector) {
        List<float[]> curVectors = this.indexVectors.get(id);
        float minDis = Float.MAX_VALUE;
        for (float[] vector : curVectors) {
            float dis = VectorTools.continentalDistance(vector, searchVector);
            minDis = minDis > dis ? dis : minDis;
        }

        return minDis;
    }

    /**
     * 生成位图索引
     */
    private void storeBitmapIndex(String indexDir, String indexPrefix) {
        log.info("storeBitmapIndex begin");
        long startTime = System.currentTimeMillis();
        String termIdFile = indexDir + indexPrefix + ".key";
        String bitmapFile = indexDir + indexPrefix + ".bitmap";
        List<Long> termIdList = new ArrayList<>();
        Triple<RandomAccessFile, FileChannel, MappedByteBuffer> mmapTriple = null;
        long realSize = 0L;
        try (FileOutputStream fos = new FileOutputStream(termIdFile);
             ObjectOutputStream objOutputStream = new ObjectOutputStream(fos)) {
            // 打开一个随机访问文件流，按读写方式
            mmapTriple = fileMmap(bitmapFile, 0);
            long position = 0L;
            MappedByteBuffer mbb = null;
            for (Long termId : termToDocList.keySet()) {
                Pair<Triple<RandomAccessFile, FileChannel, MappedByteBuffer>, Long> mmapPositionPair
                        = mmapFileResize(mmapTriple, bitmapFile, position);
                position = mmapPositionPair.getValue();
                mmapTriple = mmapPositionPair.getKey();
                mbb = mmapTriple.getRight();
                RoaringBitmap rb = termToDocList.get(termId);
                termIdList.add(termId);

                // 将位图索引写入文件
                rb = clearRb(rb);
                byte[] byteArr = serialise(rb);
                mbb.putInt(byteArr.length);
                mbb.put(byteArr);
            }

            // 将termid写入文件
            objOutputStream.writeObject(termIdList);

            if (null == mbb) {
                return;
            }

            FileChannel fc = mmapTriple.getMiddle();
            realSize = MMapTools.mmapFileSize(fc, mbb);
        } catch (IOException e) {
            log.error("storeBitmapIndex error", e);
        } finally {
            if (mmapTriple != null) {
                mmapFileClose(mmapTriple, bitmapFile, realSize);
            }
        }
        long took = (System.currentTimeMillis() - startTime) / 1000;
        log.info("storeBitmapIndex end,总体耗时[{}]s", took);
    }

    /**
     * 存储向量
     *
     * @param indexDir    索引目录
     * @param indexPrefix 索引文件前缀
     */
    private void storeVectors(String indexDir, String indexPrefix) {
        String vectorPath = indexDir + indexPrefix + ".vector";
        try (FileOutputStream fos = new FileOutputStream(vectorPath);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            indexVectors = clearIndexVectors(indexVectors);
            oos.writeObject(indexVectors);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private Map<Long, RoaringBitmap> loadBitmapIndex(String indexDir, String indexPrefix) {
        String bmKeyFile = indexDir + indexPrefix + ".key";
        String bmIndexFile = indexDir + indexPrefix + ".bitmap";
        Map<Long, RoaringBitmap> bmMap = new HashMap<>(2048);
        List<Long> keys = null;
        log.info("start to load bitmap indexControl ..........");
        long start = System.currentTimeMillis();

        //步骤一： 获取位图索引的key信息
        File bmKeyFp = new File(bmKeyFile);
        if (!bmKeyFp.exists()) {
            //说明索引文件为空
            try {
                bmKeyFp.createNewFile();
            } catch (IOException e) {
                log.error("loadBitmapIndex bmKeyFp.createNewFile ex,indexDir为[{}],indexPrefix为[{}]", indexDir,
                        indexPrefix, e);
            }
            log.error("索引文件为空, indexDir:{}", bmKeyFile);
        }
        // 如果文件中无内容就不获取位图信息了
        if (bmKeyFp.length() == 0) {
            return bmMap;
        }

        try (FileInputStream fileInputStream = new FileInputStream(bmKeyFp);
             ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream)) {
            keys = (ArrayList<Long>) objectInputStream.readObject();

            long end = System.currentTimeMillis();
            log.info("load keys, took:{}s", (end - start) / 1000);
        } catch (FileNotFoundException e) {
            log.error("loadBitmapIndex FileNotFoundException,indexDir为[{}],indexPrefix为[{}]", indexDir, indexPrefix, e);
        } catch (IOException e) {
            log.error("loadBitmapIndex IOException,indexDir为[{}],indexPrefix为[{}]", indexDir, indexPrefix, e);
        } catch (ClassNotFoundException e) {
            log.error("loadBitmapIndex IOException,indexDir为[{}],indexPrefix为[{}]", indexDir, indexPrefix, e);
        }

        //步骤二： 获取位图信息
        File file = new File(bmIndexFile);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        long fileLength = file.length();
        MappedByteBuffer mbf = null;
        try (RandomAccessFile raf = new RandomAccessFile(file, "r");
             FileChannel fc = raf.getChannel();) {
            if (fileLength < Integer.MAX_VALUE) {
                mbf = fc.map(FileChannel.MapMode.READ_ONLY, 0, fileLength);
                int i = 0;
                while (mbf.hasRemaining()) {
                    // 解析出 RoaringBitmap
                    int length = mbf.getInt();
                    byte[] bytes = new byte[length];
                    mbf.get(bytes);
                    ByteBuffer byteBuffer = ByteBuffer.wrap(bytes);

                    RoaringBitmap rb = new RoaringBitmap();
                    rb.deserialize(byteBuffer);

                    //移除删除的Docs文档
                    deleteFromRbmByBitSet(rb, delDocs);

                    //获取对应的key
                    long key = keys.get(i);

                    //存储对应的位图索引
                    bmMap.put(key, rb);
                    i++;
                }
            } else {
                //是否最后一次读取
                boolean lastTime = false;
                //每次读取大小
                long size = Integer.MAX_VALUE / 2;
                long mmapStart = 0;
                int i = 0;
                while (true) {
                    //如果文件大小 减 起始位置 小于 批次大小，则是最后一次
                    if (fileLength - mmapStart <= size) {
                        size = fileLength - mmapStart;
                        lastTime = true;
                    }
                    boolean noBreak = true;
                    mbf = fc.map(FileChannel.MapMode.READ_ONLY, mmapStart, size);
                    while (mbf.hasRemaining()) {
                        if (4 + mbf.position() > mbf.limit()) {
                            mmapStart = mmapStart + mbf.position();
                            noBreak = false;
                            break;
                        }
                        // 解析出 RoaringBitmap
                        int length = mbf.getInt();
                        byte[] bytes = new byte[length];
                        if (bytes.length + mbf.position() > mbf.limit()) {
                            mmapStart = mmapStart + mbf.position() - 4;
                            noBreak = false;
                            break;
                        }
                        mbf.get(bytes);
                        ByteBuffer byteBuffer = ByteBuffer.wrap(bytes);

                        RoaringBitmap rb = new RoaringBitmap();
                        rb.deserialize(byteBuffer);

                        //移除删除的Docs文档
                        deleteFromRbmByBitSet(rb, delDocs);

                        //获取对应的key
                        long key = keys.get(i);

                        //存储对应的位图索引
                        bmMap.put(key, rb);
                        i++;
                    }
                    if (lastTime) {
                        break;
                    }
                    if (noBreak) {
                        mmapStart = mmapStart + mbf.position();
                    }
                }
            }
        } catch (IOException e) {
            log.info("Failed to read bitmap indexControl", e);
        } finally {
            UnMmap.unMmap(mbf);
        }

        long end = System.currentTimeMillis();

        log.info("finish loading bitmap indexControl,term数量为[{}],indexDir为[{}],indexPrefix为[{}] took[{}]s",
                bmMap.keySet().size(), indexDir, indexPrefix, (end - start) / 1000);
        return bmMap;
    }

    private Map<Integer, List<float[]>> loadIndexVectors(String indexDir, String indexPrefix) {
        log.info("开始加载向量...");
        long start = System.currentTimeMillis();
        Map<Integer, List<float[]>> indexVectors = null;
        String vectorPath = indexDir + indexPrefix + ".vector";
        try (FileInputStream fis = new FileInputStream(vectorPath);
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            indexVectors = (Map<Integer, List<float[]>>) ois.readObject();
        } catch (Exception e) {
            e.printStackTrace();
        }

        log.info("加载向量耗时：{}s", (System.currentTimeMillis() - start) / 1000);
        return indexVectors;
    }

    /**
     * 将bitmap中需要删除的文档进行删除
     *
     * @param bitmap
     * @param bitSet
     */
    private void deleteFromRbmByBitSet(RoaringBitmap bitmap, BitSet bitSet) {
        RoaringBitmap deleteBitMap = new RoaringBitmap();
        bitmap.forEach((IntConsumer) docId -> {
            if (bitSet.get(docId)) {
                deleteBitMap.add(docId);
            }
        });

        // 相当于将deleteBitMap存在的整数全部删除
        bitmap.xor(deleteBitMap);
    }

    /**
     * 对文件进行内存映射
     *
     * @param filePath 映射文件
     * @param position 映射起始位置
     * @return
     * @throws IOException
     */
    private Triple<RandomAccessFile, FileChannel, MappedByteBuffer> fileMmap(String filePath, long position) throws IOException {
        // 打开一个随机访问文件流，按读写方式
        RandomAccessFile raf = new RandomAccessFile(filePath, "rw");
        FileChannel fc = raf.getChannel();
        MappedByteBuffer mbf = fc.map(FileChannel.MapMode.READ_WRITE, 0, MAX_MMAP_SIZE);
        Triple triple = Triple.of(raf, fc, mbf);

        return triple;
    }

    /**
     * 内存映射映射位置重新调整
     *
     * @param mmapTriple 映射三元组
     * @param filePath   映射的文件
     * @param position   映射的当前位置
     * @return
     * @throws IOException
     */
    private Pair<Triple<RandomAccessFile, FileChannel, MappedByteBuffer>, Long> mmapFileResize(Triple<RandomAccessFile,
            FileChannel, MappedByteBuffer> mmapTriple, String filePath, long position) throws IOException {
        RandomAccessFile raf = mmapTriple.getLeft();
        FileChannel fc = mmapTriple.getMiddle();
        MappedByteBuffer mbb = mmapTriple.getRight();

        if (mbb.remaining() < WARN_MMAP_SIZE) {
            position += mbb.position();
            fc.close();
            MMapTools.unMmap(mbb);

            log.info("position:" + position);
            raf = new RandomAccessFile(filePath, "rw");
            fc = raf.getChannel();
            mbb = fc.map(FileChannel.MapMode.READ_WRITE, position, MAX_MMAP_SIZE);
        }

        mmapTriple = Triple.of(raf, fc, mbb);
        Pair<Triple<RandomAccessFile, FileChannel, MappedByteBuffer>, Long> pair = Pair.of(mmapTriple, position);
        return pair;
    }

    /**
     * 关闭内存映射，并调整映射文件大小
     *
     * @param mmapTriple 映射三元组
     * @param filePath   文件路径
     * @param realSize   文件真实大小
     */
    private void mmapFileClose(Triple<RandomAccessFile, FileChannel, MappedByteBuffer> mmapTriple, String filePath,
                               long realSize) {
        FileChannel fc = mmapTriple.getMiddle();
        MappedByteBuffer mbb = mmapTriple.getRight();
        RandomAccessFile raf = mmapTriple.getLeft();
        try {
            fc.close();
            MMapTools.unMmap(mbb);

            raf = new RandomAccessFile(filePath, "rw");
            fc = raf.getChannel();
            fc.truncate(realSize);
            fc.close();
        } catch (IOException e) {
            log.error("mmapFileClose failed", e);
        } finally {
            try {
                if (fc != null) {
                    fc.close();
                }
                if (raf != null) {
                    raf.close();
                }
            } catch (IOException e) {
                log.error("close stream failed", e);
            }
        }
    }

    private static byte[] serialise(RoaringBitmap input) throws IOException {
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream(input.serializedSizeInBytes());
             DataOutputStream dos = new DataOutputStream(bos)) {
            input.serialize(dos);
            return bos.toByteArray();
        }
    }

    /**
     * 将删除的文档id从rb中清除
     *
     * @param rb
     * @return
     */
    private RoaringBitmap clearRb(RoaringBitmap rb) {
        delDocs.stream().boxed().forEach(delId -> {
            rb.remove(delId);
        });

        return rb;
    }

    /**
     * 将删除的文档id的向量信息中删除
     *
     * @param indexVectors
     * @return
     */
    private Map<Integer, List<float[]>> clearIndexVectors(Map<Integer, List<float[]>> indexVectors) {
        delDocs.stream().boxed().forEach(delId -> {
            indexVectors.remove(delId);
        });

        return indexVectors;
    }
}
