package com.bird.vector.utils;


import com.google.common.hash.Hashing;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;

public class TermSignature {
    private static final Logger log = LoggerFactory.getLogger(TermSignature.class);

    public TermSignature() {
    }

    public static long signatureTerm(String s) {
        long ix = 0L;

        try {
            ix = Hashing.murmur3_128().hashBytes(s.getBytes(StandardCharsets.UTF_8)).asLong();
        } catch (Exception var4) {
            log.error("TermSignature.signatureTerm error", var4);
        }

        return ix;
    }
}

