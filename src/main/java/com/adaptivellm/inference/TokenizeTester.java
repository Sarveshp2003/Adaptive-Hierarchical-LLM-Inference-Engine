package com.adaptivellm.inference;

import com.adaptivellm.runtime.NativeInferenceEngine;

public class TokenizeTester {
    public static void main(String[] args) throws Exception {
        NativeInferenceEngine eng = new NativeInferenceEngine();
        eng.initialize();
        String text = (args.length>0)?String.join(" ", args):"hello";
        System.out.println("Tokenizing: '"+text+"'");
        int[] toks = eng.tokenize(text);
        System.out.println("Got " + toks.length + " tokens:");
        for (int t : toks) System.out.print(t + " ");
        System.out.println();
        eng.shutdown();
    }
}
