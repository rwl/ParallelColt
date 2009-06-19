package cern.colt.matrix.tdouble.impl;

import cern.colt.Timer;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.DComplexMatrix3D;
import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.DoubleFactory3D;

public class BenchmarkFFT {

    private static final int[] sizes2D = new int[] { 2000, 2048, 4000, 4096, 8000, 8192 };
    private static final int[] sizes3D = new int[] { 100, 128, 200, 256, 500, 512 };
    private static int niters = 100;

    public static void benchmarkFft2() {
        Timer t = new Timer();
        double[] times = new double[sizes2D.length];
        for (int i = 0; i < sizes2D.length; i++) {
            DenseDoubleMatrix2D A = (DenseDoubleMatrix2D) DoubleFactory2D.dense.random(sizes2D[i], sizes2D[i]);
            // warm-up
            DComplexMatrix2D Ac = A.getFft2();
            Ac = A.getFft2();
            t.reset().start();
            for (int j = 0; j < niters; j++) {
                Ac = A.getFft2();
            }
            t.stop();
            times[i] = t.seconds() / niters;
            System.out.println("Average execution time for getFft2() of size " + sizes2D[i] + " x " + sizes2D[i]
                    + " : " + times[i]);
        }
    }

    public static void benchmarkFft3() {
        Timer t = new Timer();
        double[] times = new double[sizes3D.length];
        for (int i = 0; i < sizes3D.length; i++) {
            DenseDoubleMatrix3D A = (DenseDoubleMatrix3D) DoubleFactory3D.dense.random(sizes3D[i], sizes3D[i],
                    sizes3D[i]);
            // warm-up
            DComplexMatrix3D Ac = A.getFft3();
            Ac = A.getFft3();
            t.reset().start();
            for (int j = 0; j < niters; j++) {
                Ac = A.getFft3();
            }
            t.stop();
            times[i] = t.seconds() / niters;
            System.out.println("Average execution time for getFft3() of size " + sizes3D[i] + " x " + sizes3D[i]
                    + " x " + sizes3D[i] + " : " + times[i]);
        }
    }

    public static void main(String[] args) {
        benchmarkFft2();
        benchmarkFft3();
    }

}
