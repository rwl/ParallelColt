package cern.colt.matrix.tfloat.impl;

import cern.colt.Timer;
import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfcomplex.FComplexMatrix3D;
import cern.colt.matrix.tfloat.FloatFactory2D;
import cern.colt.matrix.tfloat.FloatFactory3D;

public class BenchmarkFloatFFT {

    //    private static final int[] sizes2D = new int[] { 2000, 2048, 4000, 4096, 8000, 8192 };
    private static final int[] sizes2D = new int[] { 16000, 16384 };
    private static final int[] sizes3D = new int[] { 100, 128, 200, 256, 500, 512 };
    private static int niters = 100;

    public static void benchmarkFft2() {
        Timer t = new Timer();
        double[] times = new double[sizes2D.length];
        for (int i = 0; i < sizes2D.length; i++) {
            DenseFloatMatrix2D A = (DenseFloatMatrix2D) FloatFactory2D.dense.make(sizes2D[i], sizes2D[i]);
            // warm-up
            FComplexMatrix2D Ac = A.getFft2();
            Ac = A.getFft2();
            t.reset().start();
            for (int j = 0; j < niters; j++) {
                Ac = A.getFft2();
            }
            t.stop();
            times[i] = t.millis() / niters;
            int N = sizes2D[i] * sizes2D[i];
            double p = (2.5 * N * log2(N)) / (t.nanos() / niters);
            System.out.println("Average execution time for single precision getFft2() of size " + sizes2D[i] + " x "
                    + sizes2D[i] + " : " + times[i] + " ms");
            System.out.println("Performance of single precision getFft2() of size " + sizes2D[i] + " x " + sizes2D[i]
                    + " : " + String.format("%.4f", p) + " gflops");

        }
    }

    public static double log2(double x) {
        return Math.log10(x) / Math.log10(2);
    }

    public static void benchmarkFft3() {
        Timer t = new Timer();
        double[] times = new double[sizes3D.length];
        for (int i = 0; i < sizes3D.length; i++) {
            DenseFloatMatrix3D A = (DenseFloatMatrix3D) FloatFactory3D.dense.make(sizes3D[i], sizes3D[i], sizes3D[i]);
            // warm-up
            FComplexMatrix3D Ac = A.getFft3();
            Ac = A.getFft3();
            t.reset().start();
            for (int j = 0; j < niters; j++) {
                Ac = A.getFft3();
            }
            t.stop();
            times[i] = t.millis() / niters;
            int N = sizes3D[i] * sizes3D[i] * sizes3D[i];
            double p = (2.5 * N * log2(N)) / (t.nanos() / niters);
            System.out.println("Average execution time for single precision getFft3() of size " + sizes3D[i] + " x "
                    + sizes3D[i] + " x " + sizes3D[i] + " : " + times[i] + " ms");
            System.out.println("Performance of single precision getFft3() of size " + sizes3D[i] + " x " + sizes3D[i]
                    + " x " + sizes3D[i] + " : " + String.format("%.4f", p) + " gflops");

        }
    }

    public static void main(String[] args) {
        benchmarkFft2();
        //        benchmarkFft3();
    }

}
