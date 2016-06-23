package cern.colt.matrix.tdouble.impl;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import cern.colt.Timer;
import cern.colt.matrix.io.MatrixVectorReader;
import cern.colt.matrix.tdouble.DoubleFactory1D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import edu.emory.mathcs.utils.pc.ConcurrencyUtils;

public class BenchmarkSparseRCDoubleMatrix2D {
    private static final Timer t = new Timer();
    private static int niters = 200;
    private static int[] nthreads;

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out
                    .println("Usage: java cern.colt.matrix.tdouble.impl.BenchmarkSparseRCDoubleMatrix2D fileName nthreads1 [nthreads2 ... nthreadsn]");
            System.exit(-1);
        }
        nthreads = new int[args.length - 1];
        for (int i = 1; i < args.length; i++) {
            nthreads[i - 1] = Integer.parseInt(args[i]);
        }
        benchmarkZMult(args[0]);
    }

    public static void benchmarkZMult(String fileName) {
        SparseRCDoubleMatrix2D A = null;
        File file = new File(fileName);
        try {
            A = new SparseDoubleMatrix2D(new MatrixVectorReader(new FileReader(file))).getRowCompressed(false);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Benchmark of SparseRCDoubleMatrix2D (" + file.getName() + ")");
        int rows = A.rows();
        int nnz = A.cardinality();
        DoubleMatrix1D x = DoubleFactory1D.dense.make(rows, 1);

        for (int n = 0; n < nthreads.length; n++) {
            ConcurrencyUtils.setNumberOfThreads(nthreads[n]);
            System.out.println("\tNumber of threads = " + nthreads[n]);
            DoubleMatrix1D y = A.zMult(x, null, 1, 0, false); //warm-up
            y = A.zMult(x, null, 1, 0, false); //warm-up
            t.reset().start();
            for (int i = 0; i < niters; i++) {
                y = A.zMult(x, null, 1, 0, false);
            }
            t.stop();
            double time = t.nanos() / 1000.0 / (double) niters;
            System.out.println("\t\tAverage execution time of mat-vec-mult = " + t.millis() / (double) niters
                    + " milliseconds");
            System.out.println("\t\tPerformance of mat-vec-mult = " + (2 * nnz) / time + " megaFLOPS");
            y = A.zMult(x, null, 1, 0, true); //warm-up
            y = A.zMult(x, null, 1, 0, true); //warm-up
            t.reset().start();
            for (int i = 0; i < niters; i++) {
                y = A.zMult(x, null, 1, 0, true);
            }
            t.stop();
            time = t.nanos() / 1000.0 / (double) niters;
            System.out.println("\t\tAverage execution time of mat-trans-vec-mult = " + t.millis() / (double) niters
                    + " milliseconds");
            System.out.println("\t\tPerformance of mat-trans-vec-mult = " + (2 * nnz) / time + " megaFLOPS");
        }
    }

}
