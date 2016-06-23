package cern.colt.matrix.tdouble.algo;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import cern.colt.Timer;
import cern.colt.matrix.io.MatrixVectorReader;
import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.decomposition.SparseDoubleLUDecomposition;
import cern.colt.matrix.tdouble.impl.SparseCCDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import edu.emory.mathcs.utils.pc.ConcurrencyUtils;

public class SparseDoubleAlgebraBenchmark {
    private static final Timer t = new Timer();
    private static int niters = 10;
    private static int[] nthreads;

    public static void main(String[] args) {
        benchmarkLU();
        //        if (args.length < 2) {
        //            System.out
        //                    .println("Usage: java cern.colt.matrix.tdouble.algo.SparseDoubleAlgebraBenchmark fileName nthreads1 [nthreads2 ... nthreadsn]");
        //            System.exit(-1);
        //        }
        //        nthreads = new int[args.length - 1];
        //        for (int i = 1; i < args.length; i++) {
        //            nthreads[i - 1] = Integer.parseInt(args[i]);
        //        }
        //        benchmarkLU(args[0]);
    }

    public static void benchmarkLU(String fileName) {
        SparseCCDoubleMatrix2D A = null;
        File file = new File(fileName);
        try {
            A = new SparseDoubleMatrix2D(new MatrixVectorReader(new FileReader(file))).getColumnCompressed(false);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Benchmark of sparse LU. Matrix: (" + file.getName() + ")");
        SparseDoubleAlgebra alg = SparseDoubleAlgebra.DEFAULT;

        for (int n = 0; n < nthreads.length; n++) {
            ConcurrencyUtils.setNumberOfThreads(nthreads[n]);
            System.out.println("\tNumber of threads = " + nthreads[n]);
            SparseDoubleLUDecomposition lu = alg.lu(A, 0);//warm-up            
            t.reset().start();
            for (int i = 0; i < niters; i++) {
                lu = alg.lu(A, 0);
            }
            t.stop();
            System.out.println("\t\tAverage execution time of LU = " + t.seconds() / (double) niters + " seconds");
        }
    }

    public static void benchmarkLU() {
        DoubleMatrix2D A = DoubleFactory2D.sparse.random(1000, 1000);
        A = ((SparseDoubleMatrix2D) A).getColumnCompressed(false);
        System.out.println("Benchmark of sparse LU.");
        SparseDoubleAlgebra alg = SparseDoubleAlgebra.DEFAULT;
        ConcurrencyUtils.setNumberOfThreads(1);
        SparseDoubleLUDecomposition lu = alg.lu(A, 0);//warm-up            
        t.reset().start();
        for (int i = 0; i < niters; i++) {
            lu = alg.lu(A, 0);
        }
        t.stop();
        System.out.println("\t\tAverage execution time of LU = " + t.seconds() / (double) niters + " seconds");
    }
}
