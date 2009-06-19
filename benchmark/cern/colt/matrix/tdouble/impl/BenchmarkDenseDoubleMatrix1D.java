package cern.colt.matrix.tdouble.impl;

import java.util.Random;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import cern.colt.Timer;
import cern.colt.function.tdouble.DoubleFunction;
import cern.colt.function.tdouble.DoubleProcedure;
import cern.colt.list.tdouble.DoubleArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.BenchmarkMatrixKernel;
import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix3D;
import cern.jet.math.tdouble.DoubleFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public class BenchmarkDenseDoubleMatrix1D {

    private static final Timer t = new Timer();

    private static final String outputFile = "BenchmarkDenseDoubleMatrix1D.txt";

    public static void main(String[] args) {
        org.junit.runner.JUnitCore.main("cern.colt.matrix.tdouble.impl.BenchmarkDenseDoubleMatrix1D");
    }

    private static double[] noViewTimes;

    private static double[] viewTimes;

    private static double[] a, b;

    private static final long millis = 5000;

    @BeforeClass
    public static void setUpBeforeClass() throws Exception {
        BenchmarkMatrixKernel.readSettings1D();
        Random rand = new Random(0);
        ConcurrencyUtils.setThreadsBeginN_1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);

        a = new double[BenchmarkMatrixKernel.MATRIX_SIZE_1D];
        for (int i = 0; i < BenchmarkMatrixKernel.MATRIX_SIZE_1D; i++) {
            a[i] = rand.nextDouble();
        }
        b = new double[BenchmarkMatrixKernel.MATRIX_SIZE_1D];
        for (int i = 0; i < BenchmarkMatrixKernel.MATRIX_SIZE_1D; i++) {
            b[i] = rand.nextDouble();
        }
        BenchmarkMatrixKernel.writePropertiesToFile(outputFile, new int[] { BenchmarkMatrixKernel.MATRIX_SIZE_1D });
        BenchmarkMatrixKernel.displayProperties(new int[] { BenchmarkMatrixKernel.MATRIX_SIZE_1D });
    }

    @AfterClass
    public static void tearDownAfterClass() throws Exception {
        a = null;
        b = null;
        ConcurrencyUtils.resetThreadsBeginN();
        System.gc();
    }

    @Before
    public void setUpBefore() {
        noViewTimes = new double[BenchmarkMatrixKernel.NTHREADS.length];
        viewTimes = new double[BenchmarkMatrixKernel.NTHREADS.length];
    }

    @Test
    public void testAggregateDoubleDoubleFunctionDoubleFunction() {
        /* No view */
        DoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            double aSum = A.aggregate(DoubleFunctions.plus, DoubleFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.aggregate(DoubleFunctions.plus, DoubleFunctions.square);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
            ConcurrencyUtils.sleep(millis);
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            double aSum = Av.aggregate(DoubleFunctions.plus, DoubleFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = Av.aggregate(DoubleFunctions.plus, DoubleFunctions.square);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
            ConcurrencyUtils.sleep(millis);
        }
        String method = "aggregate(DoubleDoubleFunction, DoubleFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
    }

    @Test
    public void testAssignDoubleMatrix1D() {
        /* No view */
        DoubleMatrix1D A = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);
        DoubleMatrix1D B = new DenseDoubleMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(B);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(0);
                t.reset().start();
                A.assign(B);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);
        DoubleMatrix1D Av = A.viewFlip();
        B = new DenseDoubleMatrix1D(a);
        DoubleMatrix1D Bv = B.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.assign(Bv);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(0);
                t.reset().start();
                Av.assign(Bv);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DoubleMatrix1D)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testZDotProductDoubleMatrix1D() {
        /* No view */
        DoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        DoubleMatrix1D B = new DenseDoubleMatrix1D(b);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double product = A.zDotProduct(B);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                product = A.zDotProduct(B);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseDoubleMatrix1D(a);
        DoubleMatrix1D Av = A.viewFlip();
        B = new DenseDoubleMatrix1D(b);
        DoubleMatrix1D Bv = B.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            double product = Av.zDotProduct(Bv);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                product = Av.zDotProduct(Bv);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "zDotProduct(DoubleMatrix1D, int, int)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignDoubleFunction() {
        DoubleFunction f = DoubleFunctions.mult(2.5);
        /* No view */
        DoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.assign(f);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                A.assign(f);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(f);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a);
                t.reset().start();
                Av.assign(f);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DoubleFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignDoubleMatrix1DDoubleDoubleFunction() {
        /* No view */
        DoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        DoubleMatrix1D B = new DenseDoubleMatrix1D(b);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(B, DoubleFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                A.assign(B, DoubleFunctions.div);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        DoubleMatrix1D Bv = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(b);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(Bv, DoubleFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a);
                t.reset().start();
                Av.assign(Bv, DoubleFunctions.div);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DoubleMatrix1D, DoubleDoubleFuction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAggregateDoubleMatrix1DDoubleDoubleFunctionDoubleDoubleFunction() {
        /* No view */
        DoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        DoubleMatrix1D B = new DenseDoubleMatrix1D(b);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double aSum = A.aggregate(B, DoubleFunctions.plus, DoubleFunctions.mult);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.aggregate(B, DoubleFunctions.plus, DoubleFunctions.mult);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        DoubleMatrix1D Bv = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(b);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double aSum = Av.aggregate(Bv, DoubleFunctions.plus, DoubleFunctions.mult);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = Av.aggregate(Bv, DoubleFunctions.plus, DoubleFunctions.mult);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "aggregate(DoubleMatrix1D, DoubleDoubleFunction, DoubleDoubleFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignDouble() {
        /* No view */
        DoubleMatrix1D A = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);
        double value = Math.random();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(value);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(0);
                t.reset().start();
                A.assign(value);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(value);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(0);
                t.reset().start();
                Av.assign(value);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(double)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignDoubleArray() {
        /* No view */
        DoubleMatrix1D A = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(a);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(0);
                t.reset().start();
                A.assign(a);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(a);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(0);
                t.reset().start();
                Av.assign(a);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(double[])";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignDoubleProcedureDouble() {
        /* No view */
        DoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        DoubleProcedure procedure = new DoubleProcedure() {
            public boolean apply(double element) {
                if (Math.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.assign(procedure, -1);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                A.assign(procedure, -1);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.assign(procedure, -1);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a);
                t.reset().start();
                Av.assign(procedure, -1);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DoubleProcedure, double)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignDoubleProcedureDoubleFunction() {
        /* No view */
        DoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        DoubleProcedure procedure = new DoubleProcedure() {
            public boolean apply(double element) {
                if (Math.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(procedure, DoubleFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                A.assign(procedure, DoubleFunctions.square);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(procedure, DoubleFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a);
                t.reset().start();
                Av.assign(procedure, DoubleFunctions.square);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DoubleProcedure, DoubleFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testCardinality() {
        /* No view */
        DoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            int card = A.cardinality();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                card = A.cardinality();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseDoubleMatrix1D(a);
        DoubleMatrix1D Av = A.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            int card = Av.cardinality();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                card = Av.cardinality();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "cardinality()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testFft() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            A.fft();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                A.fft();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            ((DenseDoubleMatrix1D) Av).fft();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix1D) Av).assign(a);
                t.reset().start();
                ((DenseDoubleMatrix1D) Av).fft();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "fft()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testDct() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            A.dct(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                A.dct(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            ((DenseDoubleMatrix1D) Av).dct(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix1D) Av).assign(a);
                t.reset().start();
                ((DenseDoubleMatrix1D) Av).dct(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dct(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
    }

    @Test
    public void testDht() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            A.dht();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                A.dht();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            ((DenseDoubleMatrix1D) Av).dht();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix1D) Av).assign(a);
                t.reset().start();
                ((DenseDoubleMatrix1D) Av).dht();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dht()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
    }

    @Test
    public void testDst() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            A.dst(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                A.dst(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            ((DenseDoubleMatrix1D) Av).dst(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix1D) Av).assign(a);
                t.reset().start();
                ((DenseDoubleMatrix1D) Av).dst(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dst(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testGetFft() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        DComplexMatrix1D Ac;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            Ac = A.getFft();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                Ac = A.getFft();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            Ac = ((DenseDoubleMatrix1D) Av).getFft();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix1D) Av).assign(a);
                t.reset().start();
                Ac = ((DenseDoubleMatrix1D) Av).getFft();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getFft()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testGetIfft() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        DComplexMatrix1D Ac;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            Ac = A.getIfft(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                Ac = A.getIfft(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            Ac = ((DenseDoubleMatrix1D) Av).getIfft(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix1D) Av).assign(a);
                t.reset().start();
                Ac = ((DenseDoubleMatrix1D) Av).getIfft(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getIfft(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
    }

    @Test
    public void testGetPositiveValuesIntArrayListDoubleArrayList() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        IntArrayList indexList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.getPositiveValues(indexList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                indexList.clear();
                valueList.clear();
                t.reset().start();
                A.getPositiveValues(indexList, valueList);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(a).viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseDoubleMatrix1D) Av).getPositiveValues(indexList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                indexList.clear();
                valueList.clear();
                t.reset().start();
                ((DenseDoubleMatrix1D) Av).getPositiveValues(indexList, valueList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getPositiveValues(IntArrayList, DoubleArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testGetNegativeValuesIntArrayListDoubleArrayList() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        A.assign(DoubleFunctions.mult(-1));
        IntArrayList indexList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.getNegativeValues(indexList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                indexList.clear();
                valueList.clear();
                t.reset().start();
                A.getNegativeValues(indexList, valueList);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(a).viewFlip();
        ((DenseDoubleMatrix1D) Av).assign(DoubleFunctions.mult(-1));
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseDoubleMatrix1D) Av).getNegativeValues(indexList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                indexList.clear();
                valueList.clear();
                t.reset().start();
                ((DenseDoubleMatrix1D) Av).getNegativeValues(indexList, valueList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getNegativeValues(IntArrayList, DoubleArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testIdct() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            A.idct(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                A.idct(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            ((DenseDoubleMatrix1D) Av).idct(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix1D) Av).assign(a);
                t.reset().start();
                ((DenseDoubleMatrix1D) Av).idct(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "idct(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
    }

    @Test
    public void testIdht() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            A.idht(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                A.idht(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            ((DenseDoubleMatrix1D) Av).idht(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix1D) Av).assign(a);
                t.reset().start();
                ((DenseDoubleMatrix1D) Av).idht(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "idht(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
    }

    @Test
    public void testIdst() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            A.idst(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                A.idst(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            ((DenseDoubleMatrix1D) Av).idst(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix1D) Av).assign(a);
                t.reset().start();
                ((DenseDoubleMatrix1D) Av).idst(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "idst(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testIfft() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            A.ifft(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                A.ifft(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            ConcurrencyUtils.setThreadsBeginN_1D(1);
            switch (BenchmarkMatrixKernel.NTHREADS[i]) {
            case 1:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            case 2:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
                break;
            default:
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(1);
                ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(1);
                break;
            }
            // warm-up
            ((DenseDoubleMatrix1D) Av).ifft(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix1D) Av).assign(a);
                t.reset().start();
                ((DenseDoubleMatrix1D) Av).ifft(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "ifft(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testMaxLocation() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] maxAndLoc = A.getMaxLocation();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                maxAndLoc = A.getMaxLocation();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(a).viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] maxAndLoc = ((DenseDoubleMatrix1D) Av).getMaxLocation();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                maxAndLoc = ((DenseDoubleMatrix1D) Av).getMaxLocation();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "maxLocation()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testMinLocation() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] minAndLoc = A.getMinLocation();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                minAndLoc = A.getMinLocation();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(a).viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] minAndLoc = ((DenseDoubleMatrix1D) Av).getMinLocation();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                minAndLoc = ((DenseDoubleMatrix1D) Av).getMinLocation();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "minLocation()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
    }

    @Test
    public void testReshapeIntInt() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        int rows = BenchmarkMatrixKernel.MATRIX_SIZE_1D / 64;
        int cols = 64;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DoubleMatrix2D B = A.reshape(rows, cols);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.reshape(rows, cols);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseDoubleMatrix1D(a);
        DoubleMatrix1D Av = A.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DoubleMatrix2D B = ((DenseDoubleMatrix1D) Av).reshape(rows, cols);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = ((DenseDoubleMatrix1D) Av).reshape(rows, cols);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "reshape(int, int)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testReshapeIntIntInt() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        int slices = BenchmarkMatrixKernel.MATRIX_SIZE_1D / 64;
        int rows = 16;
        int cols = 4;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DoubleMatrix3D B = A.reshape(slices, rows, cols);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.reshape(slices, rows, cols);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseDoubleMatrix1D(a);
        DoubleMatrix1D Av = A.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DoubleMatrix3D B = ((DenseDoubleMatrix1D) Av).reshape(slices, rows, cols);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = ((DenseDoubleMatrix1D) Av).reshape(slices, rows, cols);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "reshape(int, int, int)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testSwapDoubleMatrix1D() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        DoubleMatrix1D B = new DenseDoubleMatrix1D(b);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.swap(B);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                A.swap(B);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseDoubleMatrix1D(a);
        DoubleMatrix1D Av = A.viewFlip();
        B = new DenseDoubleMatrix1D(b);
        DoubleMatrix1D Bv = B.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseDoubleMatrix1D) Av).swap(Bv);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                ((DenseDoubleMatrix1D) Av).swap(Bv);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "swap(DoubleMatrix1D)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testToArrayDoubleArray() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] array = A.toArray();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                array = A.toArray();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] array = ((DenseDoubleMatrix1D) Av).toArray();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                array = ((DenseDoubleMatrix1D) Av).toArray();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "toArray()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testZDotProductDoubleMatrix1DIntInt() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        DoubleMatrix1D B = new DenseDoubleMatrix1D(b);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double product = A.zDotProduct(B, 5, (int) B.size() - 10);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                product = A.zDotProduct(B, 5, (int) B.size() - 10);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseDoubleMatrix1D(a);
        DoubleMatrix1D Av = A.viewFlip();
        B = new DenseDoubleMatrix1D(b);
        DoubleMatrix1D Bv = B.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            double product = ((DenseDoubleMatrix1D) Av).zDotProduct(Bv, 5, (int) Bv.size() - 10);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                product = ((DenseDoubleMatrix1D) Av).zDotProduct(Bv, 5, (int) Bv.size() - 10);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "zDotProduct(DoubleMatrix1D, int, int)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testZSum() {
        /* No view */
        DenseDoubleMatrix1D A = new DenseDoubleMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double aSum = A.zSum();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.zSum();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix1D Av = new DenseDoubleMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double aSum = ((DenseDoubleMatrix1D) Av).zSum();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = ((DenseDoubleMatrix1D) Av).zSum();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "zSum()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

}
