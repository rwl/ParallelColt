package cern.colt.matrix.tdouble.impl;

import java.util.Random;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import cern.colt.Timer;
import cern.colt.function.tdouble.DoubleProcedure;
import cern.colt.function.tdouble.IntIntDoubleFunction;
import cern.colt.list.tdouble.DoubleArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.BenchmarkMatrixKernel;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.jet.math.tdouble.DoubleFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public class BenchmarkDenseDoubleMatrix2D {

    private static Timer t = new Timer();

    private static final String outputFile = "BenchmarkDenseDoubleMatrix2D.txt";

    private static double[][] a_2d, b_2d;

    private static double[] a_1d, b_1d;

    private static double[] noViewTimes;

    private static double[] viewTimes;

    public static void main(String[] args) {
        org.junit.runner.JUnitCore.main("cern.colt.matrix.tdouble.impl.BenchmarkDenseDoubleMatrix2D");
    }

    @BeforeClass
    public static void setUpBeforeClass() throws Exception {
        BenchmarkMatrixKernel.readSettings2D();
        Random rand = new Random(0);
        ConcurrencyUtils.setThreadsBeginN_2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]
                * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);

        a_1d = new double[BenchmarkMatrixKernel.MATRIX_SIZE_2D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]];
        a_2d = new double[BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]][BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]];
        int idx = 0;
        for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]; r++) {
            for (int c = 0; c < BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]; c++) {
                a_2d[r][c] = rand.nextDouble();
                a_1d[idx++] = a_2d[r][c];
            }
        }

        b_1d = new double[BenchmarkMatrixKernel.MATRIX_SIZE_2D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]];
        b_2d = new double[BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]][BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]];
        idx = 0;
        for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]; r++) {
            for (int c = 0; c < BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]; c++) {
                b_2d[r][c] = rand.nextDouble();
                b_1d[idx++] = b_2d[r][c];
            }
        }
        BenchmarkMatrixKernel.writePropertiesToFile(outputFile, BenchmarkMatrixKernel.MATRIX_SIZE_2D);
        BenchmarkMatrixKernel.displayProperties(BenchmarkMatrixKernel.MATRIX_SIZE_2D);
    }

    @AfterClass
    public static void tearDownAfterClass() throws Exception {
        a_1d = null;
        a_2d = null;
        b_1d = null;
        b_2d = null;
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
        DoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
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
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
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
        }
        String method = "aggregate(DoubleDoubleFunction, DoubleFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAggregateDoubleDoubleFunctionDoubleFunctionDoubleProcedure() {
        DoubleProcedure procedure = new DoubleProcedure() {
            public boolean apply(double element) {
                if (Math.abs(element) > 0.2) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        /* No view */
        DoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double aSum = A.aggregate(DoubleFunctions.plus, DoubleFunctions.square, procedure);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.aggregate(DoubleFunctions.plus, DoubleFunctions.square, procedure);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double aSum = Av.aggregate(DoubleFunctions.plus, DoubleFunctions.square, procedure);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = Av.aggregate(DoubleFunctions.plus, DoubleFunctions.square, procedure);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "aggregate(DoubleDoubleFunction, DoubleFunction, DoubleProcedure)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAggregateDoubleDoubleFunctionDoubleFunctionIntArrayListIntArrayList() {
        /* No view */
        DoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]; r++) {
            for (int c = 0; c < BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]; c++) {
                rowList.add(r);
                columnList.add(c);
            }
        }
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double aSum = A.aggregate(DoubleFunctions.plus, DoubleFunctions.square, rowList, columnList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.aggregate(DoubleFunctions.plus, DoubleFunctions.square, rowList, columnList);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double aSum = Av.aggregate(DoubleFunctions.plus, DoubleFunctions.square, rowList, columnList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = Av.aggregate(DoubleFunctions.plus, DoubleFunctions.square, rowList, columnList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "aggregate(DoubleDoubleFunction, DoubleFunction, IntArrayList, IntArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAggregateDoubleMatrix2DDoubleDoubleFunctionDoubleDoubleFunction() {
        /* No view */
        DoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        DoubleMatrix2D B = new DenseDoubleMatrix2D(b_2d);
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
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        DoubleMatrix2D Bv = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(b_2d);
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
        String method = "aggregate(DoubleMatrix2D, DoubleDoubleFunction, DoubleDoubleFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignDouble() {
        /* No view */
        DoubleMatrix2D A = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
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
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice();
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
        DoubleMatrix2D A = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(a_1d);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(0);
                t.reset().start();
                A.assign(a_1d);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(a_1d);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(0);
                t.reset().start();
                Av.assign(a_1d);
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
    public void testAssignDoubleArrayArray() {
        /* No view */
        DoubleMatrix2D A = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(a_2d);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(0);
                t.reset().start();
                A.assign(a_2d);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(a_2d);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(0);
                t.reset().start();
                Av.assign(a_2d);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(double[][])";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignDoubleFunction() {
        /* No view */
        DoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(DoubleFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.assign(DoubleFunctions.square);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(DoubleFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.assign(DoubleFunctions.square);
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
    public void testAssignDoubleMatrix2D() {
        /* No view */
        DoubleMatrix2D A = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        DoubleMatrix2D B = new DenseDoubleMatrix2D(a_2d);
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
        A = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0], BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        DoubleMatrix2D Av = A.viewDice();
        B = new DenseDoubleMatrix2D(a_2d);
        DoubleMatrix2D Bv = B.viewDice();
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
        String method = "assign(DoubleMatrix2D)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
    }

    @Test
    public void testAssignDoubleMatrix2DDoubleDoubleFunction() {
        /* No view */
        DoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        DoubleMatrix2D B = new DenseDoubleMatrix2D(b_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(B, DoubleFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.assign(B, DoubleFunctions.div);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        DoubleMatrix2D Bv = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(b_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(Bv, DoubleFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.assign(Bv, DoubleFunctions.div);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DoubleMatrix2D, DoubleDoubleFuction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignDoubleMatrix2DDoubleDoubleFunctionIntArrayListIntArrayList() {
        /* No view */
        DoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        DoubleMatrix2D B = new DenseDoubleMatrix2D(b_2d);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]; r++) {
            for (int c = 0; c < BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]; c++) {
                rowList.add(r);
                columnList.add(c);
            }
        }
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(B, DoubleFunctions.div, rowList, columnList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.assign(B, DoubleFunctions.div, rowList, columnList);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        DoubleMatrix2D Bv = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(b_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(Bv, DoubleFunctions.div, rowList, columnList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.assign(Bv, DoubleFunctions.div, rowList, columnList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DoubleMatrix2D, DoubleDoubleFuction, IntArrayList, IntArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignDoubleProcedureDouble() {
        /* No view */
        DoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
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
                A.assign(a_2d);
                t.reset().start();
                A.assign(procedure, -1);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(procedure, -1);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
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
        DoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
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
                A.assign(a_2d);
                t.reset().start();
                A.assign(procedure, DoubleFunctions.square);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(procedure, DoubleFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
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
    public void testAssignFloatArray() {
        /* No view */
        DoubleMatrix2D A = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        float[] a_1d_float = new float[BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]
                * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]];
        for (int i = 0; i < BenchmarkMatrixKernel.MATRIX_SIZE_2D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]; i++) {
            a_1d_float[i] = (float) Math.random();
        }
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(a_1d_float);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(0);
                t.reset().start();
                A.assign(a_1d_float);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(a_1d_float);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(0);
                t.reset().start();
                Av.assign(a_1d_float);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(float[])";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testCardinality() {
        /* No view */
        DoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
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
        A = new DenseDoubleMatrix2D(a_2d);
        DoubleMatrix2D Av = A.viewDice();
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
    public void testDct2() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.dct2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.dct2(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix2D) Av).dct2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).dct2(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dct2(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testDctColumns() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.dctColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.dctColumns(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix2D) Av).dctColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).dctColumns(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dctColumns(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testDctRows() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.dctRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.dctRows(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix2D) Av).dctRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).dctRows(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dctRows(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
    }

    @Test
    public void testDht2() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.dct2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.dct2(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix2D) Av).dht2();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).dht2();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dht2()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testDhtColumns() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.dhtColumns();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.dhtColumns();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix2D) Av).dhtColumns();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).dhtColumns();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dhtColumns()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testDhtRows() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.dhtRows();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.dhtRows();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix2D) Av).dhtRows();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).dhtRows();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dhtRows()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
    }

    @Test
    public void testDst2() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.dst2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.dst2(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseDoubleMatrix2D) Av).dst2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).dst2(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dst2(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testDstColumns() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.dstColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.dstColumns(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix2D) Av).dstColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).dstColumns(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dstColumns(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testDstRows() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.dstRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.dstRows(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix2D) Av).dstRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).dstRows(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "dstRows(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testFft2() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.fft2();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.fft2();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseDoubleMatrix2D) Av).fft2();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).fft2();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "fft2()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testForEachNonZero() {
        IntIntDoubleFunction function = new IntIntDoubleFunction() {
            public double apply(int first, int second, double third) {
                return Math.sqrt(third);
            }
        };
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.forEachNonZero(function);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                A.forEachNonZero(function);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(a_2d).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseDoubleMatrix2D) Av).forEachNonZero(function);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).forEachNonZero(function);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "forEachNonZero(IntIntDoubleFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testGetFft2() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        DComplexMatrix2D Ac;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Ac = A.getFft2();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                Ac = A.getFft2();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Ac = ((DenseDoubleMatrix2D) Av).getFft2();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                Ac = ((DenseDoubleMatrix2D) Av).getFft2();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getFft2()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testGetFftColumns() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        DComplexMatrix2D Ac;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Ac = A.getFftColumns();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                Ac = A.getFftColumns();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Ac = ((DenseDoubleMatrix2D) Av).getFftColumns();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                Ac = ((DenseDoubleMatrix2D) Av).getFftColumns();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "getFftColumns()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testGetFftRows() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        DComplexMatrix2D Ac;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Ac = A.getFftRows();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                Ac = A.getFftRows();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Ac = ((DenseDoubleMatrix2D) Av).getFftRows();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                Ac = ((DenseDoubleMatrix2D) Av).getFftRows();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "getFftRows()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testGetIfft2() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.ifft2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.ifft2(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseDoubleMatrix2D) Av).ifft2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).ifft2(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getIfft2(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testGetIfftColumns() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        DComplexMatrix2D Ac;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Ac = A.getIfftColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                Ac = A.getIfftColumns(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Ac = ((DenseDoubleMatrix2D) Av).getIfftColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                Ac = ((DenseDoubleMatrix2D) Av).getIfftColumns(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "getIfftColumns(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testGetIfftRows() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        DComplexMatrix2D Ac;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Ac = A.getIfftRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                Ac = A.getIfftRows(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Ac = ((DenseDoubleMatrix2D) Av).getIfftRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                Ac = ((DenseDoubleMatrix2D) Av).getIfftRows(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "getIfftRows(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testGetNonZerosIntArrayListIntArrayListDoubleArrayList() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        IntArrayList rowList = new IntArrayList();
        IntArrayList colList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.getNonZeros(rowList, colList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                rowList.clear();
                colList.clear();
                valueList.clear();
                t.reset().start();
                A.getNonZeros(rowList, colList, valueList);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(a_2d).viewDice();
        rowList.clear();
        colList.clear();
        valueList.clear();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix2D) Av).getNonZeros(rowList, colList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                rowList.clear();
                colList.clear();
                valueList.clear();
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).getNonZeros(rowList, colList, valueList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getNonZeros(IntArrayList, IntArrayList, DoubleArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testGetPositiveValuesIntArrayListIntArrayListDoubleArrayList() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.getPositiveValues(rowList, columnList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                rowList.clear();
                columnList.clear();
                valueList.clear();
                t.reset().start();
                A.getPositiveValues(rowList, columnList, valueList);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(a_2d).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseDoubleMatrix2D) Av).getPositiveValues(rowList, columnList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                rowList.clear();
                columnList.clear();
                valueList.clear();
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).getPositiveValues(rowList, columnList, valueList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getPositiveValues(IntArrayList, IntArrayList, DoubleArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testGetNegativeValuesIntArrayListIntArrayListDoubleArrayList() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        A.assign(DoubleFunctions.mult(-1));
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.getNegativeValues(rowList, columnList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                rowList.clear();
                columnList.clear();
                valueList.clear();
                t.reset().start();
                A.getNegativeValues(rowList, columnList, valueList);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(a_2d).viewDice();
        ((DenseDoubleMatrix2D) Av).assign(DoubleFunctions.mult(-1));
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseDoubleMatrix2D) Av).getNegativeValues(rowList, columnList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                rowList.clear();
                columnList.clear();
                valueList.clear();
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).getNegativeValues(rowList, columnList, valueList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getNegativeValues(IntArrayList, IntArrayList, DoubleArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testIdct2() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.idct2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.idct2(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseDoubleMatrix2D) Av).idct2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).idct2(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "idct2(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testIdctColumns() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.idctColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.idctColumns(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix2D) Av).idctColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).idctColumns(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "idctColumns(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testIdctRows() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.idctRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.idctRows(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix2D) Av).idctRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).idctRows(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "idctRows(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testIdht2() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.idht2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.idht2(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseDoubleMatrix2D) Av).idht2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).idht2(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "idht2(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testIdhtColumns() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.idhtColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.idhtColumns(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix2D) Av).idhtColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).idhtColumns(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "idhtColumns(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testIdhtRows() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.idhtRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.idhtRows(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix2D) Av).idhtRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).idhtRows(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "idhtRows(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testIdst2() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.idst2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.idst2(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseDoubleMatrix2D) Av).idst2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).idst2(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "idst2(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testIdstColumns() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.idstColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.idstColumns(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix2D) Av).idstColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).idstColumns(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "idstColumns(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
    }

    @Test
    public void testIdstRows() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.idstRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.idstRows(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix2D) Av).idstRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).idstRows(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "idstRows(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testIfft2() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.ifft2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.ifft2(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseDoubleMatrix2D) Av).ifft2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix2D) Av).assign(a_2d);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).ifft2(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "ifft2(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testMaxLocation() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
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
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(a_2d).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] maxAndLoc = ((DenseDoubleMatrix2D) Av).getMaxLocation();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                maxAndLoc = ((DenseDoubleMatrix2D) Av).getMaxLocation();
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
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
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
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(a_2d).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] minAndLoc = ((DenseDoubleMatrix2D) Av).getMinLocation();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                minAndLoc = ((DenseDoubleMatrix2D) Av).getMinLocation();
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
    public void testSum() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
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
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double aSum = ((DenseDoubleMatrix2D) Av).zSum();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = ((DenseDoubleMatrix2D) Av).zSum();
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

    @Test
    public void testToArray() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[][] array = A.toArray();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                array = A.toArray();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[][] array = ((DenseDoubleMatrix2D) Av).toArray();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                array = ((DenseDoubleMatrix2D) Av).toArray();
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
    public void testVectorize() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DoubleMatrix1D B = A.vectorize();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.vectorize();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DoubleMatrix1D B = ((DenseDoubleMatrix2D) Av).vectorize();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = ((DenseDoubleMatrix2D) Av).vectorize();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "vectorize()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testZMultDoubleMatrix1DDoubleMatrix1DDoubleDoubleBoolean() {
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        DoubleMatrix1D y = new DenseDoubleMatrix1D(A.columns());
        double alpha = 3;
        double beta = 5;
        for (int i = 0; i < y.size(); i++) {
            y.set(i, Math.random());
        }
        DoubleMatrix1D z = new DenseDoubleMatrix1D(A.rows());
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.zMult(y, z, alpha, beta, false);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                z.assign(0);
                t.reset().start();
                A.zMult(y, z, alpha, beta, false);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix2D) Av).zMult(y, z, alpha, beta, false);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                z.assign(0);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).zMult(y, z, alpha, beta, false);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "zMult(DoubleMatrix1D, DoubleMatrix1D, double, double, boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testZMultDoubleMatrix2DDoubleMatrix2DDoubleDoubleBooleanBoolean() {
        int oldNiters = BenchmarkMatrixKernel.NITERS;
        BenchmarkMatrixKernel.NITERS = 10;
        /* No view */
        DenseDoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        DoubleMatrix2D B = new DenseDoubleMatrix2D(b_2d);
        B = B.viewDice().copy();
        DoubleMatrix2D C = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]);
        double alpha = 3;
        double beta = 5;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.zMult(B, C, alpha, beta, false, false);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                C.assign(0);
                t.reset().start();
                A.zMult(B, C, alpha, beta, false, false);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseDoubleMatrix2D(a_2d);
        DoubleMatrix2D Av = A.viewDice();
        B = new DenseDoubleMatrix2D(b_2d);
        B = B.viewDice().copy();
        DoubleMatrix2D Bv = B.viewDice();
        C = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        DoubleMatrix2D Cv = C.viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix2D) Av).zMult(Bv, Cv, alpha, beta, false, false);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Cv.assign(0);
                t.reset().start();
                ((DenseDoubleMatrix2D) Av).zMult(Bv, Cv, alpha, beta, false, false);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "zMult(DoubleMatrix2D, DoubleMatrix2D, double, double, boolean, boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
        BenchmarkMatrixKernel.NITERS = oldNiters;
    }
    
    @Test
    public void testZMultDoubleMatrix2DDoubleMatrix2D() {
        int oldNiters = BenchmarkMatrixKernel.NITERS;
        BenchmarkMatrixKernel.NITERS = 10;
        /* No view */
        DoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        DoubleMatrix2D B = new DenseDoubleMatrix2D(b_2d);
        B = B.viewDice().copy();
        DoubleMatrix2D C = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.zMult(B, C);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                C.assign(0);
                t.reset().start();
                A.zMult(B, C);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseDoubleMatrix2D(a_2d);
        DoubleMatrix2D Av = A.viewDice();
        B = new DenseDoubleMatrix2D(b_2d);
        B = B.viewDice().copy();
        DoubleMatrix2D Bv = B.viewDice();
        C = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        DoubleMatrix2D Cv = C.viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.zMult(Bv, Cv);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Cv.assign(0);
                t.reset().start();
                Av.zMult(Bv, Cv);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "zMult(DoubleMatrix2D, DoubleMatrix2D)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
        BenchmarkMatrixKernel.NITERS = oldNiters;
    }

    @Test
    public void testZMultDoubleMatrix1DDoubleMatrix1D() {
        /* No view */
        DoubleMatrix2D A = new DenseDoubleMatrix2D(a_2d);
        DoubleMatrix1D y = new DenseDoubleMatrix1D(A.columns());
        for (int i = 0; i < y.size(); i++) {
            y.set(i, Math.random());
        }
        DoubleMatrix1D z = new DenseDoubleMatrix1D(A.rows());
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.zMult(y, z);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                z.assign(0);
                t.reset().start();
                A.zMult(y, z);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix2D Av = new DenseDoubleMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.zMult(y, z);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                z.assign(0);
                t.reset().start();
                Av.zMult(y, z);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "zMult(DoubleMatrix1D, DoubleMatrix1D)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

}
