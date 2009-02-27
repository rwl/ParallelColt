package cern.colt.matrix.tdouble.impl;

import java.util.Random;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import cern.colt.Timer;
import cern.colt.function.tdouble.DoubleProcedure;
import cern.colt.list.tdouble.DoubleArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.BenchmarkMatrixKernel;
import cern.colt.matrix.tdcomplex.DComplexMatrix3D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix3D;
import cern.jet.math.tdouble.DoubleFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public class BenchmarkDenseDoubleMatrix3D {

    private static Timer t = new Timer();

    private static final String outputFile = "BenchmarkDenseDoubleMatrix3D.txt";

    private static double[][][] a_3d, b_3d;

    private static double[] a_1d, b_1d;

    private static double[] noViewTimes;

    private static double[] viewTimes;

    public static void main(String[] args) {
        org.junit.runner.JUnitCore.main("cern.colt.matrix.tdouble.impl.BenchmarkDenseDoubleMatrix3D");
    }

    @BeforeClass
    public static void setUpBeforeClass() throws Exception {
        BenchmarkMatrixKernel.readSettings3D();
        Random rand = new Random(0);
        ConcurrencyUtils.setThreadsBeginN_3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_3D[1] * BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]);
        a_1d = new double[BenchmarkMatrixKernel.MATRIX_SIZE_3D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_3D[1] * BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]];
        a_3d = new double[BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]][BenchmarkMatrixKernel.MATRIX_SIZE_3D[1]][BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]];
        int idx = 0;
        for (int s = 0; s < BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]; s++) {
            for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_3D[1]; r++) {
                for (int c = 0; c < BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]; c++) {
                    a_3d[s][r][c] = rand.nextDouble();
                    a_1d[idx++] = a_3d[s][r][c];
                }
            }
        }
        b_1d = new double[BenchmarkMatrixKernel.MATRIX_SIZE_3D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_3D[1] * BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]];
        b_3d = new double[BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]][BenchmarkMatrixKernel.MATRIX_SIZE_3D[1]][BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]];
        idx = 0;
        for (int s = 0; s < BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]; s++) {
            for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_3D[1]; r++) {
                for (int c = 0; c < BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]; c++) {
                    b_3d[s][r][c] = rand.nextDouble();
                    b_1d[idx++] = b_3d[s][r][c];
                }
            }
        }
        BenchmarkMatrixKernel.writePropertiesToFile(outputFile, BenchmarkMatrixKernel.MATRIX_SIZE_3D);
        BenchmarkMatrixKernel.displayProperties(BenchmarkMatrixKernel.MATRIX_SIZE_3D);

    }

    @AfterClass
    public static void tearDownAfterClass() throws Exception {
        a_1d = null;
        a_3d = null;
        b_1d = null;
        b_3d = null;
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
        DoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
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
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
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
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

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
        DoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
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
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
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
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAggregateDoubleDoubleFunctionDoubleFunctionIntArrayListIntArrayListIntArrayList() {
        /* No view */
        DoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int s = 0; s < BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]; s++) {
            for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_3D[1]; r++) {
                for (int c = 0; c < BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]; c++) {
                    sliceList.add(s);
                    rowList.add(r);
                    columnList.add(c);
                }
            }
        }
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double aSum = A.aggregate(DoubleFunctions.plus, DoubleFunctions.square, sliceList, rowList, columnList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.aggregate(DoubleFunctions.plus, DoubleFunctions.square, sliceList, rowList, columnList);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double aSum = Av.aggregate(DoubleFunctions.plus, DoubleFunctions.square, sliceList, rowList, columnList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = Av.aggregate(DoubleFunctions.plus, DoubleFunctions.square, sliceList, rowList, columnList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "aggregate(DoubleDoubleFunction, DoubleFunction, IntArrayList, IntArrayList, IntArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAggregateDoubleMatrix3DDoubleDoubleFunctionDoubleDoubleFunction() {
        /* No view */
        DoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        DoubleMatrix3D B = new DenseDoubleMatrix3D(b_3d);
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
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        DoubleMatrix3D Bv = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(b_3d);
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
        String method = "aggregate(DoubleMatrix3D, DoubleDoubleFunction, DoubleDoubleFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignDouble() {
        /* No view */
        DoubleMatrix3D A = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[0], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]);
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
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
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
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }

    @Test
    public void testAssignDoubleArray() {
        /* No view */
        DoubleMatrix3D A = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[0], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]);
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
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
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
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignDoubleArrayArrayArray() {
        /* No view */
        DoubleMatrix3D A = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[0], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(a_3d);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(0);
                t.reset().start();
                A.assign(a_3d);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(a_3d);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(0);
                t.reset().start();
                Av.assign(a_3d);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(double[][][])";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignDoubleFunction() {
        /* No view */
        DoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(DoubleFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.assign(DoubleFunctions.square);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(DoubleFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                Av.assign(DoubleFunctions.square);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DoubleFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }

    @Test
    public void testAssignDoubleMatrix3D() {
        /* No view */
        DoubleMatrix3D A = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[0], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]);
        DoubleMatrix3D B = new DenseDoubleMatrix3D(a_3d);
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
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        DoubleMatrix3D Bv = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(b_3d);
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
        String method = "assign(DoubleMatrix3D)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }

    @Test
    public void testAssignDoubleMatrix3DDoubleDoubleFunction() {
        /* No view */
        DoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        DoubleMatrix3D B = new DenseDoubleMatrix3D(b_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(B, DoubleFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.assign(B, DoubleFunctions.div);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        DoubleMatrix3D Bv = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(b_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(Bv, DoubleFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                Av.assign(Bv, DoubleFunctions.div);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DoubleMatrix3D, DoubleDoubleFuction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignDoubleMatrix3DDoubleDoubleFunctionIntArrayListIntArrayListIntArrayList() {
        /* No view */
        DoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        DoubleMatrix3D B = new DenseDoubleMatrix3D(b_3d);
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int s = 0; s < BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]; s++) {
            for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_3D[1]; r++) {
                for (int c = 0; c < BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]; c++) {
                    sliceList.add(s);
                    rowList.add(r);
                    columnList.add(c);
                }
            }
        }
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(B, DoubleFunctions.div, sliceList, rowList, columnList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.assign(B, DoubleFunctions.div, sliceList, rowList, columnList);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        DoubleMatrix3D Bv = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(b_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(Bv, DoubleFunctions.div, sliceList, rowList, columnList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                Av.assign(Bv, DoubleFunctions.div, sliceList, rowList, columnList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DoubleMatrix3D, DoubleDoubleFuction, IntArrayList, IntArrayList, IntArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignDoubleProcedureDouble() {
        /* No view */
        DoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
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
                A.assign(a_3d);
                t.reset().start();
                A.assign(procedure, -1);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(procedure, -1);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                Av.assign(procedure, -1);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DoubleProcedure, double)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignDoubleProcedureDoubleFunction() {
        /* No view */
        DoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
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
                A.assign(a_3d);
                t.reset().start();
                A.assign(procedure, DoubleFunctions.square);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(procedure, DoubleFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                Av.assign(procedure, DoubleFunctions.square);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DoubleProcedure, DoubleFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testCardinality() {
        /* No view */
        DoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
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
        A = new DenseDoubleMatrix3D(a_3d);
        DoubleMatrix3D Av = A.viewDice(2, 1, 0);
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
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testDct3() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.dct3(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.dct3(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix3D) Av).dct3(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix3D) Av).assign(a_3d);
                t.reset().start();
                ((DenseDoubleMatrix3D) Av).dct3(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dct3(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testDct2Slices() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.dct2Slices(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.dct2Slices(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix3D) Av).dct2Slices(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix3D) Av).assign(a_3d);
                t.reset().start();
                ((DenseDoubleMatrix3D) Av).dct2Slices(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dct2Slices(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testDst3() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.dst3(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.dst3(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix3D) Av).dst3(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix3D) Av).assign(a_3d);
                t.reset().start();
                ((DenseDoubleMatrix3D) Av).dst3(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dst3(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testDst2Slices() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.dst2Slices(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.dst2Slices(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix3D) Av).dst2Slices(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix3D) Av).assign(a_3d);
                t.reset().start();
                ((DenseDoubleMatrix3D) Av).dst2Slices(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dst2Slices(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testDht3() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.dht3();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.dht3();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix3D) Av).dht3();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix3D) Av).assign(a_3d);
                t.reset().start();
                ((DenseDoubleMatrix3D) Av).dht3();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dht3()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testDht2Slices() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.dht2Slices();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.dht2Slices();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix3D) Av).dht2Slices();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix3D) Av).assign(a_3d);
                t.reset().start();
                ((DenseDoubleMatrix3D) Av).dht2Slices();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dht2Slices()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testFft3() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.fft3();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.fft3();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix3D) Av).fft3();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix3D) Av).assign(a_3d);
                t.reset().start();
                ((DenseDoubleMatrix3D) Av).fft3();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "fft3()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetFft3() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        DComplexMatrix3D Ac;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Ac = A.getFft3();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                Ac = A.getFft3();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Ac = ((DenseDoubleMatrix3D) Av).getFft3();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix3D) Av).assign(a_3d);
                t.reset().start();
                Ac = ((DenseDoubleMatrix3D) Av).getFft3();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getFft3()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetFft2Slices() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        DComplexMatrix3D Ac;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Ac = A.getFft2Slices();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                Ac = A.getFft2Slices();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Ac = ((DenseDoubleMatrix3D) Av).getFft2Slices();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix3D) Av).assign(a_3d);
                t.reset().start();
                Ac = ((DenseDoubleMatrix3D) Av).getFft2Slices();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "getFft2Slices()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetIfft3() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        DComplexMatrix3D Ac;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Ac = A.getIfft3(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                Ac = A.getIfft3(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Ac = ((DenseDoubleMatrix3D) Av).getIfft3(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix3D) Av).assign(a_3d);
                t.reset().start();
                Ac = ((DenseDoubleMatrix3D) Av).getIfft3(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getIfft3(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetIfft2Slices() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        DComplexMatrix3D Ac;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Ac = A.getIfft2Slices(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                Ac = A.getIfft2Slices(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Ac = ((DenseDoubleMatrix3D) Av).getIfft2Slices(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix3D) Av).assign(a_3d);
                t.reset().start();
                Ac = ((DenseDoubleMatrix3D) Av).getIfft2Slices(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "getIfft2Slices(true)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetNegativeValuesIntArrayListIntArrayListIntArrayListDoubleArrayList() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        A.assign(DoubleFunctions.mult(-1));
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.getNegativeValues(sliceList, rowList, columnList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                sliceList.clear();
                rowList.clear();
                columnList.clear();
                valueList.clear();
                t.reset().start();
                A.getNegativeValues(sliceList, rowList, columnList, valueList);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        sliceList.clear();
        rowList.clear();
        columnList.clear();
        valueList.clear();
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(a_3d).viewDice(2, 1, 0);
        ((DenseDoubleMatrix3D) Av).assign(DoubleFunctions.mult(-1));
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseDoubleMatrix3D) Av).getNegativeValues(sliceList, rowList, columnList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                sliceList.clear();
                rowList.clear();
                columnList.clear();
                valueList.clear();
                t.reset().start();
                ((DenseDoubleMatrix3D) Av).getNegativeValues(sliceList, rowList, columnList, valueList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getNegativeValues(IntArrayList, IntArrayList, IntArrayList, DoubleArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetNonZerosIntArrayListIntArrayListIntArrayListDoubleArrayList() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList colList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.getNonZeros(sliceList, rowList, colList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                sliceList.clear();
                rowList.clear();
                colList.clear();
                valueList.clear();
                t.reset().start();
                A.getNonZeros(sliceList, rowList, colList, valueList);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        sliceList.clear();
        rowList.clear();
        colList.clear();
        valueList.clear();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix3D) Av).getNonZeros(sliceList, rowList, colList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                sliceList.clear();
                rowList.clear();
                colList.clear();
                valueList.clear();
                t.reset().start();
                ((DenseDoubleMatrix3D) Av).getNonZeros(sliceList, rowList, colList, valueList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getNonZeros(IntArrayList, IntArrayList, IntArrayList, DoubleArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetPositiveValuesIntArrayListIntArrayListIntArrayListDoubleArrayList() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.getPositiveValues(sliceList, rowList, columnList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                sliceList.clear();
                rowList.clear();
                columnList.clear();
                valueList.clear();
                t.reset().start();
                A.getPositiveValues(sliceList, rowList, columnList, valueList);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        sliceList.clear();
        rowList.clear();
        columnList.clear();
        valueList.clear();
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(a_3d).viewDice(2, 1, 0);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseDoubleMatrix3D) Av).getPositiveValues(sliceList, rowList, columnList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                sliceList.clear();
                rowList.clear();
                columnList.clear();
                valueList.clear();
                t.reset().start();
                ((DenseDoubleMatrix3D) Av).getPositiveValues(sliceList, rowList, columnList, valueList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getPositiveValues(IntArrayList, IntArrayList, IntArrayList, DoubleArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIdct3() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.idct3(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.idct3(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix3D) Av).idct3(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix3D) Av).assign(a_3d);
                t.reset().start();
                ((DenseDoubleMatrix3D) Av).idct3(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "idct3(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIdct2Slices() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.idct2Slices(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.idct2Slices(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix3D) Av).idct2Slices(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix3D) Av).assign(a_3d);
                t.reset().start();
                ((DenseDoubleMatrix3D) Av).idct2Slices(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "idct2Slices(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIdst3() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.idst3(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.idst3(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix3D) Av).idst3(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix3D) Av).assign(a_3d);
                t.reset().start();
                ((DenseDoubleMatrix3D) Av).idst3(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "idst3(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIdst2Slices() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.idst2Slices(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.idst2Slices(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix3D) Av).idst2Slices(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix3D) Av).assign(a_3d);
                t.reset().start();
                ((DenseDoubleMatrix3D) Av).idst2Slices(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "idst2Slices(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIdht3() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.idht3(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.idht3(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix3D) Av).idht3(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix3D) Av).assign(a_3d);
                t.reset().start();
                ((DenseDoubleMatrix3D) Av).idht3(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "idht3(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIdht2Slices() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.idht2Slices(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.idht2Slices(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix3D) Av).idht2Slices(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix3D) Av).assign(a_3d);
                t.reset().start();
                ((DenseDoubleMatrix3D) Av).idht2Slices(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "idht2Slices(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIfft3() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.ifft3(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.ifft3(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDoubleMatrix3D) Av).ifft3(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseDoubleMatrix3D) Av).assign(a_3d);
                t.reset().start();
                ((DenseDoubleMatrix3D) Av).ifft3(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "ifft3(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testMaxLocation() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
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
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] maxAndLoc = ((DenseDoubleMatrix3D) Av).getMaxLocation();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                maxAndLoc = ((DenseDoubleMatrix3D) Av).getMaxLocation();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "maxLocation()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testMinLocation() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
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
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] minAndLoc = ((DenseDoubleMatrix3D) Av).getMinLocation();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                minAndLoc = ((DenseDoubleMatrix3D) Av).getMinLocation();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "minLocation()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testSum() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
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
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double aSum = ((DenseDoubleMatrix3D) Av).zSum();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = ((DenseDoubleMatrix3D) Av).zSum();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "zSum()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testToArray() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[][][] array = A.toArray();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                array = A.toArray();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[][][] array = ((DenseDoubleMatrix3D) Av).toArray();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                array = ((DenseDoubleMatrix3D) Av).toArray();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "toArray()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testVectorize() {
        /* No view */
        DenseDoubleMatrix3D A = new DenseDoubleMatrix3D(a_3d);
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
        DoubleMatrix3D Av = new DenseDoubleMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0).assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DoubleMatrix1D B = ((DenseDoubleMatrix3D) Av).vectorize();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = ((DenseDoubleMatrix3D) Av).vectorize();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "vectorize()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }
}
