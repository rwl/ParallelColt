package cern.colt.matrix.tdcomplex.impl;

import java.util.ArrayList;
import java.util.Random;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import cern.colt.Timer;
import cern.colt.function.tdcomplex.DComplexProcedure;
import cern.colt.function.tdcomplex.IntIntDComplexFunction;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.BenchmarkMatrixKernel;
import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.jet.math.tdcomplex.DComplex;
import cern.jet.math.tdcomplex.DComplexFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public class BenchmarkDenseDComplexMatrix2D {

    private static Timer t = new Timer();

    private static final String outputFile = "BenchmarkDenseDComplexMatrix2D.txt";

    private static double[][] a_2d, b_2d;

    private static double[] a_1d, b_1d;

    private static double[] noViewTimes;

    private static double[] viewTimes;

    public static void main(String[] args) {
        org.junit.runner.JUnitCore.main("cern.colt.matrix.tdcomplex.impl.BenchmarkDenseDComplexMatrix2D");
    }

    @BeforeClass
    public static void setUpBeforeClass() throws Exception {
        BenchmarkMatrixKernel.readSettings2D();
        Random rand = new Random(0);
        ConcurrencyUtils.setThreadsBeginN_2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]
                * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);

        a_1d = new double[2 * BenchmarkMatrixKernel.MATRIX_SIZE_2D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]];
        a_2d = new double[BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]][2 * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]];
        int idx = 0;
        for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]; r++) {
            for (int c = 0; c < 2 * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]; c++) {
                a_2d[r][c] = rand.nextDouble();
                a_1d[idx++] = a_2d[r][c];
            }
        }

        b_1d = new double[2 * BenchmarkMatrixKernel.MATRIX_SIZE_2D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]];
        b_2d = new double[BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]][2 * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]];
        idx = 0;
        for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]; r++) {
            for (int c = 0; c < 2 * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]; c++) {
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
    public void testAggregateDComplexDComplexDComplexFunctionDComplexDComplexFunction() {
        /* No view */
        DComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] aSum = A.aggregate(DComplexFunctions.plus, DComplexFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.aggregate(DComplexFunctions.plus, DComplexFunctions.square);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] aSum = Av.aggregate(DComplexFunctions.plus, DComplexFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = Av.aggregate(DComplexFunctions.plus, DComplexFunctions.square);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "aggregate(DComplexDComplexDComplexFunction, DComplexDComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAggregateDComplexMatrix2DDComplexDComplexDComplexFunctionDComplexDComplexDComplexFunction() {
        /* No view */
        DComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
        DComplexMatrix2D B = new DenseDComplexMatrix2D(b_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] aSum = A.aggregate(B, DComplexFunctions.plus, DComplexFunctions.mult);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.aggregate(B, DComplexFunctions.plus, DComplexFunctions.mult);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        DComplexMatrix2D Bv = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(b_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] aSum = Av.aggregate(Bv, DComplexFunctions.plus, DComplexFunctions.mult);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = Av.aggregate(Bv, DComplexFunctions.plus, DComplexFunctions.mult);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "aggregate(DComplexMatrix2D, DComplexDComplexDComplexFunction, DComplexDComplexDComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignDoubleDouble() {
        /* No view */
        DComplexMatrix2D A = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        double value = Math.random();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(value, value);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(0, 0);
                t.reset().start();
                A.assign(value, value);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(value, value);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(0, 0);
                t.reset().start();
                Av.assign(value, value);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(double, double)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
    }

    @Test
    public void testAssignDoubleArray() {
        /* No view */
        DComplexMatrix2D A = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(a_1d);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(0, 0);
                t.reset().start();
                A.assign(a_1d);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(a_1d);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(0, 0);
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
        DComplexMatrix2D A = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(a_2d);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(0, 0);
                t.reset().start();
                A.assign(a_2d);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(a_2d);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(0, 0);
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
    public void testAssignDComplexDComplexFunction() {
        /* No view */
        DComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(DComplexFunctions.acos);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.assign(DComplexFunctions.acos);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(DComplexFunctions.acos);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.assign(DComplexFunctions.acos);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DComplexDComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
    }

    @Test
    public void testAssignDComplexMatrix2D() {
        /* No view */
        DComplexMatrix2D A = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        DComplexMatrix2D B = new DenseDComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(B);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(0, 0);
                t.reset().start();
                A.assign(B);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0], BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        DComplexMatrix2D Av = A.viewDice();
        B = new DenseDComplexMatrix2D(a_2d);
        DComplexMatrix2D Bv = B.viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.assign(Bv);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(0, 0);
                t.reset().start();
                Av.assign(Bv);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DComplexMatrix2D)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
    }

    @Test
    public void testAssignDComplexMatrix2DDComplexDComplexDComplexFunction() {
        /* No view */
        DComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
        DComplexMatrix2D B = new DenseDComplexMatrix2D(b_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(B, DComplexFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.assign(B, DComplexFunctions.div);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        DComplexMatrix2D Bv = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(b_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(Bv, DComplexFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.assign(Bv, DComplexFunctions.div);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DComplexMatrix2D, DComplexDComplexDComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignDComplexProcedureDoubleArray() {
        /* No view */
        DComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
        double[] value = new double[] { -1, -2 };
        DComplexProcedure procedure = new DComplexProcedure() {
            public boolean apply(double[] element) {
                if (DComplex.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(procedure, value);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.assign(procedure, value);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(procedure, value);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.assign(procedure, value);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DComplexProcedure, double[])";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignDComplexProcedureDComplexDComplexFunction() {
        /* No view */
        DComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
        DComplexProcedure procedure = new DComplexProcedure() {
            public boolean apply(double[] element) {
                if (DComplex.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(procedure, DComplexFunctions.tan);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.assign(procedure, DComplexFunctions.tan);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(procedure, DComplexFunctions.tan);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.assign(procedure, DComplexFunctions.tan);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DComplexProcedure, DComplexDComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignFloatArray() {
        /* No view */
        DComplexMatrix2D A = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        float[] a_1d_float = new float[2 * BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]
                * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]];
        for (int i = 0; i < 2 * BenchmarkMatrixKernel.MATRIX_SIZE_2D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]; i++) {
            a_1d_float[i] = (float) Math.random();
        }
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(a_1d_float);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(0, 0);
                t.reset().start();
                A.assign(a_1d_float);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(a_1d_float);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(0, 0);
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
        DComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
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
        A = new DenseDComplexMatrix2D(a_2d);
        DComplexMatrix2D Av = A.viewDice();
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
    public void testFft2() {
        /* No view */
        DenseDComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
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
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseDComplexMatrix2D) Av).fft2();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                ((DenseDComplexMatrix2D) Av).fft2();
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
    public void testFftColumns() {
        /* No view */
        DenseDComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.fftColumns();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.fftColumns();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDComplexMatrix2D) Av).fftColumns();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                ((DenseDComplexMatrix2D) Av).fftColumns();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "fftColumns()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testFftRows() {
        /* No view */
        DenseDComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.fftRows();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.fftRows();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDComplexMatrix2D) Av).fftRows();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                ((DenseDComplexMatrix2D) Av).fftRows();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "fftRows()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testForEachNonZero() {
        IntIntDComplexFunction function = new IntIntDComplexFunction() {
            public double[] apply(int first, int second, double[] third) {
                return DComplex.sqrt(third);
            }
        };
        /* No view */
        DComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.forEachNonZero(function);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.forEachNonZero(function);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(a_2d).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.forEachNonZero(function);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.forEachNonZero(function);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "forEachNonZero(IntIntDComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testGetConjugateTranspose() {
        /* No view */
        DComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DComplexMatrix2D B = A.getConjugateTranspose();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.getConjugateTranspose();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DComplexMatrix2D Bv = Av.getConjugateTranspose();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                Bv = Av.getConjugateTranspose();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getConjugateTranspose()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testGetRealPart() {
        /* No view */
        DComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DoubleMatrix2D B = A.getRealPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.getRealPart();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DoubleMatrix2D Bv = Av.getRealPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                Bv = Av.getRealPart();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getRealPart()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testGetImaginaryPart() {
        /* No view */
        DComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DoubleMatrix2D B = A.getImaginaryPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.getImaginaryPart();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DoubleMatrix2D Bv = Av.getImaginaryPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                Bv = Av.getImaginaryPart();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getImaginaryPart()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testGetNonZerosIntArrayListIntArrayListArrayList() {
        /* No view */
        DComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
        IntArrayList rowList = new IntArrayList();
        IntArrayList colList = new IntArrayList();
        ArrayList<double[]> valueList = new ArrayList<double[]>();
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
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(a_2d).viewDice();
        rowList.clear();
        colList.clear();
        valueList.clear();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.getNonZeros(rowList, colList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                rowList.clear();
                colList.clear();
                valueList.clear();
                t.reset().start();
                Av.getNonZeros(rowList, colList, valueList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getNonZeros(IntArrayList, IntArrayList, ArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testIfft2() {
        /* No view */
        DenseDComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
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
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseDComplexMatrix2D) Av).ifft2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                ((DenseDComplexMatrix2D) Av).ifft2(true);
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
    public void testIfftColumns() {
        /* No view */
        DenseDComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.ifftColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.ifftColumns(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDComplexMatrix2D) Av).ifftColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                ((DenseDComplexMatrix2D) Av).ifftColumns(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "ifftColumns(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testIfftRows() {
        /* No view */
        DenseDComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.ifftRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.ifftRows(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseDComplexMatrix2D) Av).ifftRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                ((DenseDComplexMatrix2D) Av).ifftRows(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "ifftRows(true)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testSum() {
        /* No view */
        DComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] aSum = A.zSum();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.zSum();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] aSum = Av.zSum();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = Av.zSum();
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
        DComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
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
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[][] array = Av.toArray();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                array = Av.toArray();
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
        DComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DComplexMatrix1D B = A.vectorize();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.vectorize();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DComplexMatrix1D B = Av.vectorize();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = Av.vectorize();
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
    public void testZMultDComplexMatrix1DDComplexMatrix1DDoubleArrayDoubleArrayBoolean() {
        /* No view */
        DComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
        DComplexMatrix1D y = new DenseDComplexMatrix1D(A.columns());
        double[] alpha = new double[] { 3, 4 };
        double[] beta = new double[] { 5, 6 };
        for (int i = 0; i < y.size(); i++) {
            y.set(i, new double[] { Math.random(), Math.random() });
        }
        DComplexMatrix1D z = new DenseDComplexMatrix1D(A.rows());
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.zMult(y, z, alpha, beta, false);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                z.assign(0, 0);
                t.reset().start();
                A.zMult(y, z, alpha, beta, false);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix2D Av = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.zMult(y, z, alpha, beta, false);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                z.assign(0, 0);
                t.reset().start();
                Av.zMult(y, z, alpha, beta, false);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "zMult(DComplexMatrix1D, DComplexMatrix1D, double[], double[], boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testZMultDComplexMatrix2DDComplexMatrix2DDoubleArrayDoubleArrayBooleanBoolean() {
        int oldNiters = BenchmarkMatrixKernel.NITERS;
        BenchmarkMatrixKernel.NITERS = 10;
        /* No view */
        DComplexMatrix2D A = new DenseDComplexMatrix2D(a_2d);
        DComplexMatrix2D B = new DenseDComplexMatrix2D(b_2d);
        B = B.viewDice().copy();
        DComplexMatrix2D C = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0],
                BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]);
        double[] alpha = new double[] { 3, 4 };
        double[] beta = new double[] { 5, 6 };
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.zMult(B, C, alpha, beta, false, false);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                C.assign(0, 0);
                t.reset().start();
                A.zMult(B, C, alpha, beta, false, false);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseDComplexMatrix2D(a_2d);
        DComplexMatrix2D Av = A.viewDice();
        B = new DenseDComplexMatrix2D(b_2d);
        B = B.viewDice().copy();
        DComplexMatrix2D Bv = B.viewDice();
        C = new DenseDComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        DComplexMatrix2D Cv = C.viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.zMult(Bv, Cv, alpha, beta, false, false);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Cv.assign(0, 0);
                t.reset().start();
                Av.zMult(Bv, Cv, alpha, beta, false, false);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "zMult(DComplexMatrix2D, DComplexMatrix2D, double[], double[], boolean, boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
        BenchmarkMatrixKernel.NITERS = oldNiters;
    }

}
