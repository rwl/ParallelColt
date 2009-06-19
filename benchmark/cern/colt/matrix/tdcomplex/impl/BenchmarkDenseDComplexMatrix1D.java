package cern.colt.matrix.tdcomplex.impl;

import java.util.Random;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import cern.colt.Timer;
import cern.colt.function.tdcomplex.DComplexProcedure;
import cern.colt.matrix.BenchmarkMatrixKernel;
import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.DComplexMatrix3D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.jet.math.tdcomplex.DComplex;
import cern.jet.math.tdcomplex.DComplexFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public class BenchmarkDenseDComplexMatrix1D {

    private static final Timer t = new Timer();

    private static final String outputFile = "BenchmarkDenseDComplexMatrix1D.txt";

    public static void main(String[] args) {
        org.junit.runner.JUnitCore.main("cern.colt.matrix.tdcomplex.impl.BenchmarkDenseDComplexMatrix1D");
    }

    private static double[] noViewTimes;

    private static double[] viewTimes;

    private static double[] a, b;

    @BeforeClass
    public static void setUpBeforeClass() throws Exception {
        BenchmarkMatrixKernel.readSettings1D();
        Random rand = new Random(0);
        ConcurrencyUtils.setThreadsBeginN_1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);

        a = new double[2 * BenchmarkMatrixKernel.MATRIX_SIZE_1D];
        for (int i = 0; i < 2 * BenchmarkMatrixKernel.MATRIX_SIZE_1D; i++) {
            a[i] = rand.nextDouble();
        }
        b = new double[2 * BenchmarkMatrixKernel.MATRIX_SIZE_1D];
        for (int i = 0; i < 2 * BenchmarkMatrixKernel.MATRIX_SIZE_1D; i++) {
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
    public void testAggregateDComplexDComplexDComplexFunctionDComplexDComplexFunction() {
        /* No view */
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
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
        DComplexMatrix1D Av = new DenseDComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
    public void testAggregateDComplexMatrix1DDComplexDComplexDComplexFunctionDComplexDComplexDComplexFunction() {
        /* No view */
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D B = new DenseDComplexMatrix1D(b);
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
        DComplexMatrix1D Av = new DenseDComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        DComplexMatrix1D Bv = new DenseDComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(b);
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
        String method = "aggregate(DComplexMatrix1D, DComplexDComplexDComplexFunction, DComplexDComplexDComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignDoubleDouble() {
        /* No view */
        DComplexMatrix1D A = new DenseDComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);
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
        DComplexMatrix1D Av = new DenseDComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip();
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
        DComplexMatrix1D A = new DenseDComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(a);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(0, 0);
                t.reset().start();
                A.assign(a);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix1D Av = new DenseDComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(a);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(0, 0);
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
    public void testAssignDComplexDComplexFunction() {
        /* No view */
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(DComplexFunctions.acos);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                A.assign(DComplexFunctions.acos);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix1D Av = new DenseDComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(DComplexFunctions.acos);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a);
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
    public void testAssignDComplexMatrix1D() {
        /* No view */
        DComplexMatrix1D A = new DenseDComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);
        DComplexMatrix1D B = new DenseDComplexMatrix1D(a);
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
        A = new DenseDComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);
        DComplexMatrix1D Av = A.viewFlip();
        B = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Bv = B.viewFlip();
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
        String method = "assign(DComplexMatrix1D)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignDComplexMatrix1DDComplexDComplexDComplexFunction() {
        /* No view */
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D B = new DenseDComplexMatrix1D(b);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(B, DComplexFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                A.assign(B, DComplexFunctions.div);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix1D Av = new DenseDComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        DComplexMatrix1D Bv = new DenseDComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(b);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(Bv, DComplexFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a);
                t.reset().start();
                Av.assign(Bv, DComplexFunctions.div);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(DComplexMatrix1D, DComplexDComplexDComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignDComplexProcedureDoubleArray() {
        /* No view */
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
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
                A.assign(a);
                t.reset().start();
                A.assign(procedure, value);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix1D Av = new DenseDComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.assign(procedure, value);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a);
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
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
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
                A.assign(a);
                t.reset().start();
                A.assign(procedure, DComplexFunctions.tan);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        DComplexMatrix1D Av = new DenseDComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(procedure, DComplexFunctions.tan);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a);
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
    public void testCardinality() {
        /* No view */
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
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
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
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
    public void testGetImaginaryPart() {
        /* No view */
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        int slices = BenchmarkMatrixKernel.MATRIX_SIZE_1D / 64;
        int rows = 16;
        int cols = 4;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DoubleMatrix1D B = A.getImaginaryPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.getImaginaryPart();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DoubleMatrix1D B = Av.getImaginaryPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = Av.getImaginaryPart();
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
    public void testGetRealPart() {
        /* No view */
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        int slices = BenchmarkMatrixKernel.MATRIX_SIZE_1D / 64;
        int rows = 16;
        int cols = 4;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DoubleMatrix1D B = A.getRealPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.getRealPart();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DoubleMatrix1D B = Av.getRealPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = Av.getRealPart();
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
    public void testFft() {
        /* No view */
        DenseDComplexMatrix1D A = new DenseDComplexMatrix1D(a);
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
        DComplexMatrix1D Av = new DenseDComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
            ((DenseDComplexMatrix1D) Av).fft();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a);
                t.reset().start();
                ((DenseDComplexMatrix1D) Av).fft();
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
    public void testIfft() {
        /* No view */
        DenseDComplexMatrix1D A = new DenseDComplexMatrix1D(a);
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
        DComplexMatrix1D Av = new DenseDComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
            ((DenseDComplexMatrix1D) Av).ifft(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a);
                t.reset().start();
                ((DenseDComplexMatrix1D) Av).ifft(true);
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
    public void testReshapeIntInt() {
        /* No view */
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        int rows = BenchmarkMatrixKernel.MATRIX_SIZE_1D / 64;
        int cols = 64;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DComplexMatrix2D B = A.reshape(rows, cols);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.reshape(rows, cols);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DComplexMatrix2D B = Av.reshape(rows, cols);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = Av.reshape(rows, cols);
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
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        int slices = BenchmarkMatrixKernel.MATRIX_SIZE_1D / 64;
        int rows = 16;
        int cols = 4;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DComplexMatrix3D B = A.reshape(slices, rows, cols);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.reshape(slices, rows, cols);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            DComplexMatrix3D B = Av.reshape(slices, rows, cols);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = Av.reshape(slices, rows, cols);
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
    public void testSwapDComplexMatrix1D() {
        /* No view */
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D B = new DenseDComplexMatrix1D(b);
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
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        B = new DenseDComplexMatrix1D(b);
        DComplexMatrix1D Bv = B.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.swap(Bv);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                Av.swap(Bv);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "swap(DComplexMatrix1D)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testToArrayDoubleArray() {
        /* No view */
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
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
        DComplexMatrix1D Av = new DenseDComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] array = Av.toArray();
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
    public void testZDotProductDComplexMatrix1DIntInt() {
        /* No view */
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D B = new DenseDComplexMatrix1D(b);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            double[] product = A.zDotProduct(B, 5, (int) B.size() - 10);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                product = A.zDotProduct(B, 5, (int) B.size() - 10);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        B = new DenseDComplexMatrix1D(b);
        DComplexMatrix1D Bv = B.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            double[] product = Av.zDotProduct(Bv, 5, (int) Bv.size() - 10);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                product = Av.zDotProduct(Bv, 5, (int) Bv.size() - 10);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "zDotProduct(DComplexMatrix1D, int, int)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testZSum() {
        /* No view */
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
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
        DComplexMatrix1D Av = new DenseDComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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

}
