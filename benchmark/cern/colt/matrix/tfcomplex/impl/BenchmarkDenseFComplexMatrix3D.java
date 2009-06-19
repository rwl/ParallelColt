package cern.colt.matrix.tfcomplex.impl;

import java.util.ArrayList;
import java.util.Random;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import cern.colt.Timer;
import cern.colt.function.tfcomplex.FComplexProcedure;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.BenchmarkMatrixKernel;
import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.FComplexMatrix3D;
import cern.colt.matrix.tfloat.FloatMatrix3D;
import cern.jet.math.tfcomplex.FComplex;
import cern.jet.math.tfcomplex.FComplexFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public class BenchmarkDenseFComplexMatrix3D {

    private static Timer t = new Timer();

    private static final String outputFile = "BenchmarkDenseFComplexMatrix3D.txt";

    private static float[][][] a_3d, b_3d;

    private static float[] a_1d, b_1d;

    private static double[] noViewTimes;

    private static double[] viewTimes;

    public static void main(String[] args) {
        org.junit.runner.JUnitCore.main("cern.colt.matrix.tfcomplex.impl.BenchmarkDenseFComplexMatrix3D");
    }

    @BeforeClass
    public static void setUpBeforeClass() throws Exception {
        BenchmarkMatrixKernel.readSettings3D();
        Random rand = new Random(0);
        ConcurrencyUtils.setThreadsBeginN_3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]
                * BenchmarkMatrixKernel.MATRIX_SIZE_3D[1] * BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]);
        a_1d = new float[2 * BenchmarkMatrixKernel.MATRIX_SIZE_3D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_3D[1]
                * BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]];
        a_3d = new float[BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]][BenchmarkMatrixKernel.MATRIX_SIZE_3D[1]][2 * BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]];
        int idx = 0;
        for (int s = 0; s < BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]; s++) {
            for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_3D[1]; r++) {
                for (int c = 0; c < 2 * BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]; c++) {
                    a_3d[s][r][c] = rand.nextFloat();
                    a_1d[idx++] = a_3d[s][r][c];
                }
            }
        }
        b_1d = new float[2 * BenchmarkMatrixKernel.MATRIX_SIZE_3D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_3D[1]
                * BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]];
        b_3d = new float[BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]][BenchmarkMatrixKernel.MATRIX_SIZE_3D[1]][2 * BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]];
        idx = 0;
        for (int s = 0; s < BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]; s++) {
            for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_3D[1]; r++) {
                for (int c = 0; c < 2 * BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]; c++) {
                    b_3d[s][r][c] = rand.nextFloat();
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
    public void testAggregateFComplexFComplexFComplexFunctionFComplexFComplexFunction() {
        /* No view */
        FComplexMatrix3D A = new DenseFComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[] aSum = A.aggregate(FComplexFunctions.plus, FComplexFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.aggregate(FComplexFunctions.plus, FComplexFunctions.square);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix3D Av = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0)
                .assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[] aSum = Av.aggregate(FComplexFunctions.plus, FComplexFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = Av.aggregate(FComplexFunctions.plus, FComplexFunctions.square);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "aggregate(FComplexFComplexFComplexFunction, FComplexFComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAggregateFComplexMatrix3DFComplexFComplexFComplexFunctionFComplexFComplexFComplexFunction() {
        /* No view */
        FComplexMatrix3D A = new DenseFComplexMatrix3D(a_3d);
        FComplexMatrix3D B = new DenseFComplexMatrix3D(b_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[] aSum = A.aggregate(B, FComplexFunctions.plus, FComplexFunctions.mult);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.aggregate(B, FComplexFunctions.plus, FComplexFunctions.mult);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix3D Av = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0)
                .assign(a_3d);
        FComplexMatrix3D Bv = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0)
                .assign(b_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[] aSum = Av.aggregate(Bv, FComplexFunctions.plus, FComplexFunctions.mult);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = Av.aggregate(Bv, FComplexFunctions.plus, FComplexFunctions.mult);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "aggregate(FComplexMatrix3D, FComplexFComplexFComplexFunction, FComplexFComplexFComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignFloatFloat() {
        /* No view */
        FComplexMatrix3D A = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[0],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]);
        float value = (float) Math.random();
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
        FComplexMatrix3D Av = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0);
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
        String method = "assign(float, float)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
    }

    @Test
    public void testAssignFloatArray() {
        /* No view */
        FComplexMatrix3D A = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[0],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]);
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
        FComplexMatrix3D Av = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0);
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
        String method = "assign(float[])";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignFloatArrayArrayArray() {
        /* No view */
        FComplexMatrix3D A = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[0],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(a_3d);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(0, 0);
                t.reset().start();
                A.assign(a_3d);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix3D Av = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(a_3d);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(0, 0);
                t.reset().start();
                Av.assign(a_3d);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(float[][][])";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignFComplexFComplexFunction() {
        /* No view */
        FComplexMatrix3D A = new DenseFComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(FComplexFunctions.acos);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.assign(FComplexFunctions.acos);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix3D Av = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0)
                .assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(FComplexFunctions.acos);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                Av.assign(FComplexFunctions.acos);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(FComplexFComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
    }

    @Test
    public void testAssignFComplexMatrix3D() {
        /* No view */
        FComplexMatrix3D A = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[0],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]);
        FComplexMatrix3D B = new DenseFComplexMatrix3D(a_3d);
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
        A = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[0], BenchmarkMatrixKernel.MATRIX_SIZE_3D[1],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[2]);
        FComplexMatrix3D Av = A.viewDice(2, 1, 0);
        B = new DenseFComplexMatrix3D(a_3d);
        FComplexMatrix3D Bv = B.viewDice(2, 1, 0);
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
        String method = "assign(FComplexMatrix3D)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);
    }

    @Test
    public void testAssignFComplexMatrix3DFComplexFComplexFComplexFunction() {
        /* No view */
        FComplexMatrix3D A = new DenseFComplexMatrix3D(a_3d);
        FComplexMatrix3D B = new DenseFComplexMatrix3D(b_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(B, FComplexFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.assign(B, FComplexFunctions.div);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix3D Av = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0)
                .assign(a_3d);
        FComplexMatrix3D Bv = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 00)
                .assign(b_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(Bv, FComplexFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                Av.assign(Bv, FComplexFunctions.div);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(FComplexMatrix3D, FComplexFComplexFComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignFComplexProcedureFloatArray() {
        /* No view */
        FComplexMatrix3D A = new DenseFComplexMatrix3D(a_3d);
        float[] value = new float[] { -1, -2 };
        FComplexProcedure procedure = new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 0.1) {
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
                A.assign(a_3d);
                t.reset().start();
                A.assign(procedure, value);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix3D Av = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0)
                .assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(procedure, value);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                Av.assign(procedure, value);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(FComplexProcedure, float[])";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testAssignFComplexProcedureFComplexFComplexFunction() {
        /* No view */
        FComplexMatrix3D A = new DenseFComplexMatrix3D(a_3d);
        FComplexProcedure procedure = new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(procedure, FComplexFunctions.tan);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.assign(procedure, FComplexFunctions.tan);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix3D Av = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0)
                .assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(procedure, FComplexFunctions.tan);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                Av.assign(procedure, FComplexFunctions.tan);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(FComplexProcedure, FComplexFComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testCardinality() {
        /* No view */
        FComplexMatrix3D A = new DenseFComplexMatrix3D(a_3d);
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
        A = new DenseFComplexMatrix3D(a_3d);
        FComplexMatrix3D Av = A.viewDice(2, 1, 0);
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
    public void testFft3() {
        /* No view */
        DenseFComplexMatrix3D A = new DenseFComplexMatrix3D(a_3d);
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
        FComplexMatrix3D Av = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0)
                .assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseFComplexMatrix3D) Av).fft3();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                ((DenseFComplexMatrix3D) Av).fft3();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "fft3()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testFft2Slices() {
        /* No view */
        DenseFComplexMatrix3D A = new DenseFComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.fft2Slices();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.fft2Slices();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        FComplexMatrix3D Av = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0)
                .assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseFComplexMatrix3D) Av).fft2Slices();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                ((DenseFComplexMatrix3D) Av).fft2Slices();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "fft2Slices()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testGetRealPart() {
        /* No view */
        FComplexMatrix3D A = new DenseFComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FloatMatrix3D B = A.getRealPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.getRealPart();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix3D Av = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0)
                .assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FloatMatrix3D Bv = Av.getRealPart();
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
        FComplexMatrix3D A = new DenseFComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FloatMatrix3D B = A.getImaginaryPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.getImaginaryPart();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix3D Av = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0)
                .assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FloatMatrix3D Bv = Av.getImaginaryPart();
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
    public void testGetNonZerosIntArrayListIntArrayListIntArrayListArrayList() {
        /* No view */
        FComplexMatrix3D A = new DenseFComplexMatrix3D(a_3d);
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList colList = new IntArrayList();
        ArrayList<float[]> valueList = new ArrayList<float[]>();
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
        FComplexMatrix3D Av = new DenseFComplexMatrix3D(a_3d).viewDice(2, 1, 0);
        sliceList.clear();
        rowList.clear();
        colList.clear();
        valueList.clear();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.getNonZeros(sliceList, rowList, colList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                sliceList.clear();
                rowList.clear();
                colList.clear();
                valueList.clear();
                t.reset().start();
                Av.getNonZeros(sliceList, rowList, colList, valueList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getNonZeros(IntArrayList, IntArrayList, IntArrayList, ArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testIfft3() {
        /* No view */
        DenseFComplexMatrix3D A = new DenseFComplexMatrix3D(a_3d);
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
        FComplexMatrix3D Av = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0)
                .assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseFComplexMatrix3D) Av).ifft3(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                ((DenseFComplexMatrix3D) Av).ifft3(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "ifft3(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testIfft2Slices() {
        /* No view */
        DenseFComplexMatrix3D A = new DenseFComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            A.ifft2Slices(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_3d);
                t.reset().start();
                A.ifft2Slices(true);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        /* View */
        FComplexMatrix3D Av = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0)
                .assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            ((DenseFComplexMatrix3D) Av).ifft2Slices(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_3d);
                t.reset().start();
                ((DenseFComplexMatrix3D) Av).ifft2Slices(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "ifft2Slices(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS,
                noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes,
                viewTimes);

    }

    @Test
    public void testSum() {
        /* No view */
        FComplexMatrix3D A = new DenseFComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[] aSum = A.zSum();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.zSum();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix3D Av = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0)
                .assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[] aSum = Av.zSum();
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
        FComplexMatrix3D A = new DenseFComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[][][] array = A.toArray();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                array = A.toArray();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix3D Av = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0)
                .assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[][][] array = Av.toArray();
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
        FComplexMatrix3D A = new DenseFComplexMatrix3D(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FComplexMatrix1D B = A.vectorize();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.vectorize();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix3D Av = new DenseFComplexMatrix3D(BenchmarkMatrixKernel.MATRIX_SIZE_3D[2],
                BenchmarkMatrixKernel.MATRIX_SIZE_3D[1], BenchmarkMatrixKernel.MATRIX_SIZE_3D[0]).viewDice(2, 1, 0)
                .assign(a_3d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FComplexMatrix1D B = Av.vectorize();
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
}
