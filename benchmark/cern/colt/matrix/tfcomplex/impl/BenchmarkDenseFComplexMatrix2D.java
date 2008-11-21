package cern.colt.matrix.tfcomplex.impl;

import java.util.ArrayList;
import java.util.Random;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import cern.colt.Timer;
import cern.colt.function.tfcomplex.FComplexProcedure;
import cern.colt.function.tfcomplex.IntIntFComplexFunction;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.BenchmarkMatrixKernel;
import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix1D;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.jet.math.tfcomplex.FComplex;
import cern.jet.math.tfcomplex.FComplexFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public class BenchmarkDenseFComplexMatrix2D {

    private static Timer t = new Timer();

    private static final String outputFile = "BenchmarkDenseFComplexMatrix2D.txt";

    private static float[][] a_2d, b_2d;

    private static float[] a_1d, b_1d;

    private static double[] noViewTimes;

    private static double[] viewTimes;

    public static void main(String[] args) {
        org.junit.runner.JUnitCore.main("cern.colt.matrix.impl.BenchmarkDenseFComplexMatrix2D");
    }

    @BeforeClass
    public static void setUpBeforeClass() throws Exception {
        BenchmarkMatrixKernel.readSettings2D();
        Random rand = new Random(0);
        noViewTimes = new double[BenchmarkMatrixKernel.NTHREADS.length];
        viewTimes = new double[BenchmarkMatrixKernel.NTHREADS.length];
        ConcurrencyUtils.setThreadsBeginN_2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);

        a_1d = new float[2 * BenchmarkMatrixKernel.MATRIX_SIZE_2D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]];
        a_2d = new float[BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]][2 * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]];
        int idx = 0;
        for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]; r++) {
            for (int c = 0; c < 2 * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]; c++) {
                a_2d[r][c] = rand.nextFloat();
                a_1d[idx++] = a_2d[r][c];
            }
        }

        b_1d = new float[2 * BenchmarkMatrixKernel.MATRIX_SIZE_2D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]];
        b_2d = new float[BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]][2 * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]];
        idx = 0;
        for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]; r++) {
            for (int c = 0; c < 2 * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]; c++) {
                b_2d[r][c] = rand.nextFloat();
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

    @Test
    public void testAggregateFComplexFComplexFComplexFunctionFComplexFComplexFunction() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

  
    @Test
    public void testAggregateFComplexMatrix2DFComplexFComplexFComplexFunctionFComplexFComplexFComplexFunction() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D B = new DenseFComplexMatrix2D(b_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        FComplexMatrix2D Bv = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(b_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        String method = "aggregate(FComplexMatrix2D, FComplexFComplexFComplexFunction, FComplexFComplexFComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFloatFloat() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0], BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        float value = (float)Math.random();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }

    @Test
    public void testAssignFloatArray() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0], BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFloatArrayArray() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0], BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        String method = "assign(float[][])";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFComplexFComplexFunction() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(FComplexFunctions.acos);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.assign(FComplexFunctions.acos);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(FComplexFunctions.acos);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.assign(FComplexFunctions.acos);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(FComplexFComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }

    @Test
    public void testAssignFComplexMatrix2D() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0], BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        FComplexMatrix2D B = new DenseFComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        A = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0], BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        FComplexMatrix2D Av = A.viewDice();
        B = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Bv = B.viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        String method = "assign(FComplexMatrix2D)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }

    @Test
    public void testAssignFComplexMatrix2DFComplexFComplexFComplexFunction() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D B = new DenseFComplexMatrix2D(b_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(B, FComplexFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.assign(B, FComplexFunctions.div);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        FComplexMatrix2D Bv = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(b_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(Bv, FComplexFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.assign(Bv, FComplexFunctions.div);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(FComplexMatrix2D, FComplexFComplexFComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

   
    @Test
    public void testAssignFComplexProcedureFloatArray() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        float[] value = new float[] {-1, -2};
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
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        String method = "assign(FComplexProcedure, float[])";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFComplexProcedureFComplexFComplexFunction() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
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
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(procedure, FComplexFunctions.tan);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.assign(procedure, FComplexFunctions.tan);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(procedure, FComplexFunctions.tan);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.assign(procedure, FComplexFunctions.tan);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(FComplexProcedure, FComplexFComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testCardinality() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
    public void testFft2() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.fft2();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.fft2();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "fft2()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }
    
    @Test
    public void testFftColumns() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.fftColumns();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.fftColumns();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "fftColumns()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testFftRows() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.fftRows();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.fftRows();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "fftRows()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testForEachNonZero() {
        IntIntFComplexFunction function = new IntIntFComplexFunction() {
            public float[] apply(int first, int second, float[] third) {
                return FComplex.sqrt(third);
            }
        };
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(a_2d).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        String method = "forEachNonZero(IntIntFComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }
    
    @Test
    public void testGetConjugateTranspose() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FComplexMatrix2D B = A.getConjugateTranspose();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.getConjugateTranspose();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FComplexMatrix2D Bv = Av.getConjugateTranspose();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                Bv = Av.getConjugateTranspose();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getConjugateTranspose()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }


    @Test
    public void testGetRealPart() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FloatMatrix2D B = A.getRealPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.getRealPart();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FloatMatrix2D Bv = Av.getRealPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                Bv = Av.getRealPart();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getRealPart()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetImaginaryPart() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FloatMatrix2D B = A.getImaginaryPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.getImaginaryPart();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FloatMatrix2D Bv = Av.getImaginaryPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                Bv = Av.getImaginaryPart();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getImaginaryPart()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }
    
    

    @Test
    public void testGetNonZerosIntArrayListIntArrayListArrayList() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        IntArrayList rowList = new IntArrayList();
        IntArrayList colList = new IntArrayList();
        ArrayList<float[]> valueList = new ArrayList<float[]>();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(a_2d).viewDice();
        rowList.clear();
        colList.clear();
        valueList.clear();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIfft2() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.ifft2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.ifft2(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "ifft2(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }
    
    @Test
    public void testIfftColumns() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.ifftColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.ifftColumns(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "ifftColumns(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIfftRows() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.ifftRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.ifftRows(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "ifftRows(true)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testSum() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testToArray() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[][] array = A.toArray();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                array = A.toArray();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[][] array = Av.toArray();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                array = Av.toArray();
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
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testZMultFComplexMatrix1DFComplexMatrix1DFloatArrayFloatArrayBoolean() {
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix1D y = new DenseFComplexMatrix1D(A.columns());
        float[] alpha = new float[] {3, 4};
        float[] beta = new float[] {5, 6};
        for (int i = 0; i < y.size(); i++) {
            y.set(i, new float[] {(float)Math.random(), (float)Math.random()});
        }
        FComplexMatrix1D z = new DenseFComplexMatrix1D(A.rows());
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FComplexMatrix2D Av = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        String method = "zMult(FComplexMatrix1D, FComplexMatrix1D, float[], float[], boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testZMultFComplexMatrix2DFComplexMatrix2DFloatArrayFloatArrayBooleanBoolean() {
        int oldNiters = BenchmarkMatrixKernel.NITERS;
        BenchmarkMatrixKernel.NITERS = 10;
        /* No view */
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D B = new DenseFComplexMatrix2D(b_2d);
        B = B.viewDice().copy();
        FComplexMatrix2D C = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]);
        float[] alpha = new float[] {3, 4};
        float[] beta = new float[] {5, 6};
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        B = new DenseFComplexMatrix2D(b_2d);
        B = B.viewDice().copy();
        FComplexMatrix2D Bv = B.viewDice();
        C = new DenseFComplexMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        FComplexMatrix2D Cv = C.viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        String method = "zMult(FComplexMatrix2D, FComplexMatrix2D, float[], float[], boolean, boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.NITERS = oldNiters;
    }

}
