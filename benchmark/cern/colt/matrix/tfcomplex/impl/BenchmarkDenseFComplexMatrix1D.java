package cern.colt.matrix.tfcomplex.impl;

import java.util.Random;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import cern.colt.Timer;
import cern.colt.function.tfcomplex.FComplexProcedure;
import cern.colt.matrix.BenchmarkMatrixKernel;
import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfcomplex.FComplexMatrix3D;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.jet.math.tfcomplex.FComplex;
import cern.jet.math.tfcomplex.FComplexFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public class BenchmarkDenseFComplexMatrix1D {

    private static final Timer t = new Timer();

    private static final String outputFile = "BenchmarkDenseFComplexMatrix1D.txt";

    public static void main(String[] args) {
        org.junit.runner.JUnitCore.main("cern.colt.matrix.tfcomplex.impl.BenchmarkDenseFComplexMatrix1D");
    }

    private static double[] noViewTimes;

    private static double[] viewTimes;

    private static float[] a, b;

    @BeforeClass
    public static void setUpBeforeClass() throws Exception {
        BenchmarkMatrixKernel.readSettings1D();
        Random rand = new Random(0);
        ConcurrencyUtils.setThreadsBeginN_1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);

        a = new float[2 * BenchmarkMatrixKernel.MATRIX_SIZE_1D];
        for (int i = 0; i < 2 * BenchmarkMatrixKernel.MATRIX_SIZE_1D; i++) {
            a[i] = rand.nextFloat();
        }
        b = new float[2 * BenchmarkMatrixKernel.MATRIX_SIZE_1D];
        for (int i = 0; i < 2 * BenchmarkMatrixKernel.MATRIX_SIZE_1D; i++) {
            b[i] = rand.nextFloat();
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
    public void testAggregateFComplexFComplexFComplexFunctionFComplexFComplexFunction() {
        /* No view */
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
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
        FComplexMatrix1D Av = new DenseFComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }

    @Test
    public void testAggregateFComplexMatrix1DFComplexFComplexFComplexFunctionFComplexFComplexFComplexFunction() {
        /* No view */
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D B = new DenseFComplexMatrix1D(b);
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
        FComplexMatrix1D Av = new DenseFComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        FComplexMatrix1D Bv = new DenseFComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(b);
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
        String method = "aggregate(FComplexMatrix1D, FComplexFComplexFComplexFunction, FComplexFComplexFComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFloatFloat() {
        /* No view */
        FComplexMatrix1D A = new DenseFComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);
        float value = (float)Math.random();
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
        FComplexMatrix1D Av = new DenseFComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip();
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
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFloatArray() {
        /* No view */
        FComplexMatrix1D A = new DenseFComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);
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
        FComplexMatrix1D Av = new DenseFComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip();
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
        String method = "assign(float[])";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFComplexFComplexFunction() {
        /* No view */
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(FComplexFunctions.acos);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                A.assign(FComplexFunctions.acos);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix1D Av = new DenseFComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(FComplexFunctions.acos);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a);
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
    public void testAssignFComplexMatrix1D() {
        /* No view */
        FComplexMatrix1D A = new DenseFComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);
        FComplexMatrix1D B = new DenseFComplexMatrix1D(a);
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
        A = new DenseFComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);
        FComplexMatrix1D Av = A.viewFlip();
        B = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Bv = B.viewFlip();
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
        String method = "assign(FComplexMatrix1D)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFComplexMatrix1DFComplexFComplexFComplexFunction() {
        /* No view */
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D B = new DenseFComplexMatrix1D(b);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(B, FComplexFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                A.assign(B, FComplexFunctions.div);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix1D Av = new DenseFComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        FComplexMatrix1D Bv = new DenseFComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(b);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(Bv, FComplexFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a);
                t.reset().start();
                Av.assign(Bv, FComplexFunctions.div);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(FComplexMatrix1D, FComplexFComplexFComplexFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFComplexProcedureFloatArray() {
        /* No view */
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
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
        FComplexMatrix1D Av = new DenseFComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
        String method = "assign(FComplexProcedure, float[])";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFComplexProcedureFComplexFComplexFunction() {
        /* No view */
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
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
                A.assign(a);
                t.reset().start();
                A.assign(procedure, FComplexFunctions.tan);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix1D Av = new DenseFComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(procedure, FComplexFunctions.tan);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a);
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
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
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
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
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
    public void testGetImaginaryPart() {
        /* No view */
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        int slices = BenchmarkMatrixKernel.MATRIX_SIZE_1D / 64;
        int rows = 16;
        int cols = 4;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FloatMatrix1D B = A.getImaginaryPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.getImaginaryPart();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FloatMatrix1D B = Av.getImaginaryPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = Av.getImaginaryPart();
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
    public void testGetRealPart() {
        /* No view */
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        int slices = BenchmarkMatrixKernel.MATRIX_SIZE_1D / 64;
        int rows = 16;
        int cols = 4;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FloatMatrix1D B = A.getRealPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.getRealPart();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FloatMatrix1D B = Av.getRealPart();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = Av.getRealPart();
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
    public void testFft() {
        /* No view */
        DenseFComplexMatrix1D A = new DenseFComplexMatrix1D(a);
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
        FComplexMatrix1D Av = new DenseFComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
            ((DenseFComplexMatrix1D)Av).fft();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a);
                t.reset().start();
                ((DenseFComplexMatrix1D)Av).fft();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "fft()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIfft() {
        /* No view */
        DenseFComplexMatrix1D A = new DenseFComplexMatrix1D(a);
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
        FComplexMatrix1D Av = new DenseFComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
            ((DenseFComplexMatrix1D)Av).ifft(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a);
                t.reset().start();
                ((DenseFComplexMatrix1D)Av).ifft(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "ifft(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testReshapeIntInt() {
        /* No view */
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        int rows = BenchmarkMatrixKernel.MATRIX_SIZE_1D / 64;
        int cols = 64;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FComplexMatrix2D B = A.reshape(rows, cols);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.reshape(rows, cols);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FComplexMatrix2D B = Av.reshape(rows, cols);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = Av.reshape(rows, cols);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "reshape(int, int)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testReshapeIntIntInt() {
        /* No view */
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        int slices = BenchmarkMatrixKernel.MATRIX_SIZE_1D / 64;
        int rows = 16;
        int cols = 4;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FComplexMatrix3D B = A.reshape(slices, rows, cols);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.reshape(slices, rows, cols);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FComplexMatrix3D B = Av.reshape(slices, rows, cols);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = Av.reshape(slices, rows, cols);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "reshape(int, int, int)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testSwapFComplexMatrix1D() {
        /* No view */
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D B = new DenseFComplexMatrix1D(b);
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
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        B = new DenseFComplexMatrix1D(b);
        FComplexMatrix1D Bv = B.viewFlip();
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
        String method = "swap(FComplexMatrix1D)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testToArrayFloatArray() {
        /* No view */
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[] array = A.toArray();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                array = A.toArray();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FComplexMatrix1D Av = new DenseFComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[] array = Av.toArray();
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
    public void testZDotProductFComplexMatrix1DIntInt() {
        /* No view */
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D B = new DenseFComplexMatrix1D(b);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[] product = A.zDotProduct(B, 5, B.size() - 10);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                product = A.zDotProduct(B, 5, B.size() - 10);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        B = new DenseFComplexMatrix1D(b);
        FComplexMatrix1D Bv = B.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            float[] product = Av.zDotProduct(Bv, 5, Bv.size() - 10);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                product = Av.zDotProduct(Bv, 5, Bv.size() - 10);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "zDotProduct(FComplexMatrix1D, int, int)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testZSum() {
        /* No view */
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
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
        FComplexMatrix1D Av = new DenseFComplexMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

}
