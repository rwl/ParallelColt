package cern.colt.matrix.tfloat.impl;

import java.util.Random;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import cern.colt.Timer;
import cern.colt.function.tfloat.FloatFunction;
import cern.colt.function.tfloat.FloatProcedure;
import cern.colt.list.tfloat.FloatArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.BenchmarkMatrixKernel;
import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix3D;
import cern.jet.math.tfloat.FloatFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public class BenchmarkDenseFloatMatrix1D {

    private static final Timer t = new Timer();

    private static final String outputFile = "BenchmarkDenseFloatMatrix1D.txt";

    public static void main(String[] args) {
        org.junit.runner.JUnitCore.main("cern.colt.matrix.tfloat.impl.BenchmarkDenseFloatMatrix1D");
    }

    private static double[] noViewTimes;

    private static double[] viewTimes;

    private static float[] a, b;

    private static final long millis = 5000;

    @BeforeClass
    public static void setUpBeforeClass() throws Exception {
        BenchmarkMatrixKernel.readSettings1D();
        Random rand = new Random(0);
        ConcurrencyUtils.setThreadsBeginN_1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);

        a = new float[BenchmarkMatrixKernel.MATRIX_SIZE_1D];
        for (int i = 0; i < BenchmarkMatrixKernel.MATRIX_SIZE_1D; i++) {
            a[i] = rand.nextFloat();
        }
        b = new float[BenchmarkMatrixKernel.MATRIX_SIZE_1D];
        for (int i = 0; i < BenchmarkMatrixKernel.MATRIX_SIZE_1D; i++) {
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
    public void testAggregateFloatFloatFunctionFloatFunction() {
        /* No view */
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            float aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.square);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
            ConcurrencyUtils.sleep(millis);
        }
        /* View */
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            float aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.square);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
            ConcurrencyUtils.sleep(millis);
        }
        String method = "aggregate(FloatFloatFunction, FloatFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }

    @Test
    public void testAssignFloatMatrix1D() {
        /* No view */
        FloatMatrix1D A = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);
        FloatMatrix1D B = new DenseFloatMatrix1D(a);
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
        A = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);
        FloatMatrix1D Av = A.viewFlip();
        B = new DenseFloatMatrix1D(a);
        FloatMatrix1D Bv = B.viewFlip();
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
        String method = "assign(FloatMatrix1D)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testZDotProductFloatMatrix1D() {
        /* No view */
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatMatrix1D B = new DenseFloatMatrix1D(b);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float product = A.zDotProduct(B);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                product = A.zDotProduct(B);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        B = new DenseFloatMatrix1D(b);
        FloatMatrix1D Bv = B.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            float product = Av.zDotProduct(Bv);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                product = Av.zDotProduct(Bv);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "zDotProduct(FloatMatrix1D, int, int)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFloatFunction() {
        FloatFunction f = FloatFunctions.mult(2.5f);
        /* No view */
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
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
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
        String method = "assign(FloatFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFloatMatrix1DFloatFloatFunction() {
        /* No view */
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatMatrix1D B = new DenseFloatMatrix1D(b);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(B, FloatFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                A.assign(B, FloatFunctions.div);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        FloatMatrix1D Bv = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(b);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(Bv, FloatFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a);
                t.reset().start();
                Av.assign(Bv, FloatFunctions.div);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(FloatMatrix1D, FloatFloatFuction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAggregateFloatMatrix1DFloatFloatFunctionFloatFloatFunction() {
        /* No view */
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatMatrix1D B = new DenseFloatMatrix1D(b);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float aSum = A.aggregate(B, FloatFunctions.plus, FloatFunctions.mult);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.aggregate(B, FloatFunctions.plus, FloatFunctions.mult);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        FloatMatrix1D Bv = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(b);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float aSum = Av.aggregate(Bv, FloatFunctions.plus, FloatFunctions.mult);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = Av.aggregate(Bv, FloatFunctions.plus, FloatFunctions.mult);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "aggregate(FloatMatrix1D, FloatFloatFunction, FloatFloatFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFloat() {
        /* No view */
        FloatMatrix1D A = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);
        float value = (float) Math.random();
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
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip();
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
        String method = "assign(float)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFloatArray() {
        /* No view */
        FloatMatrix1D A = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D);
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
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip();
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
        String method = "assign(float[])";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFloatProcedureFloat() {
        /* No view */
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatProcedure procedure = new FloatProcedure() {
            public boolean apply(float element) {
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
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
        String method = "assign(FloatProcedure, float)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFloatProcedureFloatFunction() {
        /* No view */
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatProcedure procedure = new FloatProcedure() {
            public boolean apply(float element) {
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
            A.assign(procedure, FloatFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a);
                t.reset().start();
                A.assign(procedure, FloatFunctions.square);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(procedure, FloatFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a);
                t.reset().start();
                Av.assign(procedure, FloatFunctions.square);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(FloatProcedure, FloatFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testCardinality() {
        /* No view */
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
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
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
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
    public void testFft() {
        /* No view */
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
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
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
            ((DenseFloatMatrix1D) Av).fft();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseFloatMatrix1D) Av).assign(a);
                t.reset().start();
                ((DenseFloatMatrix1D) Av).fft();
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
    public void testDct() {
        /* No view */
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
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
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
            ((DenseFloatMatrix1D) Av).dct(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseFloatMatrix1D) Av).assign(a);
                t.reset().start();
                ((DenseFloatMatrix1D) Av).dct(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dct(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }

    @Test
    public void testDht() {
        /* No view */
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
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
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
            ((DenseFloatMatrix1D) Av).dht();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseFloatMatrix1D) Av).assign(a);
                t.reset().start();
                ((DenseFloatMatrix1D) Av).dht();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dht()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }

    @Test
    public void testDst() {
        /* No view */
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
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
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
            ((DenseFloatMatrix1D) Av).dst(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseFloatMatrix1D) Av).assign(a);
                t.reset().start();
                ((DenseFloatMatrix1D) Av).dst(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dst(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetFft() {
        /* No view */
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
        FComplexMatrix1D Ac;
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
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
            Ac = ((DenseFloatMatrix1D) Av).getFft();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseFloatMatrix1D) Av).assign(a);
                t.reset().start();
                Ac = ((DenseFloatMatrix1D) Av).getFft();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getFft()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetIfft() {
        /* No view */
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
        FComplexMatrix1D Ac;
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
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
            Ac = ((DenseFloatMatrix1D) Av).getIfft(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseFloatMatrix1D) Av).assign(a);
                t.reset().start();
                Ac = ((DenseFloatMatrix1D) Av).getIfft(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getIfft(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }

    @Test
    public void testGetPositiveValuesIntArrayListFloatArrayList() {
        /* No view */
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
        IntArrayList indexList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
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
        FloatMatrix1D Av = new DenseFloatMatrix1D(a).viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseFloatMatrix1D) Av).getPositiveValues(indexList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                indexList.clear();
                valueList.clear();
                t.reset().start();
                ((DenseFloatMatrix1D) Av).getPositiveValues(indexList, valueList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getPositiveValues(IntArrayList, FloatArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetNegativeValuesIntArrayListFloatArrayList() {
        /* No view */
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
        A.assign(FloatFunctions.mult(-1));
        IntArrayList indexList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
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
        FloatMatrix1D Av = new DenseFloatMatrix1D(a).viewFlip();
        ((DenseFloatMatrix1D) Av).assign(FloatFunctions.mult(-1));
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseFloatMatrix1D) Av).getNegativeValues(indexList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                indexList.clear();
                valueList.clear();
                t.reset().start();
                ((DenseFloatMatrix1D) Av).getNegativeValues(indexList, valueList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getNegativeValues(IntArrayList, FloatArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIdct() {
        /* No view */
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
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
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
            ((DenseFloatMatrix1D) Av).idct(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseFloatMatrix1D) Av).assign(a);
                t.reset().start();
                ((DenseFloatMatrix1D) Av).idct(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "idct(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }

    @Test
    public void testIdht() {
        /* No view */
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
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
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
            ((DenseFloatMatrix1D) Av).idht(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseFloatMatrix1D) Av).assign(a);
                t.reset().start();
                ((DenseFloatMatrix1D) Av).idht(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "idht(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }

    @Test
    public void testIdst() {
        /* No view */
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
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
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
            ((DenseFloatMatrix1D) Av).idst(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseFloatMatrix1D) Av).assign(a);
                t.reset().start();
                ((DenseFloatMatrix1D) Av).idst(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "idst(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIfft() {
        /* No view */
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
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
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
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
            ((DenseFloatMatrix1D) Av).ifft(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                ((DenseFloatMatrix1D) Av).assign(a);
                t.reset().start();
                ((DenseFloatMatrix1D) Av).ifft(true);
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
    public void testMaxLocation() {
        /* No view */
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[] maxAndLoc = A.getMaxLocation();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                maxAndLoc = A.getMaxLocation();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FloatMatrix1D Av = new DenseFloatMatrix1D(a).viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[] maxAndLoc = ((DenseFloatMatrix1D) Av).getMaxLocation();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                maxAndLoc = ((DenseFloatMatrix1D) Av).getMaxLocation();
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
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[] minAndLoc = A.getMinLocation();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                minAndLoc = A.getMinLocation();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FloatMatrix1D Av = new DenseFloatMatrix1D(a).viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[] minAndLoc = ((DenseFloatMatrix1D) Av).getMinLocation();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                minAndLoc = ((DenseFloatMatrix1D) Av).getMinLocation();
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
    public void testReshapeIntInt() {
        /* No view */
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
        int rows = BenchmarkMatrixKernel.MATRIX_SIZE_1D / 64;
        int cols = 64;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FloatMatrix2D B = A.reshape(rows, cols);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.reshape(rows, cols);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FloatMatrix2D B = ((DenseFloatMatrix1D) Av).reshape(rows, cols);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = ((DenseFloatMatrix1D) Av).reshape(rows, cols);
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
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
        int slices = BenchmarkMatrixKernel.MATRIX_SIZE_1D / 64;
        int rows = 16;
        int cols = 4;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FloatMatrix3D B = A.reshape(slices, rows, cols);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.reshape(slices, rows, cols);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FloatMatrix3D B = ((DenseFloatMatrix1D) Av).reshape(slices, rows, cols);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = ((DenseFloatMatrix1D) Av).reshape(slices, rows, cols);
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
    public void testSwapFloatMatrix1D() {
        /* No view */
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatMatrix1D B = new DenseFloatMatrix1D(b);
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
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        B = new DenseFloatMatrix1D(b);
        FloatMatrix1D Bv = B.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            ((DenseFloatMatrix1D) Av).swap(Bv);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                ((DenseFloatMatrix1D) Av).swap(Bv);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "swap(FloatMatrix1D)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testToArrayFloatArray() {
        /* No view */
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
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
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[] array = ((DenseFloatMatrix1D) Av).toArray();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                array = ((DenseFloatMatrix1D) Av).toArray();
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
    public void testZDotProductFloatMatrix1DIntInt() {
        /* No view */
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatMatrix1D B = new DenseFloatMatrix1D(b);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float product = A.zDotProduct(B, 5, B.size() - 10);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                product = A.zDotProduct(B, 5, B.size() - 10);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        B = new DenseFloatMatrix1D(b);
        FloatMatrix1D Bv = B.viewFlip();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            float product = ((DenseFloatMatrix1D) Av).zDotProduct(Bv, 5, Bv.size() - 10);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                product = ((DenseFloatMatrix1D) Av).zDotProduct(Bv, 5, Bv.size() - 10);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "zDotProduct(FloatMatrix1D, int, int)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testZSum() {
        /* No view */
        DenseFloatMatrix1D A = new DenseFloatMatrix1D(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float aSum = A.zSum();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.zSum();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FloatMatrix1D Av = new DenseFloatMatrix1D(BenchmarkMatrixKernel.MATRIX_SIZE_1D).viewFlip().assign(a);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfThreads(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float aSum = ((DenseFloatMatrix1D) Av).zSum();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = ((DenseFloatMatrix1D) Av).zSum();
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
