package cern.colt.matrix.impl;

import java.util.Random;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import cern.colt.Timer;
import cern.colt.function.FloatProcedure;
import cern.colt.function.IntIntFloatFunction;
import cern.colt.list.FloatArrayList;
import cern.colt.list.IntArrayList;
import cern.colt.matrix.BenchmarkMatrixKernel;
import cern.colt.matrix.FComplexMatrix2D;
import cern.colt.matrix.FloatMatrix1D;
import cern.colt.matrix.FloatMatrix2D;
import cern.jet.math.FloatFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public class BenchmarkDenseFloatMatrix2D {

    private static Timer t = new Timer();

    private static final String outputFile = "BenchmarkDenseFloatMatrix2D.txt";

    private static float[][] a_2d, b_2d;

    private static float[] a_1d, b_1d;

    private static double[] noViewTimes;

    private static double[] viewTimes;

    public static void main(String[] args) {
        org.junit.runner.JUnitCore.main("cern.colt.matrix.impl.BenchmarkDenseFloatMatrix2D");
    }

    @BeforeClass
    public static void setUpBeforeClass() throws Exception {
        BenchmarkMatrixKernel.readSettings2D();
        Random rand = new Random(0);
        noViewTimes = new double[BenchmarkMatrixKernel.NTHREADS.length];
        viewTimes = new double[BenchmarkMatrixKernel.NTHREADS.length];
        ConcurrencyUtils.setThreadsBeginN_2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);

        a_1d = new float[BenchmarkMatrixKernel.MATRIX_SIZE_2D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]];
        a_2d = new float[BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]][BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]];
        int idx = 0;
        for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]; r++) {
            for (int c = 0; c < BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]; c++) {
                a_2d[r][c] = rand.nextFloat();
                a_1d[idx++] = a_2d[r][c];
            }
        }

        b_1d = new float[BenchmarkMatrixKernel.MATRIX_SIZE_2D[0] * BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]];
        b_2d = new float[BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]][BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]];
        idx = 0;
        for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]; r++) {
            for (int c = 0; c < BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]; c++) {
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
    public void testAggregateFloatFloatFunctionFloatFunction() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.square);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.square);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "aggregate(FloatFloatFunction, FloatFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAggregateFloatFloatFunctionFloatFunctionFloatProcedure() {
        FloatProcedure procedure = new FloatProcedure() {
            public boolean apply(float element) {
                if (Math.abs(element) > 0.2) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.square, procedure);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.square, procedure);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.square, procedure);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.square, procedure);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "aggregate(FloatFloatFunction, FloatFunction, FloatProcedure)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAggregateFloatFloatFunctionFloatFunctionIntArrayListIntArrayList() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]; r++) {
            for (int c = 0; c < BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]; c++) {
                rowList.add(r);
                columnList.add(c);
            }
        }
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.square, rowList, columnList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.square, rowList, columnList);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.square, rowList, columnList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.square, rowList, columnList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "aggregate(FloatFloatFunction, FloatFunction, IntArrayList, IntArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAggregateFloatMatrix2DFloatFloatFunctionFloatFloatFunction() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D B = new DenseFloatMatrix2D(b_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        FloatMatrix2D Bv = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(b_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        String method = "aggregate(FloatMatrix2D, FloatFloatFunction, FloatFloatFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFloat() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0], BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        float value = (float)Math.random();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FloatMatrix2D A = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0], BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        String method = "assign(float[])";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFloatArrayArray() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0], BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        String method = "assign(float[][])";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFloatFunction() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(FloatFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.assign(FloatFunctions.square);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(FloatFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.assign(FloatFunctions.square);
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
    public void testAssignFloatMatrix2D() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0], BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        FloatMatrix2D B = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        A = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0], BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        FloatMatrix2D Av = A.viewDice();
        B = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Bv = B.viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        String method = "assign(FloatMatrix2D)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }

    @Test
    public void testAssignFloatMatrix2DFloatFloatFunction() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D B = new DenseFloatMatrix2D(b_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(B, FloatFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.assign(B, FloatFunctions.div);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        FloatMatrix2D Bv = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(b_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(Bv, FloatFunctions.div);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.assign(Bv, FloatFunctions.div);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(FloatMatrix2D, FloatFloatFuction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFloatMatrix2DFloatFloatFunctionIntArrayListIntArrayList() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D B = new DenseFloatMatrix2D(b_2d);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int r = 0; r < BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]; r++) {
            for (int c = 0; c < BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]; c++) {
                rowList.add(r);
                columnList.add(c);
            }
        }
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(B, FloatFunctions.div, rowList, columnList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.assign(B, FloatFunctions.div, rowList, columnList);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        FloatMatrix2D Bv = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(b_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(Bv, FloatFunctions.div, rowList, columnList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.assign(Bv, FloatFunctions.div, rowList, columnList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "assign(FloatMatrix2D, FloatFloatFuction, IntArrayList, IntArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFloatProcedureFloat() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
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
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        String method = "assign(FloatProcedure, float)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testAssignFloatProcedureFloatFunction() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
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
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            A.assign(procedure, FloatFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                A.assign(a_2d);
                t.reset().start();
                A.assign(procedure, FloatFunctions.square);
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.assign(procedure, FloatFunctions.square);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
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
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
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
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
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
    public void testDct2() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.dct2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.dct2(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dct2(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testDctColumns() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.dctColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.dctColumns(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dctColumns(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testDctRows() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.dctRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.dctRows(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dctRows(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }

    @Test
    public void testDht2() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.dht2();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.dht2();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dht2()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testDhtColumns() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.dhtColumns();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.dhtColumns();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dhtColumns()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testDhtRows() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.dhtRows();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.dhtRows();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dhtRows()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }
    
    @Test
    public void testDst2() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.dst2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.dst2(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dst2(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testDstColumns() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.dstColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.dstColumns(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "dstColumns(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testDstRows() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.dstRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.dstRows(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "dstRows(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testFft2() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
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
    public void testForEachNonZero() {
        IntIntFloatFunction function = new IntIntFloatFunction() {
            public float apply(int first, int second, float third) {
                return (float)Math.sqrt(third);
            }
        };
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FloatMatrix2D Av = new DenseFloatMatrix2D(a_2d).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.forEachNonZero(function);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                Av.forEachNonZero(function);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "forEachNonZero(IntIntFloatFunction)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }


    @Test
    public void testGetFft2() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FComplexMatrix2D Ac;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Ac = Av.getFft2();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Ac = Av.getFft2();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getFft2()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetFftColumns() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FComplexMatrix2D Ac;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Ac = Av.getFftColumns();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Ac = Av.getFftColumns();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "getFftColumns()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetFftRows() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FComplexMatrix2D Ac;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Ac = Av.getFftRows();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Ac = Av.getFftRows();
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "getFftRows()";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetIfft2() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
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
        String method = "getIfft2(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetIfftColumns() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FComplexMatrix2D Ac;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Ac = Av.getIfftColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Ac = Av.getIfftColumns(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "getIfftColumns(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetIfftRows() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FComplexMatrix2D Ac;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Ac = Av.getIfftRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Ac = Av.getIfftRows(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "getIfftRows(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetNonZerosIntArrayListIntArrayListFloatArrayList() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        IntArrayList rowList = new IntArrayList();
        IntArrayList colList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(a_2d).viewDice();
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
        String method = "getNonZeros(IntArrayList, IntArrayList, FloatArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testGetPositiveValuesIntArrayListIntArrayListFloatArrayList() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FloatMatrix2D Av = new DenseFloatMatrix2D(a_2d).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.getPositiveValues(rowList, columnList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                rowList.clear();
                columnList.clear();
                valueList.clear();
                t.reset().start();
                Av.getPositiveValues(rowList, columnList, valueList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getPositiveValues(IntArrayList, IntArrayList, FloatArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }
    
    @Test
    public void testGetNegativeValuesIntArrayListIntArrayListFloatArrayList() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        A.assign(FloatFunctions.mult(-1));
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FloatMatrix2D Av = new DenseFloatMatrix2D(a_2d).viewDice();
        Av.assign(FloatFunctions.mult(-1));
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.getNegativeValues(rowList, columnList, valueList);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                rowList.clear();
                columnList.clear();
                valueList.clear();
                t.reset().start();
                Av.getNegativeValues(rowList, columnList, valueList);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "getNegativeValues(IntArrayList, IntArrayList, FloatArrayList)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }


    @Test
    public void testIdct2() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.idct2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.idct2(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "idct2(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIdctColumns() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.idctColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.idctColumns(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "idctColumns(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIdctRows() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.idctRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.idctRows(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "idctRows(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }
    
    @Test
    public void testIdht2() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.idht2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.idht2(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "idht2(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIdhtColumns() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.idhtColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.idhtColumns(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "idhtColumns(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIdhtRows() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.idhtRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.idhtRows(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "idhtRows(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIdst2() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            Av.idst2(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.idst2(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "idst2(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIdstColumns() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.idstColumns(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.idstColumns(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "idstColumns(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
    }

    @Test
    public void testIdstRows() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.idstRows(true);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Av.assign(a_2d);
                t.reset().start();
                Av.idstRows(true);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }

        String method = "idstRows(boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testIfft2() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
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
    public void testMaxLocation() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FloatMatrix2D Av = new DenseFloatMatrix2D(a_2d).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[] maxAndLoc = Av.getMaxLocation();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                maxAndLoc = Av.getMaxLocation();
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
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FloatMatrix2D Av = new DenseFloatMatrix2D(a_2d).viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float[] minAndLoc = Av.getMinLocation();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                minAndLoc = Av.getMinLocation();
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
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            float aSum = Av.zSum();
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
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
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
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FloatMatrix1D B = A.vectorize();
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                t.reset().start();
                B = A.vectorize();
                t.stop();
                noViewTimes[i] += t.millis();
            }
            noViewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        /* View */
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

            // warm-up
            FloatMatrix1D B = Av.vectorize();
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
    public void testZMultFloatMatrix1DFloatMatrix1DFloatFloatBoolean() {
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix1D y = new DenseFloatMatrix1D(A.columns());
        float alpha = 3;
        float beta = 5;
        for (int i = 0; i < y.size(); i++) {
            y.set(i, (float)Math.random());
        }
        FloatMatrix1D z = new DenseFloatMatrix1D(A.rows());
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        FloatMatrix2D Av = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]).viewDice().assign(a_2d);
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.zMult(y, z, alpha, beta, false);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                z.assign(0);
                t.reset().start();
                Av.zMult(y, z, alpha, beta, false);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "zMult(FloatMatrix1D, FloatMatrix1D, float, float, boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);

    }

    @Test
    public void testZMultFloatMatrix2DFloatMatrix2DFloatFloatBooleanBoolean() {
        int oldNiters = BenchmarkMatrixKernel.NITERS;
        BenchmarkMatrixKernel.NITERS = 10;
        /* No view */
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D B = new DenseFloatMatrix2D(b_2d);
        B = B.viewDice().copy();
        FloatMatrix2D C = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[0], BenchmarkMatrixKernel.MATRIX_SIZE_2D[0]);
        float alpha = 3;
        float beta = 5;
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);

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
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        B = new DenseFloatMatrix2D(b_2d);
        B = B.viewDice().copy();
        FloatMatrix2D Bv = B.viewDice();
        C = new DenseFloatMatrix2D(BenchmarkMatrixKernel.MATRIX_SIZE_2D[1], BenchmarkMatrixKernel.MATRIX_SIZE_2D[1]);
        FloatMatrix2D Cv = C.viewDice();
        for (int i = 0; i < BenchmarkMatrixKernel.NTHREADS.length; i++) {
            ConcurrencyUtils.setNumberOfProcessors(BenchmarkMatrixKernel.NTHREADS[i]);
            // warm-up
            Av.zMult(Bv, Cv, alpha, beta, false, false);
            for (int j = 0; j < BenchmarkMatrixKernel.NITERS; j++) {
                Cv.assign(0);
                t.reset().start();
                Av.zMult(Bv, Cv, alpha, beta, false, false);
                t.stop();
                viewTimes[i] += t.millis();
            }
            viewTimes[i] /= BenchmarkMatrixKernel.NITERS;
        }
        String method = "zMult(FloatMatrix2D, FloatMatrix2D, float, float, boolean, boolean)";
        BenchmarkMatrixKernel.writeMatrixBenchmarkResultsToFile(outputFile, method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.displayMatrixBenchmarkResults(method, BenchmarkMatrixKernel.NTHREADS, noViewTimes, viewTimes);
        BenchmarkMatrixKernel.NITERS = oldNiters;
    }

}
