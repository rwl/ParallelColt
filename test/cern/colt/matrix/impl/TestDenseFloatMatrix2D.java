package cern.colt.matrix.impl;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Random;

import junit.framework.Assert;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import cern.colt.function.FloatProcedure;
import cern.colt.function.IntIntFloatFunction;
import cern.colt.list.FloatArrayList;
import cern.colt.list.IntArrayList;
import cern.colt.matrix.FComplexMatrix2D;
import cern.colt.matrix.FloatMatrix1D;
import cern.colt.matrix.FloatMatrix1DProcedure;
import cern.colt.matrix.FloatMatrix2D;
import cern.jet.math.FloatFunctions;
import edu.emory.mathcs.utils.AssertUtils;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public class TestDenseFloatMatrix2D {
    private static final int rows = 113;

    private static final int cols = 117;

    private static final float tol = 1e-1f;

    private static final int nThreads = 3;

    private static final int nThreadsBegin = 1;

    private float[][] a_2d, b_2d;

    private float[] a_1d, b_1d;

    private Random rand;

    @Before
    public void setUpBeforeClass() throws Exception {
        rand = new Random(0);

        a_1d = new float[rows * cols];
        a_2d = new float[rows][cols];
        int idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                a_2d[r][c] = rand.nextFloat();
                a_1d[idx++] = a_2d[r][c];
            }
        }

        b_1d = new float[rows * cols];
        b_2d = new float[rows][cols];
        idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                b_2d[r][c] = rand.nextFloat();
                b_1d[idx++] = b_2d[r][c];
            }
        }
    }

    @After
    public void tearDownAfterClass() throws Exception {
        a_1d = null;
        a_2d = null;
        b_1d = null;
        b_2d = null;
        System.gc();
    }

    @Test
    public void testAggregateFloatFloatFunctionFloatFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        float aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.square);
        float tmpSum = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmpSum += a_2d[r][c] * a_2d[r][c];
            }
        }
        assertEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.square);
        tmpSum = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmpSum += a_2d[r][c] * a_2d[r][c];
            }
        }
        assertEquals(tmpSum, aSum, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.square);
        tmpSum = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmpSum += a_2d[r][c] * a_2d[r][c];
            }
        }
        assertEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Av = A.viewDice();
        aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.square);
        tmpSum = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmpSum += a_2d[r][c] * a_2d[r][c];
            }
        }
        assertEquals(tmpSum, aSum, tol);
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
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        float aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.square, procedure);
        float tmpSum = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (Math.abs(a_2d[r][c]) > 0.2) {
                    tmpSum += a_2d[r][c] * a_2d[r][c];
                }
            }
        }
        assertEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.square, procedure);
        tmpSum = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (Math.abs(a_2d[r][c]) > 0.2) {
                    tmpSum += a_2d[r][c] * a_2d[r][c];
                }
            }
        }
        assertEquals(tmpSum, aSum, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.square, procedure);
        tmpSum = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (Math.abs(a_2d[r][c]) > 0.2) {
                    tmpSum += a_2d[r][c] * a_2d[r][c];
                }
            }
        }
        assertEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Av = A.viewDice();
        aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.square, procedure);
        tmpSum = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (Math.abs(a_2d[r][c]) > 0.2) {
                    tmpSum += a_2d[r][c] * a_2d[r][c];
                }
            }
        }
        assertEquals(tmpSum, aSum, tol);
    }

    @Test
    public void testAggregateFloatFloatFunctionFloatFunctionIntArrayListIntArrayList() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                rowList.add(r);
                columnList.add(c);
            }  
        }     
        float aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.square, rowList, columnList);
        float tmpSum = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                    tmpSum += a_2d[r][c] * a_2d[r][c];
            }
        }
        assertEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.square, rowList, columnList);
        tmpSum = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                    tmpSum += a_2d[r][c] * a_2d[r][c];
            }
        }
        assertEquals(tmpSum, aSum, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.square, columnList, rowList);
        tmpSum = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                    tmpSum += a_2d[r][c] * a_2d[r][c];
            }
        }
        assertEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.square, columnList, rowList);
        tmpSum = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                    tmpSum += a_2d[r][c] * a_2d[r][c];
            }
        }
        assertEquals(tmpSum, aSum, tol);
    }
    
    @Test
    public void testAggregateFloatMatrix2DFloatFloatFunctionFloatFloatFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D B = new DenseFloatMatrix2D(b_2d);
        float sumMult = A.aggregate(B, FloatFunctions.plus, FloatFunctions.mult);
        float tmpSumMult = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmpSumMult += a_2d[r][c] * b_2d[r][c];
            }
        }
        assertEquals(tmpSumMult, sumMult, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        B = new DenseFloatMatrix2D(b_2d);
        sumMult = A.aggregate(B, FloatFunctions.plus, FloatFunctions.mult);
        tmpSumMult = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmpSumMult += a_2d[r][c] * b_2d[r][c];
            }
        }
        assertEquals(tmpSumMult, sumMult, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        B = new DenseFloatMatrix2D(b_2d);
        FloatMatrix2D Bv = B.viewDice();
        sumMult = Av.aggregate(Bv, FloatFunctions.plus, FloatFunctions.mult);
        tmpSumMult = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmpSumMult += a_2d[r][c] * b_2d[r][c];
            }
        }
        assertEquals(tmpSumMult, sumMult, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Av = A.viewDice();
        B = new DenseFloatMatrix2D(b_2d);
        Bv = B.viewDice();
        sumMult = Av.aggregate(Bv, FloatFunctions.plus, FloatFunctions.mult);
        tmpSumMult = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmpSumMult += a_2d[r][c] * b_2d[r][c];
            }
        }
        assertEquals(tmpSumMult, sumMult, tol);

    }

    @Test
    public void testAssignFloat() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        float value = (float)Math.random();
        A.assign(value);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(value, A.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(value);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(value, A.getQuick(r, c), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(rows, cols);
        FloatMatrix2D Av = A.viewDice();
        value = (float)Math.random();
        Av.assign(value);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(value, Av.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        Av = A.viewDice();
        value = (float)Math.random();
        Av.assign(value);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(value, Av.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testAssignFloatArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        float[] aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a_1d, aElts, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a_1d, aElts, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(rows, cols);
        FloatMatrix2D Av = A.viewDice();
        Av.assign(a_1d);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(a_1d[r * rows + c], Av.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        Av = A.viewDice();
        Av.assign(a_1d);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(a_1d[r * rows + c], Av.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testAssignFloatArrayArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_2d);
        float[] aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a_1d, aElts, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_2d);
        aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a_1d, aElts, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(cols, rows);
        FloatMatrix2D Av = A.viewDice();
        Av.assign(a_2d);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_2d[r][c], Av.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(cols, rows);
        Av = A.viewDice();
        Av.assign(a_2d);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_2d[r][c], Av.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testAssignFloatFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        A.assign(FloatFunctions.acos);
        float tmp;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmp = a_2d[r][c];
                tmp = (float)Math.acos(tmp);
                assertEquals(tmp, A.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        A.assign(FloatFunctions.acos);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmp = a_2d[r][c];
                tmp = (float)Math.acos(tmp);
                assertEquals(tmp, A.getQuick(r, c), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        Av.assign(FloatFunctions.acos);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp = a_2d[c][r];
                tmp = (float)Math.acos(tmp);
                assertEquals(tmp, Av.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Av = A.viewDice();
        Av.assign(FloatFunctions.acos);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp = a_2d[c][r];
                tmp = (float)Math.acos(tmp);
                assertEquals(tmp, Av.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testAssignFloatMatrix2D() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        FloatMatrix2D B = new DenseFloatMatrix2D(a_2d);
        A.assign(B);
        float[] aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a_1d, aElts, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        B = new DenseFloatMatrix2D(a_2d);
        A.assign(B);
        aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a_1d, aElts, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(rows, cols);
        FloatMatrix2D Av = A.viewDice();
        B = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Bv = B.viewDice();
        Av.assign(Bv);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(a_2d[c][r], Av.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        Av = A.viewDice();
        B = new DenseFloatMatrix2D(a_2d);
        Bv = B.viewDice();
        Av.assign(Bv);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(a_2d[c][r], Av.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testAssignFloatMatrix2DFloatFloatFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D B = new DenseFloatMatrix2D(b_2d);
        A.assign(B, FloatFunctions.div);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_2d[r][c] / b_2d[r][c], A.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        B = new DenseFloatMatrix2D(b_2d);
        A.assign(B, FloatFunctions.div);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_2d[r][c] / b_2d[r][c], A.getQuick(r, c), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        B = new DenseFloatMatrix2D(b_2d);
        FloatMatrix2D Bv = B.viewDice();
        Av.assign(Bv, FloatFunctions.div);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(a_2d[c][r] / b_2d[c][r], Av.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Av = A.viewDice();
        B = new DenseFloatMatrix2D(b_2d);
        Bv = B.viewDice();
        Av.assign(Bv, FloatFunctions.div);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(a_2d[c][r] / b_2d[c][r], Av.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testAssignFloatMatrix2DFloatFloatFunctionIntArrayListIntArrayList() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D B = new DenseFloatMatrix2D(b_2d);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                rowList.add(r);
                columnList.add(c);
            }  
        } 
        A.assign(B, FloatFunctions.div, rowList, columnList);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_2d[r][c] / b_2d[r][c], A.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        B = new DenseFloatMatrix2D(b_2d);
        A.assign(B, FloatFunctions.div, rowList, columnList);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_2d[r][c] / b_2d[r][c], A.getQuick(r, c), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        B = new DenseFloatMatrix2D(b_2d);
        FloatMatrix2D Bv = B.viewDice();
        Av.assign(Bv, FloatFunctions.div, columnList, rowList);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(a_2d[c][r] / b_2d[c][r], Av.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Av = A.viewDice();
        B = new DenseFloatMatrix2D(b_2d);
        Bv = B.viewDice();
        Av.assign(Bv, FloatFunctions.div, columnList, rowList);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(a_2d[c][r] / b_2d[c][r], Av.getQuick(r, c), tol);
            }
        }
    }
    
    @Test
    public void testAssignFloatProcedureFloat() {
        FloatProcedure procedure = new FloatProcedure() {
            public boolean apply(float element) {
                if ((float)Math.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D B = A.copy();
        A.assign(procedure, -1);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if ((float)Math.abs(B.getQuick(r, c)) > 0.1) {
                    B.setQuick(r, c, -1);
                }
            }
        }
        AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        B = A.copy();
        A.assign(procedure, -1);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if ((float)Math.abs(B.getQuick(r, c)) > 0.1) {
                    B.setQuick(r, c, -1);
                }
            }
        }
        AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        B = A.copy();
        FloatMatrix2D Bv = B.viewDice();
        Av.assign(procedure, -1);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                if ((float)Math.abs(Bv.getQuick(r, c)) > 0.1) {
                    Bv.setQuick(r, c, -1);
                }
            }
        }
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(Bv.getQuick(r, c), Av.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Av = A.viewDice();
        B = A.copy();
        Bv = B.viewDice();
        Av.assign(procedure, -1);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                if ((float)Math.abs(Bv.getQuick(r, c)) > 0.1) {
                    Bv.setQuick(r, c, -1);
                }
            }
        }
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(Bv.getQuick(r, c), Av.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testAssignFloatProcedureFloatFunction() {
        FloatProcedure procedure = new FloatProcedure() {
            public boolean apply(float element) {
                if ((float)Math.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D B = A.copy();
        A.assign(procedure, FloatFunctions.tan);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if ((float)Math.abs(B.getQuick(r, c)) > 0.1) {
                    B.setQuick(r, c, (float)Math.tan(B.getQuick(r, c)));
                }
            }
        }
        AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        B = A.copy();
        A.assign(procedure, FloatFunctions.tan);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if ((float)Math.abs(B.getQuick(r, c)) > 0.1) {
                    B.setQuick(r, c, (float)Math.tan(B.getQuick(r, c)));
                }
            }
        }
        AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        B = A.copy();
        FloatMatrix2D Bv = B.viewDice();
        Av.assign(procedure, FloatFunctions.tan);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                if ((float)Math.abs(Bv.getQuick(r, c)) > 0.1) {
                    Bv.setQuick(r, c, (float)Math.tan(Bv.getQuick(r, c)));
                }
            }
        }
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(Bv.getQuick(r, c), Av.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Av = A.viewDice();
        B = A.copy();
        Bv = B.viewDice();
        Av.assign(procedure, FloatFunctions.tan);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                if ((float)Math.abs(Bv.getQuick(r, c)) > 0.1) {
                    Bv.setQuick(r, c, (float)Math.tan(Bv.getQuick(r, c)));
                }
            }
        }
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(Bv.getQuick(r, c), Av.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testCardinality() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        int card = A.cardinality();
        assertEquals(rows * cols, card);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        card = A.cardinality();
        assertEquals(rows * cols, card);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        card = Av.cardinality();
        assertEquals(rows * cols, card);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Av = A.viewDice();
        card = Av.cardinality();
        assertEquals(rows * cols, card);
    }

    @Test
    public void testDct2() {
        int rows = 512;
        int cols = 256;
        float[] a_1d = new float[rows * cols];
        for (int i = 0; i < rows * cols; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.dct2(true);
        A.idct2(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.dct2(true);
        A.idct2(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        FloatMatrix2D Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        float[] av_elems = (float[])Av.copy().elements();
        Av.dct2(true);
        Av.idct2(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        av_elems = (float[])Av.copy().elements();
        Av.dct2(true);
        Av.idct2(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
    }

    @Test
    public void testDctColumns() {
        int rows = 512;
        int cols = 256;
        float[] a_1d = new float[rows * cols];
        for (int i = 0; i < rows * cols; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.dctColumns(true);
        A.idctColumns(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.dctColumns(true);
        A.idctColumns(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        FloatMatrix2D Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        float[] av_elems = (float[])Av.copy().elements();
        Av.dctColumns(true);
        Av.idctColumns(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        av_elems = (float[])Av.copy().elements();
        Av.dctColumns(true);
        Av.idctColumns(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
    }
    
    @Test
    public void testDctRows() {
        int rows = 512;
        int cols = 256;
        float[] a_1d = new float[rows * cols];
        for (int i = 0; i < rows * cols; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.dctRows(true);
        A.idctRows(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.dctRows(true);
        A.idctRows(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        FloatMatrix2D Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        float[] av_elems = (float[])Av.copy().elements();
        Av.dctRows(true);
        Av.idctRows(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        av_elems = (float[])Av.copy().elements();
        Av.dctRows(true);
        Av.idctRows(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
    }
    
    @Test
    public void testDht2() {
        int rows = 512;
        int cols = 256;
        float[] a_1d = new float[rows * cols];
        for (int i = 0; i < rows * cols; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.dht2();
        A.idht2(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.dht2();
        A.idht2(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        FloatMatrix2D Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        float[] av_elems = (float[])Av.copy().elements();
        Av.dht2();
        Av.idht2(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        av_elems = (float[])Av.copy().elements();
        Av.dht2();
        Av.idht2(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
    }

    @Test
    public void testDhtColumns() {
        int rows = 512;
        int cols = 256;
        float[] a_1d = new float[rows * cols];
        for (int i = 0; i < rows * cols; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.dhtColumns();
        A.idhtColumns(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.dhtColumns();
        A.idhtColumns(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        FloatMatrix2D Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        float[] av_elems = (float[])Av.copy().elements();
        Av.dhtColumns();
        Av.idhtColumns(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        av_elems = (float[])Av.copy().elements();
        Av.dhtColumns();
        Av.idhtColumns(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
    }
    
    @Test
    public void testDhtRows() {
        int rows = 512;
        int cols = 256;
        float[] a_1d = new float[rows * cols];
        for (int i = 0; i < rows * cols; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.dhtRows();
        A.idhtRows(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.dhtRows();
        A.idhtRows(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        FloatMatrix2D Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        float[] av_elems = (float[])Av.copy().elements();
        Av.dhtRows();
        Av.idhtRows(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        av_elems = (float[])Av.copy().elements();
        Av.dhtRows();
        Av.idhtRows(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
    }
    
    @Test
    public void testDst2() {
        int rows = 512;
        int cols = 256;
        float[] a_1d = new float[rows * cols];
        for (int i = 0; i < rows * cols; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.dst2(true);
        A.idst2(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.dst2(true);
        A.idst2(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        FloatMatrix2D Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        float[] av_elems = (float[])Av.copy().elements();
        Av.dst2(true);
        Av.idst2(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        av_elems = (float[])Av.copy().elements();
        Av.dst2(true);
        Av.idst2(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
    }
    
    @Test
    public void testDstColumns() {
        int rows = 512;
        int cols = 256;
        float[] a_1d = new float[rows * cols];
        for (int i = 0; i < rows * cols; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.dstColumns(true);
        A.idstColumns(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.dstColumns(true);
        A.idstColumns(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        FloatMatrix2D Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        float[] av_elems = (float[])Av.copy().elements();
        Av.dstColumns(true);
        Av.idstColumns(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        av_elems = (float[])Av.copy().elements();
        Av.dstColumns(true);
        Av.idstColumns(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
    }
    
    @Test
    public void testDstRows() {
        int rows = 512;
        int cols = 256;
        float[] a_1d = new float[rows * cols];
        for (int i = 0; i < rows * cols; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.dstRows(true);
        A.idstRows(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.dstRows(true);
        A.idstRows(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        FloatMatrix2D Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        float[] av_elems = (float[])Av.copy().elements();
        Av.dstRows(true);
        Av.idstRows(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        av_elems = (float[])Av.copy().elements();
        Av.dstRows(true);
        Av.idstRows(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
    }
    
    @Test
    public void testEqualsFloat() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        float value = 1;
        A.assign(value);
        boolean eq = A.equals(value);
        assertEquals(true, eq);
        eq = A.equals(2);
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(value);
        eq = A.equals(value);
        assertEquals(true, eq);
        eq = A.equals(2);
        assertEquals(false, eq);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(rows, cols);
        FloatMatrix2D Av = A.viewDice();
        A.assign(value);
        eq = Av.equals(value);
        assertEquals(true, eq);
        eq = Av.equals(2);
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        Av = A.viewDice();
        A.assign(value);
        eq = Av.equals(value);
        assertEquals(true, eq);
        eq = Av.equals(2);
        assertEquals(false, eq);
    }
    
    @Test
    public void testEqualsObject() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D B = new DenseFloatMatrix2D(b_2d);
        boolean eq = A.equals(A);
        assertEquals(true, eq);
        eq = A.equals(B);
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        B = new DenseFloatMatrix2D(b_2d);
        eq = A.equals(A);
        assertEquals(true, eq);
        eq = A.equals(B);
        assertEquals(false, eq);
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        B = new DenseFloatMatrix2D(b_2d);
        FloatMatrix2D Bv = B.viewDice();
        eq = Av.equals(Av);
        assertEquals(true, eq);
        eq = Av.equals(Bv);
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Av = A.viewDice();
        B = new DenseFloatMatrix2D(b_2d);
        Bv = B.viewDice();
        eq = Av.equals(Av);
        assertEquals(true, eq);
        eq = Av.equals(Bv);
        assertEquals(false, eq);
    }
    
    
    @Test
    public void testFft2() {
        int rows = 512;
        int cols = 256;
        float[] a_1d = new float[rows * cols];
        for (int i = 0; i < rows * cols; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.fft2();
        A.ifft2(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        A.fft2();
        A.ifft2(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        FloatMatrix2D Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        float[] av_elems = (float[])Av.copy().elements();
        Av.fft2();
        Av.ifft2(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        av_elems = (float[])Av.copy().elements();
        Av.fft2();
        Av.ifft2(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
    }
    
    @Test
    public void testForEachNonZero() {
        IntIntFloatFunction function = new IntIntFloatFunction() {
            public float apply(int first, int second, float third) {
                return (float)Math.sqrt(third);
            }
        };
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        float value = 1.5f;
        A.setQuick(0, 0, value);
        value = -3.3f;
        A.setQuick(3, 5, value);
        value = 222.3f;
        A.setQuick(11, 22, value);
        value = 0.1123f;
        A.setQuick(89, 67, value);
        float[] aElts = new float[rows * cols];
        System.arraycopy((float[]) A.elements(), 0, aElts, 0, rows * cols);
        A.forEachNonZero(function);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals((float)Math.sqrt(aElts[r * cols + c]), A.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        value = 1.5f;
        A.setQuick(0, 0, value);
        value = -3.3f;
        A.setQuick(3, 5, value);
        value = 222.3f;
        A.setQuick(11, 22, value);
        value = 0.1123f;
        A.setQuick(89, 67, value);
        aElts = new float[rows * cols];
        System.arraycopy((float[]) A.elements(), 0, aElts, 0, rows * cols);
        A.forEachNonZero(function);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals((float)Math.sqrt(aElts[r * cols + c]), A.getQuick(r, c), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(rows, cols);
        FloatMatrix2D Av = A.viewDice();
        value = 1.5f;
        Av.setQuick(0, 0, value);
        value = -3.3f;
        Av.setQuick(3, 5, value);
        value = 222.3f;
        Av.setQuick(11, 22, value);
        value = 0.1123f;
        Av.setQuick(89, 67, value);
        aElts = new float[rows * cols];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                aElts[r * rows + c] = Av.getQuick(r, c);
            }
        }
        Av.forEachNonZero(function);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals((float)Math.sqrt(aElts[r * rows + c]), Av.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        Av = A.viewDice();
        value = 1.5f;
        Av.setQuick(0, 0, value);
        value = -3.3f;
        Av.setQuick(3, 5, value);
        value = 222.3f;
        Av.setQuick(11, 22, value);
        value = 0.1123f;
        Av.setQuick(89, 67, value);
        aElts = new float[rows * cols];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                aElts[r * rows + c] = Av.getQuick(r, c);
            }
        }
        Av.forEachNonZero(function);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals((float)Math.sqrt(aElts[r * rows + c]), Av.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testGet() {
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_2d[r][c], A.get(r, c), tol);
            }
        }
    }

    @Test
    public void testGetFft2() {
        int rows = 512;
        int cols = 256;
        float[] a_1d = new float[rows * cols];
        for (int i = 0; i < rows * cols; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        FComplexMatrix2D Ac = A.getFft2();
        Ac.ifft2(true);
        float[] ac_elems = (float[])Ac.elements();
        for (int i = 0; i < rows*cols; i++) {
            assertEquals(a_1d[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        Ac = A.getFft2();
        Ac.ifft2(true);
        ac_elems = (float[])Ac.elements();
        for (int i = 0; i < rows*cols; i++) {
            assertEquals(a_1d[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        FloatMatrix2D Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        float[] av_elems = (float[])Av.copy().elements();
        Ac = Av.getFft2();
        Ac.ifft2(true);
        ac_elems = (float[])Ac.elements();
        for (int i = 0; i < av_elems.length; i++) {
            assertEquals(av_elems[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        av_elems = (float[])Av.copy().elements();
        Ac = Av.getFft2();
        Ac.ifft2(true);
        ac_elems = (float[])Ac.elements();
        for (int i = 0; i < av_elems.length; i++) {
            assertEquals(av_elems[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
    }
    
    @Test
	public void testGetFftColumns() {
		int rows = 512;
		int cols = 256;
		float[] a_1d = new float[rows * cols];
		for (int i = 0; i < rows * cols; i++) {
			a_1d[i] = (float)Math.random();
		}
		/* No view */
		// single thread
		ConcurrencyUtils.setNumberOfProcessors(1);
		FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
		A.assign(a_1d);
		FComplexMatrix2D B = A.getFftColumns();
		B.ifftColumns(true);
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				float[] elB = B.getQuick(r, c);
				Assert.assertEquals(a_1d[r * cols + c], elB[0], tol);
				Assert.assertEquals(0.0, elB[1], tol);
			}
		}
		// multiple threads
		ConcurrencyUtils.setNumberOfProcessors(nThreads);
		ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
		A = new DenseFloatMatrix2D(rows, cols);
		A.assign(a_1d);
		B = A.getFftColumns();
		B.ifftColumns(true);
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				float[] elB = B.getQuick(r, c);
				Assert.assertEquals(a_1d[r * cols + c], elB[0], tol);
				Assert.assertEquals(0.0, elB[1], tol);
			}
		}
		/* View */
		// single thread
		ConcurrencyUtils.setNumberOfProcessors(1);
		A = new DenseFloatMatrix2D(rows, cols);
		A.assign(a_1d);
		FloatMatrix2D Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
		float[] av_elems = (float[]) Av.copy().elements();
		B = Av.getFftColumns();
		B.ifftColumns(true);
		for (int r = 0; r < Av.rows(); r++) {
			for (int c = 0; c < Av.columns(); c++) {
				float[] elB = B.getQuick(r, c);
				Assert.assertEquals(av_elems[r * Av.columns() + c], elB[0], tol);
				Assert.assertEquals(0.0, elB[1], tol);
			}
		}
		// multiple threads
		ConcurrencyUtils.setNumberOfProcessors(nThreads);
		ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
		A = new DenseFloatMatrix2D(rows, cols);
		A.assign(a_1d);
		Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
		av_elems = (float[]) Av.copy().elements();
		B = Av.getFftColumns();
		B.ifftColumns(true);
		for (int r = 0; r < Av.rows(); r++) {
			for (int c = 0; c < Av.columns(); c++) {
				float[] elB = B.getQuick(r, c);
				Assert.assertEquals(av_elems[r * Av.columns() + c], elB[0], tol);
				Assert.assertEquals(0.0, elB[1], tol);
			}
		}
	}

	@Test
	public void testGetIfftColumns() {
		int rows = 512;
		int cols = 256;
		float[] a_1d = new float[rows * cols];
		for (int i = 0; i < rows * cols; i++) {
			a_1d[i] = (float)Math.random();
		}
		/* No view */
		// single thread
		ConcurrencyUtils.setNumberOfProcessors(1);
		FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
		A.assign(a_1d);
		FComplexMatrix2D B = A.getIfftColumns(true);
		B.fftColumns();
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				float[] elB = B.getQuick(r, c);
				Assert.assertEquals(a_1d[r * cols + c], elB[0], tol);
				Assert.assertEquals(0.0, elB[1], tol);
			}
		}
		// multiple threads
		ConcurrencyUtils.setNumberOfProcessors(nThreads);
		ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
		A = new DenseFloatMatrix2D(rows, cols);
		A.assign(a_1d);
		B = A.getIfftColumns(true);
		B.fftColumns();
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				float[] elB = B.getQuick(r, c);
				Assert.assertEquals(a_1d[r * cols + c], elB[0], tol);
				Assert.assertEquals(0.0, elB[1], tol);
			}
		}
		/* View */
		// single thread
		ConcurrencyUtils.setNumberOfProcessors(1);
		A = new DenseFloatMatrix2D(rows, cols);
		A.assign(a_1d);
		FloatMatrix2D Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
		float[] av_elems = (float[]) Av.copy().elements();
		B = Av.getIfftColumns(true);
		B.fftColumns();
		for (int r = 0; r < Av.rows(); r++) {
			for (int c = 0; c < Av.columns(); c++) {
				float[] elB = B.getQuick(r, c);
				Assert.assertEquals(av_elems[r * Av.columns() + c], elB[0], tol);
				Assert.assertEquals(0.0, elB[1], tol);
			}
		}
		// multiple threads
		ConcurrencyUtils.setNumberOfProcessors(nThreads);
		ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
		A = new DenseFloatMatrix2D(rows, cols);
		A.assign(a_1d);
		Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
		av_elems = (float[]) Av.copy().elements();
		B = Av.getIfftColumns(true);
		B.fftColumns();
		for (int r = 0; r < Av.rows(); r++) {
			for (int c = 0; c < Av.columns(); c++) {
				float[] elB = B.getQuick(r, c);
				Assert.assertEquals(av_elems[r * Av.columns() + c], elB[0], tol);
				Assert.assertEquals(0.0, elB[1], tol);
			}
		}
	}
	
	@Test
	public void testGetFftRows() {
		int rows = 512;
		int cols = 256;
		float[] a_1d = new float[rows * cols];
		for (int i = 0; i < rows * cols; i++) {
			a_1d[i] = (float)Math.random();
		}
		/* No view */
		// single thread
		ConcurrencyUtils.setNumberOfProcessors(1);
		FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
		A.assign(a_1d);
		FComplexMatrix2D B = A.getFftRows();
		B.ifftRows(true);
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				float[] elB = B.getQuick(r, c);
				Assert.assertEquals(a_1d[r * cols + c], elB[0], tol);
				Assert.assertEquals(0.0, elB[1], tol);
			}
		}
		// multiple threads
		ConcurrencyUtils.setNumberOfProcessors(nThreads);
		ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
		A = new DenseFloatMatrix2D(rows, cols);
		A.assign(a_1d);
		B = A.getFftRows();
		B.ifftRows(true);
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				float[] elB = B.getQuick(r, c);
				Assert.assertEquals(a_1d[r * cols + c], elB[0], tol);
				Assert.assertEquals(0.0, elB[1], tol);
			}
		}
		/* View */
		// single thread
		ConcurrencyUtils.setNumberOfProcessors(1);
		A = new DenseFloatMatrix2D(rows, cols);
		A.assign(a_1d);
		FloatMatrix2D Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
		float[] av_elems = (float[]) Av.copy().elements();
		B = Av.getFftRows();
		B.ifftRows(true);
		for (int r = 0; r < Av.rows(); r++) {
			for (int c = 0; c < Av.columns(); c++) {
				float[] elB = B.getQuick(r, c);
				Assert.assertEquals(av_elems[r * Av.columns() + c], elB[0], tol);
				Assert.assertEquals(0.0, elB[1], tol);
			}
		}
		// multiple threads
		ConcurrencyUtils.setNumberOfProcessors(nThreads);
		ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
		A = new DenseFloatMatrix2D(rows, cols);
		A.assign(a_1d);
		Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
		av_elems = (float[]) Av.copy().elements();
		B = Av.getFftRows();
		B.ifftRows(true);
		for (int r = 0; r < Av.rows(); r++) {
			for (int c = 0; c < Av.columns(); c++) {
				float[] elB = B.getQuick(r, c);
				Assert.assertEquals(av_elems[r * Av.columns() + c], elB[0], tol);
				Assert.assertEquals(0.0, elB[1], tol);
			}
		}
	}

	@Test
	public void testGetIfftRows() {
		int rows = 512;
		int cols = 256;
		float[] a_1d = new float[rows * cols];
		for (int i = 0; i < rows * cols; i++) {
			a_1d[i] = (float)Math.random();
		}
		/* No view */
		// single thread
		ConcurrencyUtils.setNumberOfProcessors(1);
		FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
		A.assign(a_1d);
		FComplexMatrix2D B = A.getIfftRows(true);
		B.fftRows();
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				float[] elB = B.getQuick(r, c);
				Assert.assertEquals(a_1d[r * cols + c], elB[0], tol);
				Assert.assertEquals(0.0, elB[1], tol);
			}
		}
		// multiple threads
		ConcurrencyUtils.setNumberOfProcessors(nThreads);
		ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
		A = new DenseFloatMatrix2D(rows, cols);
		A.assign(a_1d);
		B = A.getIfftRows(true);
		B.fftRows();
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				float[] elB = B.getQuick(r, c);
				Assert.assertEquals(a_1d[r * cols + c], elB[0], tol);
				Assert.assertEquals(0.0, elB[1], tol);
			}
		}
		/* View */
		// single thread
		ConcurrencyUtils.setNumberOfProcessors(1);
		A = new DenseFloatMatrix2D(rows, cols);
		A.assign(a_1d);
		FloatMatrix2D Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
		float[] av_elems = (float[]) Av.copy().elements();
		B = Av.getIfftRows(true);
		B.fftRows();
		for (int r = 0; r < Av.rows(); r++) {
			for (int c = 0; c < Av.columns(); c++) {
				float[] elB = B.getQuick(r, c);
				Assert.assertEquals(av_elems[r * Av.columns() + c], elB[0], tol);
				Assert.assertEquals(0.0, elB[1], tol);
			}
		}
		// multiple threads
		ConcurrencyUtils.setNumberOfProcessors(nThreads);
		ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
		A = new DenseFloatMatrix2D(rows, cols);
		A.assign(a_1d);
		Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
		av_elems = (float[]) Av.copy().elements();
		B = Av.getIfftRows(true);
		B.fftRows();
		for (int r = 0; r < Av.rows(); r++) {
			for (int c = 0; c < Av.columns(); c++) {
				float[] elB = B.getQuick(r, c);
				Assert.assertEquals(av_elems[r * Av.columns() + c], elB[0], tol);
				Assert.assertEquals(0.0, elB[1], tol);
			}
		}
	}

    @Test
    public void testGetIfft2() {
        int rows = 512;
        int cols = 256;
        float[] a_1d = new float[rows * cols];
        for (int i = 0; i < rows * cols; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        FComplexMatrix2D Ac = A.getIfft2(true);
        Ac.fft2();
        float[] ac_elems = (float[])Ac.elements();
        for (int i = 0; i < rows*cols; i++) {
            assertEquals(a_1d[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        Ac = A.getIfft2(true);
        Ac.fft2();
        ac_elems = (float[])Ac.elements();
        for (int i = 0; i < rows*cols; i++) {
            assertEquals(a_1d[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        FloatMatrix2D Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        float[] av_elems = (float[])Av.copy().elements();
        Ac = Av.getIfft2(true);
        Ac.fft2();
        ac_elems = (float[])Ac.elements();
        for (int i = 0; i < av_elems.length; i++) {
            assertEquals(av_elems[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(rows / 2 - 1, cols / 2 - 1, rows / 2, cols / 2);
        av_elems = (float[])Av.copy().elements();
        Ac = Av.getIfft2(true);
        Ac.fft2();
        ac_elems = (float[])Ac.elements();
        for (int i = 0; i < av_elems.length; i++) {
            assertEquals(av_elems[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
    }

    @Test
    public void testMaxLocation() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        A.setQuick(rows / 3, cols / 3, 0.7f);
        A.setQuick(rows / 2, cols / 2, 0.7f);
        float[] maxAndLoc = A.getMaxLocation();
        assertEquals(0.7f, maxAndLoc[0], tol);
        assertEquals(rows / 3, (int) maxAndLoc[1]);
        assertEquals(cols / 3, (int) maxAndLoc[2]);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.setQuick(rows / 3, cols / 3, 0.7f);
        A.setQuick(rows / 2, cols / 2, 0.7f);
        maxAndLoc = A.getMaxLocation();
        assertEquals(0.7f, maxAndLoc[0], tol);
        assertEquals(rows / 3, (int) maxAndLoc[1]);
        assertEquals(cols / 3, (int) maxAndLoc[2]);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(cols, rows);
        FloatMatrix2D Av = A.viewDice();
        Av.setQuick(rows / 3, cols / 3, 0.7f);
        Av.setQuick(rows / 2, cols / 2, 0.7f);
        maxAndLoc = Av.getMaxLocation();
        assertEquals(0.7f, maxAndLoc[0], tol);
        assertEquals(rows / 3, (int) maxAndLoc[1]);
        assertEquals(cols / 3, (int) maxAndLoc[2]);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(cols, rows);
        Av = A.viewDice();
        Av.setQuick(rows / 3, cols / 3, 0.7f);
        Av.setQuick(rows / 2, cols / 2, 0.7f);
        maxAndLoc = Av.getMaxLocation();
        assertEquals(0.7f, maxAndLoc[0], tol);
        assertEquals(rows / 3, (int) maxAndLoc[1]);
        assertEquals(cols / 3, (int) maxAndLoc[2]);
    }

    @Test
    public void testMinLocation() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        A.setQuick(rows / 3, cols / 3, -0.7f);
        A.setQuick(rows / 2, cols / 2, -0.7f);
        float[] minAndLoc = A.getMinLocation();
        assertEquals(-0.7f, minAndLoc[0], tol);
        assertEquals(rows / 3, (int) minAndLoc[1]);
        assertEquals(cols / 3, (int) minAndLoc[2]);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(rows, cols);
        A.setQuick(rows / 3, cols / 3, -0.7f);
        A.setQuick(rows / 2, cols / 2, -0.7f);
        minAndLoc = A.getMinLocation();
        assertEquals(-0.7f, minAndLoc[0], tol);
        assertEquals(rows / 3, (int) minAndLoc[1]);
        assertEquals(cols / 3, (int) minAndLoc[2]);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(cols, rows);
        FloatMatrix2D Av = A.viewDice();
        Av.setQuick(rows / 3, cols / 3, -0.7f);
        Av.setQuick(rows / 2, cols / 2, -0.7f);
        minAndLoc = Av.getMinLocation();
        assertEquals(-0.7f, minAndLoc[0], tol);
        assertEquals(rows / 3, (int) minAndLoc[1]);
        assertEquals(cols / 3, (int) minAndLoc[2]);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(cols, rows);
        Av = A.viewDice();
        Av.setQuick(rows / 3, cols / 3, -0.7f);
        Av.setQuick(rows / 2, cols / 2, -0.7f);
        minAndLoc = Av.getMinLocation();
        assertEquals(-0.7f, minAndLoc[0], tol);
        assertEquals(rows / 3, (int) minAndLoc[1]);
        assertEquals(cols / 3, (int) minAndLoc[2]);
    }

    @Test
    public void testGetNegativeValues() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        A.assign(FloatFunctions.mult(-1));
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        A.getNegativeValues(rowList, columnList, valueList);
        assertEquals(rows*cols, rowList.size());
        assertEquals(rows*cols, columnList.size());
        assertEquals(rows*cols, valueList.size());
        int idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(A.getQuick(rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                idx++;
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        A.assign(FloatFunctions.mult(-1));
        A.getNegativeValues(rowList, columnList, valueList);
        assertEquals(rows*cols, rowList.size());
        assertEquals(rows*cols, columnList.size());
        assertEquals(rows*cols, valueList.size());
        idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(A.getQuick(rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                idx++;
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        A.assign(FloatFunctions.mult(-1));
        FloatMatrix2D Av = A.viewDice();
        Av.getNegativeValues(rowList, columnList, valueList);
        assertEquals(rows*cols, rowList.size());
        assertEquals(rows*cols, columnList.size());
        assertEquals(rows*cols, valueList.size());
        idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(Av.getQuick(rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                idx++;
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        A.assign(FloatFunctions.mult(-1));
        Av = A.viewDice();
        Av.getNegativeValues(rowList, columnList, valueList);
        assertEquals(rows*cols, rowList.size());
        assertEquals(rows*cols, columnList.size());
        assertEquals(rows*cols, valueList.size());
        idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(Av.getQuick(rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                idx++;
            }
        }
    }

    @Test
    public void testGetNonZeros() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        A.getNonZeros(rowList, columnList, valueList);
        assertEquals(rows*cols, rowList.size());
        assertEquals(rows*cols, columnList.size());
        assertEquals(rows*cols, valueList.size());
        int idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(A.getQuick(rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                idx++;
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        A.getNonZeros(rowList, columnList, valueList);
        assertEquals(rows*cols, rowList.size());
        assertEquals(rows*cols, columnList.size());
        assertEquals(rows*cols, valueList.size());
        idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(A.getQuick(rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                idx++;
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        Av.getNonZeros(rowList, columnList, valueList);
        assertEquals(rows*cols, rowList.size());
        assertEquals(rows*cols, columnList.size());
        assertEquals(rows*cols, valueList.size());
        idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(Av.getQuick(rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                idx++;
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Av = A.viewDice();
        Av.getNonZeros(rowList, columnList, valueList);
        assertEquals(rows*cols, rowList.size());
        assertEquals(rows*cols, columnList.size());
        assertEquals(rows*cols, valueList.size());
        idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(Av.getQuick(rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                idx++;
            }
        }
    }

    @Test
    public void testGetPositiveValues() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        A.getPositiveValues(rowList, columnList, valueList);
        assertEquals(rows*cols, rowList.size());
        assertEquals(rows*cols, columnList.size());
        assertEquals(rows*cols, valueList.size());
        int idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(A.getQuick(rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                idx++;
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        A.getPositiveValues(rowList, columnList, valueList);
        assertEquals(rows*cols, rowList.size());
        assertEquals(rows*cols, columnList.size());
        assertEquals(rows*cols, valueList.size());
        idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(A.getQuick(rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                idx++;
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        Av.getPositiveValues(rowList, columnList, valueList);
        assertEquals(rows*cols, rowList.size());
        assertEquals(rows*cols, columnList.size());
        assertEquals(rows*cols, valueList.size());
        idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(Av.getQuick(rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                idx++;
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Av = A.viewDice();
        Av.getPositiveValues(rowList, columnList, valueList);
        assertEquals(rows*cols, rowList.size());
        assertEquals(rows*cols, columnList.size());
        assertEquals(rows*cols, valueList.size());
        idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(Av.getQuick(rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                idx++;
            }
        }
    }

    @Test
    public void testGetQuick() {
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_2d[r][c], A.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testSet() {
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        float elem = (float)Math.random();
        A.set(rows / 2, cols / 2, elem);
        assertEquals(elem, A.getQuick(rows / 2, cols / 2), tol);
    }

    @Test
    public void testSetQuick() {
        FloatMatrix2D A = new DenseFloatMatrix2D(rows, cols);
        float elem = (float)Math.random();
        A.setQuick(rows / 2, cols / 2, elem);
        assertEquals(elem, A.getQuick(rows / 2, cols / 2), tol);
    }

    @Test
    public void testToArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        float[][] array = A.toArray();
        AssertUtils.assertArrayEquals(a_2d, array, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        array = A.toArray();
        AssertUtils.assertArrayEquals(a_2d, array, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        array = Av.toArray();
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(a_2d[c][r], array[r][c], tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Av = A.viewDice();
        array = Av.toArray();
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(a_2d[c][r], array[r][c], tol);
            }
        }

    }

    @Test
    public void testVectorize() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix1D B = A.vectorize();
        int idx = 0;
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                assertEquals(A.getQuick(r, c), B.getQuick(idx++), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        B = A.vectorize();
        idx = 0;
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                assertEquals(A.getQuick(r, c), B.getQuick(idx++), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        B = Av.vectorize();
        idx = 0;
        for (int c = 0; c < rows; c++) {
            for (int r = 0; r < cols; r++) {
                assertEquals(Av.getQuick(r, c), B.getQuick(idx++), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Av = A.viewDice();
        B = Av.vectorize();
        idx = 0;
        for (int c = 0; c < rows; c++) {
            for (int r = 0; r < cols; r++) {
                assertEquals(Av.getQuick(r, c), B.getQuick(idx++), tol);
            }
        }
    }

    @Test
    public void testViewColumn() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix1D B = A.viewColumn(cols / 2);
        assertEquals(rows, B.size());
        for (int i = 0; i < rows; i++) {
            assertEquals(a_2d[i][cols / 2], B.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        B = A.viewColumn(cols / 2);
        assertEquals(rows, B.size());
        for (int i = 0; i < rows; i++) {
            assertEquals(a_2d[i][cols / 2], B.getQuick(i), tol);
        }
    }

    @Test
    public void testViewColumnFlip() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D B = A.viewColumnFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_2d[r][cols - 1 - c], B.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        B = A.viewColumnFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_2d[r][cols - 1 - c], B.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testViewDice() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D B = A.viewDice();
        assertEquals(A.rows(), B.columns());
        assertEquals(A.columns(), B.rows());
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(c, r), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        B = A.viewDice();
        assertEquals(A.rows(), B.columns());
        assertEquals(A.columns(), B.rows());
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(c, r), tol);
            }
        }
    }

    @Test
    public void testViewPart() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D B = A.viewPart(15, 11, 21, 27);
        for (int r = 0; r < 21; r++) {
            for (int c = 0; c < 27; c++) {
                assertEquals(A.getQuick(15 + r, 11 + c), B.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        B = A.viewPart(15, 11, 21, 27);
        for (int r = 0; r < 21; r++) {
            for (int c = 0; c < 27; c++) {
                assertEquals(A.getQuick(15 + r, 11 + c), B.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testViewRow() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix1D B = A.viewRow(rows / 2);
        assertEquals(cols, B.size());
        for (int i = 0; i < cols; i++) {
            assertEquals(a_2d[rows / 2][i], B.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        B = A.viewRow(rows / 2);
        assertEquals(cols, B.size());
        for (int i = 0; i < cols; i++) {
            assertEquals(a_2d[rows / 2][i], B.getQuick(i), tol);
        }

    }

    @Test
    public void testViewRowFlip() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D B = A.viewRowFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_2d[rows - 1 - r][c], B.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        B = A.viewRowFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_2d[rows - 1 - r][c], B.getQuick(r, c), tol);
            }
        }

    }

    @Test
    public void testViewSelectionFloatMatrix1DProcedure() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        final float value = 2;
        A.setQuick(rows / 4, 0, value);
        A.setQuick(rows / 2, 0, value);
        FloatMatrix2D B = A.viewSelection(new FloatMatrix1DProcedure() {
            public boolean apply(FloatMatrix1D element) {
                if ((float)Math.abs(element.getQuick(0) - value) < tol) {
                    return true;
                } else {
                    return false;
                }
            }
        });
        assertEquals(2, B.rows());
        assertEquals(A.columns(), B.columns());
        assertEquals(A.getQuick(rows / 4, 0), B.getQuick(0, 0), tol);
        assertEquals(A.getQuick(rows / 2, 0), B.getQuick(1, 0), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        A.setQuick(rows / 4, 0, value);
        A.setQuick(rows / 2, 0, value);
        B = A.viewSelection(new FloatMatrix1DProcedure() {
            public boolean apply(FloatMatrix1D element) {
                if ((float)Math.abs(element.getQuick(0) - value) < tol) {
                    return true;
                } else {
                    return false;
                }
            }
        });
        assertEquals(2, B.rows());
        assertEquals(A.columns(), B.columns());
        assertEquals(A.getQuick(rows / 4, 0), B.getQuick(0, 0), tol);
        assertEquals(A.getQuick(rows / 2, 0), B.getQuick(1, 0), tol);
    }

    @Test
    public void testViewSelectionIntArrayIntArray() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        int[] rowIndexes = new int[] { 5, 11, 22, 37, 101 };
        int[] colIndexes = new int[] { 2, 17, 32, 47, 99, 111 };
        FloatMatrix2D B = A.viewSelection(rowIndexes, colIndexes);
        assertEquals(rowIndexes.length, B.rows());
        assertEquals(colIndexes.length, B.columns());
        for (int r = 0; r < rowIndexes.length; r++) {
            for (int c = 0; c < colIndexes.length; c++) {
                assertEquals(A.getQuick(rowIndexes[r], colIndexes[c]), B.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        rowIndexes = new int[] { 5, 11, 22, 37, 101 };
        colIndexes = new int[] { 2, 17, 32, 47, 99, 111 };
        B = A.viewSelection(rowIndexes, colIndexes);
        assertEquals(rowIndexes.length, B.rows());
        assertEquals(colIndexes.length, B.columns());
        for (int r = 0; r < rowIndexes.length; r++) {
            for (int c = 0; c < colIndexes.length; c++) {
                assertEquals(A.getQuick(rowIndexes[r], colIndexes[c]), B.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testViewSorted() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix1D Col1 = A.viewColumn(1).copy();
        float[] col1 = (float[]) ((DenseFloatMatrix1D) Col1).elements();
        Arrays.sort(col1);
        FloatMatrix2D B = A.viewSorted(1);
        for (int r = 0; r < rows; r++) {
            assertEquals(col1[r], B.getQuick(r, 1), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Col1 = A.viewColumn(1).copy();
        col1 = (float[]) ((DenseFloatMatrix1D) Col1).elements();
        Arrays.sort(col1);
        B = A.viewSorted(1);
        for (int r = 0; r < rows; r++) {
            assertEquals(col1[r], B.getQuick(r, 1), tol);
        }
    }

    @Test
    public void testViewStrides() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        int rowStride = 3;
        int colStride = 5;
        FloatMatrix2D B = A.viewStrides(rowStride, colStride);
        for (int r = 0; r < B.rows(); r++) {
            for (int c = 0; c < B.columns(); c++) {
                assertEquals(A.getQuick(r * rowStride, c * colStride), B.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        B = A.viewStrides(rowStride, colStride);
        for (int r = 0; r < B.rows(); r++) {
            for (int c = 0; c < B.columns(); c++) {
                assertEquals(A.getQuick(r * rowStride, c * colStride), B.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testZMultFloatMatrix1DFloatMatrix1D() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix1D y = new DenseFloatMatrix1D(A.columns());
        for (int i = 0; i < y.size(); i++) {
            y.set(i, rand.nextFloat());
        }
        FloatMatrix1D z = new DenseFloatMatrix1D(A.rows());
        A.zMult(y, z);
        float[] tmpMatVec = new float[A.rows()];
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                tmpMatVec[r] += A.getQuick(r, c) * y.getQuick(c);
            }
        }
        for (int r = 0; r < A.rows(); r++) {
            assertEquals(tmpMatVec[r], z.getQuick(r), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        y = new DenseFloatMatrix1D(A.columns());
        for (int i = 0; i < y.size(); i++) {
            y.set(i, rand.nextFloat());
        }
        z = new DenseFloatMatrix1D(A.rows());
        A.zMult(y, z);
        tmpMatVec = new float[A.rows()];
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                tmpMatVec[r] += A.getQuick(r, c) * y.getQuick(c);
            }
        }
        for (int r = 0; r < A.rows(); r++) {
            assertEquals(tmpMatVec[r], z.getQuick(r), tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        y = new DenseFloatMatrix1D(Av.columns());
        for (int i = 0; i < y.size(); i++) {
            y.set(i, rand.nextFloat());
        }
        z = new DenseFloatMatrix1D(Av.rows());
        Av.zMult(y, z);
        tmpMatVec = new float[Av.rows()];
        for (int r = 0; r < Av.rows(); r++) {
            for (int c = 0; c < Av.columns(); c++) {
                tmpMatVec[r] += Av.getQuick(r, c) * y.getQuick(c);
            }
        }
        for (int r = 0; r < A.rows(); r++) {
            assertEquals(tmpMatVec[r], z.getQuick(r), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Av = A.viewDice();
        y = new DenseFloatMatrix1D(Av.columns());
        for (int i = 0; i < y.size(); i++) {
            y.set(i, rand.nextFloat());
        }
        z = new DenseFloatMatrix1D(Av.rows());
        Av.zMult(y, z);
        tmpMatVec = new float[Av.rows()];
        for (int r = 0; r < Av.rows(); r++) {
            for (int c = 0; c < Av.columns(); c++) {
                tmpMatVec[r] += Av.getQuick(r, c) * y.getQuick(c);
            }
        }
        for (int r = 0; r < A.rows(); r++) {
            assertEquals(tmpMatVec[r], z.getQuick(r), tol);
        }
    }

    @Test
    public void testZMultFloatMatrix1DFloatMatrix1DFloatFloatBoolean() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix1D y = new DenseFloatMatrix1D(A.columns());
        float alpha = 3;
        float beta = 5;
        for (int i = 0; i < y.size(); i++) {
            y.set(i, rand.nextFloat());
        }
        FloatMatrix1D z = new DenseFloatMatrix1D(A.rows());
        A.zMult(y, z, alpha, beta, false);
        float[] tmpMatVec = new float[A.rows()];
        float s;
        for (int r = 0; r < rows; r++) {
            s = 0;
            for (int c = 0; c < cols; c++) {
                s += A.getQuick(r, c) * y.getQuick(c);
            }
            tmpMatVec[r] = s * alpha + tmpMatVec[r] * beta;
        }
        for (int r = 0; r < A.rows(); r++) {
            assertEquals(tmpMatVec[r], z.getQuick(r), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        y = new DenseFloatMatrix1D(A.columns());
        for (int i = 0; i < y.size(); i++) {
            y.set(i, rand.nextFloat());
        }
        z = new DenseFloatMatrix1D(A.rows());
        A.zMult(y, z, alpha, beta, false);
        tmpMatVec = new float[A.rows()];
        for (int r = 0; r < rows; r++) {
            s = 0;
            for (int c = 0; c < cols; c++) {
                s += A.getQuick(r, c) * y.getQuick(c);
            }
            tmpMatVec[r] = s * alpha + tmpMatVec[r] * beta;
        }
        for (int r = 0; r < A.rows(); r++) {
            assertEquals(tmpMatVec[r], z.getQuick(r), tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        y = new DenseFloatMatrix1D(Av.columns());
        for (int i = 0; i < y.size(); i++) {
            y.set(i, rand.nextFloat());
        }
        z = new DenseFloatMatrix1D(Av.rows());
        Av.zMult(y, z, alpha, beta, false);
        tmpMatVec = new float[Av.rows()];
        for (int r = 0; r < cols; r++) {
            s = 0;
            for (int c = 0; c < rows; c++) {
                s += Av.getQuick(r, c) * y.getQuick(c);
            }
            tmpMatVec[r] = s * alpha + tmpMatVec[r] * beta;
        }
        for (int r = 0; r < Av.rows(); r++) {
            assertEquals(tmpMatVec[r], z.getQuick(r), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Av = A.viewDice();
        y = new DenseFloatMatrix1D(Av.columns());
        for (int i = 0; i < y.size(); i++) {
            y.set(i, rand.nextFloat());
        }
        z = new DenseFloatMatrix1D(Av.rows());
        Av.zMult(y, z, alpha, beta, false);
        tmpMatVec = new float[Av.rows()];
        for (int r = 0; r < cols; r++) {
            s = 0;
            for (int c = 0; c < rows; c++) {
                s += Av.getQuick(r, c) * y.getQuick(c);
            }
            tmpMatVec[r] = s * alpha + tmpMatVec[r] * beta;
        }
        for (int r = 0; r < Av.rows(); r++) {
            assertEquals(tmpMatVec[r], z.getQuick(r), tol);
        }
    }

    @Test
    public void testZMultFloatMatrix2DFloatMatrix2D() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D B = new DenseFloatMatrix2D(b_2d);
        B = B.viewDice().copy();
        FloatMatrix2D C = new DenseFloatMatrix2D(rows, rows);
        A.zMult(B, C);
        float[][] tmpMatMat = new float[rows][rows];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < rows; c++) {
                for (int k = 0; k < cols; k++) {
                    tmpMatMat[c][r] += A.getQuick(c, k) * B.getQuick(k, r);
                }
            }
        }
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(tmpMatMat[r][c], C.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        B = new DenseFloatMatrix2D(b_2d);
        B = B.viewDice().copy();
        C = new DenseFloatMatrix2D(rows, rows);
        A.zMult(B, C);
        tmpMatMat = new float[rows][rows];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < rows; c++) {
                for (int k = 0; k < cols; k++) {
                    tmpMatMat[c][r] += A.getQuick(c, k) * B.getQuick(k, r);
                }
            }
        }
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(tmpMatMat[r][c], C.getQuick(r, c), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        B = new DenseFloatMatrix2D(b_2d);
        B = B.viewDice().copy();
        FloatMatrix2D Bv = B.viewDice();
        C = new DenseFloatMatrix2D(cols, cols);
        FloatMatrix2D Cv = C.viewDice();
        Av.zMult(Bv, Cv);
        tmpMatMat = new float[cols][cols];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < cols; c++) {
                for (int k = 0; k < rows; k++) {
                    tmpMatMat[c][r] += Av.getQuick(c, k) * Bv.getQuick(k, r);
                }
            }
        }
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(tmpMatMat[r][c], Cv.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Av = A.viewDice();
        B = new DenseFloatMatrix2D(b_2d);
        B = B.viewDice().copy();
        Bv = B.viewDice();
        C = new DenseFloatMatrix2D(cols, cols);
        Cv = C.viewDice();
        Av.zMult(Bv, Cv);
        tmpMatMat = new float[cols][cols];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < cols; c++) {
                for (int k = 0; k < rows; k++) {
                    tmpMatMat[c][r] += Av.getQuick(c, k) * Bv.getQuick(k, r);
                }
            }
        }
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(tmpMatMat[r][c], Cv.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testZMultFloatMatrix2DFloatMatrix2DFloatFloatBooleanBoolean() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D B = new DenseFloatMatrix2D(b_2d);
        B = B.viewDice().copy();
        FloatMatrix2D C = new DenseFloatMatrix2D(rows, rows);
        float alpha = 3;
        float beta = 5;
        A.zMult(B, C, alpha, beta, false, false);
        float[][] tmpMatMat = new float[rows][rows];
        float s;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < rows; c++) {
                s = 0;
                for (int k = 0; k < cols; k++) {
                    s += A.getQuick(c, k) * B.getQuick(k, r);
                }
                tmpMatMat[c][r] = s * alpha + tmpMatMat[c][r] * beta;
            }
        }
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(tmpMatMat[r][c], C.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        B = new DenseFloatMatrix2D(b_2d);
        B = B.viewDice().copy();
        C = new DenseFloatMatrix2D(rows, rows);
        A.zMult(B, C, alpha, beta, false, false);
        tmpMatMat = new float[rows][rows];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < rows; c++) {
                s = 0;
                for (int k = 0; k < cols; k++) {
                    s += A.getQuick(c, k) * B.getQuick(k, r);
                }
                tmpMatMat[c][r] = s * alpha + tmpMatMat[c][r] * beta;
            }
        }
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(tmpMatMat[r][c], C.getQuick(r, c), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        B = new DenseFloatMatrix2D(b_2d);
        B = B.viewDice().copy();
        FloatMatrix2D Bv = B.viewDice();
        C = new DenseFloatMatrix2D(cols, cols);
        FloatMatrix2D Cv = C.viewDice();
        Av.zMult(Bv, Cv, alpha, beta, false, false);
        tmpMatMat = new float[cols][cols];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < cols; c++) {
                s = 0;
                for (int k = 0; k < rows; k++) {
                    s += Av.getQuick(c, k) * Bv.getQuick(k, r);
                }
                tmpMatMat[c][r] = s * alpha + tmpMatMat[c][r] * beta;
            }
        }
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(tmpMatMat[r][c], Cv.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Av = A.viewDice();
        B = new DenseFloatMatrix2D(b_2d);
        B = B.viewDice().copy();
        Bv = B.viewDice();
        C = new DenseFloatMatrix2D(cols, cols);
        Cv = C.viewDice();
        Av.zMult(Bv, Cv, alpha, beta, false, false);
        tmpMatMat = new float[cols][cols];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < cols; c++) {
                s = 0;
                for (int k = 0; k < rows; k++) {
                    s += Av.getQuick(c, k) * Bv.getQuick(k, r);
                }
                tmpMatMat[c][r] = s * alpha + tmpMatMat[c][r] * beta;
            }
        }
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(tmpMatMat[r][c], Cv.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testZSum() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix2D A = new DenseFloatMatrix2D(a_2d);
        float aSum = A.zSum();
        float tmpSum = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmpSum += a_2d[r][c];
            }
        }
        assertEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        aSum = A.zSum();
        tmpSum = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmpSum += a_2d[r][c];
            }
        }
        assertEquals(tmpSum, aSum, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix2D(a_2d);
        FloatMatrix2D Av = A.viewDice();
        aSum = Av.zSum();
        tmpSum = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmpSum += a_2d[r][c];
            }
        }
        assertEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix2D(a_2d);
        Av = A.viewDice();
        aSum = Av.zSum();
        tmpSum = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmpSum += a_2d[r][c];
            }
        }
        assertEquals(tmpSum, aSum, tol);
    }

}
