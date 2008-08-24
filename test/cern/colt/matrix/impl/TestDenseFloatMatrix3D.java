package cern.colt.matrix.impl;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.Random;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import cern.colt.function.FloatProcedure;
import cern.colt.list.FloatArrayList;
import cern.colt.list.IntArrayList;
import cern.colt.matrix.FComplexMatrix3D;
import cern.colt.matrix.FloatMatrix1D;
import cern.colt.matrix.FloatMatrix2D;
import cern.colt.matrix.FloatMatrix2DProcedure;
import cern.colt.matrix.FloatMatrix3D;
import cern.jet.math.FloatFunctions;
import edu.emory.mathcs.utils.AssertUtils;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public class TestDenseFloatMatrix3D {
    private static final int slices = 5;

    private static final int rows = 53;

    private static final int cols = 57;

    private static final float tol = 1e-1f;

    private static final int nThreads = 3;

    private static final int nThreadsBegin = 1;

    private float[][][] a_3d, b_3d;

    private float[] a_1d, b_1d;

    private Random rand;

    @Before
    public void setUpBeforeClass() throws Exception {
        rand = new Random(0);
        a_1d = new float[slices * rows * cols];
        a_3d = new float[slices][rows][cols];
        int idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    a_3d[s][r][c] = rand.nextFloat();
                    a_1d[idx++] = a_3d[s][r][c];
                }
            }
        }
        b_1d = new float[slices * rows * cols];
        b_3d = new float[slices][rows][cols];
        idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    b_3d[s][r][c] = rand.nextFloat();
                    b_1d[idx++] = b_3d[s][r][c];
                }
            }
        }
    }

    @After
    public void tearDownAfterClass() throws Exception {
        a_1d = null;
        a_3d = null;
        b_1d = null;
        b_3d = null;
        System.gc();
    }

    @Test
    public void testAggregateFloatFloatFunctionFloatFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        float aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.sqrt);
        float tmpSum = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    tmpSum += Math.sqrt(a_3d[s][r][c]);
                }
            }
        }
        assertEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.sqrt);
        tmpSum = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    tmpSum += Math.sqrt(a_3d[s][r][c]);
                }
            }
        }
        assertEquals(tmpSum, aSum, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.sqrt);
        tmpSum = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    tmpSum += Math.sqrt(a_3d[s][r][c]);
                }
            }
        }
        assertEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        Av = A.viewDice(2, 1, 0);
        aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.sqrt);
        tmpSum = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    tmpSum += Math.sqrt(a_3d[s][r][c]);
                }
            }
        }

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
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        float aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.sqrt, procedure);
        float tmpSum = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    if (Math.abs(a_3d[s][r][c]) > 0.2) {
                        tmpSum += Math.sqrt(a_3d[s][r][c]);
                    }
                }
            }
        }
        assertEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.sqrt, procedure);
        tmpSum = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    if (Math.abs(a_3d[s][r][c]) > 0.2) {
                        tmpSum += Math.sqrt(a_3d[s][r][c]);
                    }
                }
            }
        }
        assertEquals(tmpSum, aSum, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.sqrt, procedure);
        tmpSum = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    if (Math.abs(a_3d[s][r][c]) > 0.2) {
                        tmpSum += Math.sqrt(a_3d[s][r][c]);
                    }
                }
            }
        }
        assertEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        Av = A.viewDice(2, 1, 0);
        aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.sqrt, procedure);
        tmpSum = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    if (Math.abs(a_3d[s][r][c]) > 0.2) {
                        tmpSum += Math.sqrt(a_3d[s][r][c]);
                    }
                }
            }
        }
    }

    @Test
    public void testAggregateFloatFloatFunctionFloatFunctionIntArrayListIntArrayList() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    sliceList.add(s);
                    rowList.add(r);
                    columnList.add(c);
                }
            }
        }
        float aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.square, sliceList, rowList, columnList);
        float tmpSum = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    tmpSum += a_3d[s][r][c] * a_3d[s][r][c];
                }
            }
        }
        assertEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.square, sliceList, rowList, columnList);
        tmpSum = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    tmpSum += a_3d[s][r][c] * a_3d[s][r][c];
                }
            }
        }
        assertEquals(tmpSum, aSum, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.square, columnList, rowList, sliceList);
        tmpSum = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    tmpSum += a_3d[s][r][c] * a_3d[s][r][c];
                }
            }
        }
        assertEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.square, columnList, rowList, sliceList);
        tmpSum = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    tmpSum += a_3d[s][r][c] * a_3d[s][r][c];
                }
            }
        }
        assertEquals(tmpSum, aSum, tol);
    }

    @Test
    public void testAggregateFloatMatrix3DFloatFloatFunctionFloatFloatFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D B = new DenseFloatMatrix3D(b_3d);
        float sumMult = A.aggregate(B, FloatFunctions.plus, FloatFunctions.mult);
        float tmpSumMult = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    tmpSumMult += a_3d[s][r][c] * b_3d[s][r][c];
                }
            }
        }
        assertEquals(tmpSumMult, sumMult, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        B = new DenseFloatMatrix3D(b_3d);
        sumMult = A.aggregate(B, FloatFunctions.plus, FloatFunctions.mult);
        tmpSumMult = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    tmpSumMult += a_3d[s][r][c] * b_3d[s][r][c];
                }
            }
        }
        assertEquals(tmpSumMult, sumMult, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        B = new DenseFloatMatrix3D(b_3d);
        FloatMatrix3D Bv = B.viewDice(2, 1, 0);
        sumMult = Av.aggregate(Bv, FloatFunctions.plus, FloatFunctions.mult);
        tmpSumMult = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    tmpSumMult += a_3d[s][r][c] * b_3d[s][r][c];
                }
            }
        }
        assertEquals(tmpSumMult, sumMult, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        Av = A.viewDice(2, 1, 0);
        B = new DenseFloatMatrix3D(b_3d);
        Bv = B.viewDice(2, 1, 0);
        sumMult = Av.aggregate(Bv, FloatFunctions.plus, FloatFunctions.mult);
        tmpSumMult = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    tmpSumMult += a_3d[s][r][c] * b_3d[s][r][c];
                }
            }
        }
        assertEquals(tmpSumMult, sumMult, tol);
    }

    @Test
    public void testAssignFloat() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        float value = (float)Math.random();
        A.assign(value);
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(value, A.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(value);
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(value, A.getQuick(s, r, c), tol);
                }
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        value = (float)Math.random();
        Av.assign(value);
        for (int s = 0; s < cols; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < slices; c++) {
                    assertEquals(value, Av.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        Av = A.viewDice(2, 1, 0);
        value = (float)Math.random();
        Av.assign(value);
        for (int s = 0; s < cols; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < slices; c++) {
                    assertEquals(value, Av.getQuick(s, r, c), tol);
                }
            }
        }
    }

    @Test
    public void testAssignFloatArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        float[] aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a_1d, aElts, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a_1d, aElts, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        Av.assign(a_1d);
        for (int s = 0; s < cols; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < slices; c++) {
                    assertEquals(a_1d[s * rows * slices + r * slices + c], Av.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        Av = A.viewDice(2, 1, 0);
        Av.assign(a_1d);
        for (int s = 0; s < cols; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < slices; c++) {
                    assertEquals(a_1d[s * rows * slices + r * slices + c], Av.getQuick(s, r, c), tol);
                }
            }
        }

    }

    @Test
    public void testAssignFloatArrayArrayArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_3d);
        float[] aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a_1d, aElts, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_3d);
        aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a_1d, aElts, tol);
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(cols, rows, slices);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        Av.assign(a_3d);
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(a_3d[s][r][c], Av.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(cols, rows, slices);
        Av = A.viewDice(2, 1, 0);
        Av.assign(a_3d);
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(a_3d[s][r][c], Av.getQuick(s, r, c), tol);
                }
            }
        }

    }

    @Test
    public void testAssignFloatFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        A.assign(FloatFunctions.acos);
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(Math.acos(a_3d[s][r][c]), A.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        A.assign(FloatFunctions.acos);
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(Math.acos(a_3d[s][r][c]), A.getQuick(s, r, c), tol);
                }
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        Av.assign(FloatFunctions.acos);
        for (int s = 0; s < cols; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < slices; c++) {
                    assertEquals(Math.acos(a_3d[c][r][s]), Av.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        Av = A.viewDice(2, 1, 0);
        Av.assign(FloatFunctions.acos);
        for (int s = 0; s < cols; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < slices; c++) {
                    assertEquals(Math.acos(a_3d[c][r][s]), Av.getQuick(s, r, c), tol);
                }
            }
        }
    }

    @Test
    public void testAssignFloatMatrix3D() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        FloatMatrix3D B = new DenseFloatMatrix3D(a_3d);
        A.assign(B);
        float[] aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a_1d, aElts, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        B = new DenseFloatMatrix3D(a_3d);
        A.assign(B);
        aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a_1d, aElts, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        B = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D Bv = B.viewDice(2, 1, 0);
        Av.assign(Bv);
        for (int s = 0; s < cols; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < slices; c++) {
                    assertEquals(a_3d[c][r][s], Av.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        Av = A.viewDice(2, 1, 0);
        B = new DenseFloatMatrix3D(a_3d);
        Bv = B.viewDice(2, 1, 0);
        Av.assign(Bv);
        for (int s = 0; s < cols; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < slices; c++) {
                    assertEquals(a_3d[c][r][s], Av.getQuick(s, r, c), tol);
                }
            }
        }
    }

    @Test
    public void testAssignFloatMatrix3DFloatFloatFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D B = new DenseFloatMatrix3D(b_3d);
        A.assign(B, FloatFunctions.div);
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(a_3d[s][r][c] / b_3d[s][r][c], A.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        B = new DenseFloatMatrix3D(b_3d);
        A.assign(B, FloatFunctions.div);
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(a_3d[s][r][c] / b_3d[s][r][c], A.getQuick(s, r, c), tol);
                }
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        B = new DenseFloatMatrix3D(b_3d);
        FloatMatrix3D Bv = B.viewDice(2, 1, 0);
        Av.assign(Bv, FloatFunctions.div);
        for (int s = 0; s < cols; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < slices; c++) {
                    assertEquals(a_3d[c][r][s] / b_3d[c][r][s], Av.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        Av = A.viewDice(2, 1, 0);
        B = new DenseFloatMatrix3D(b_3d);
        Bv = B.viewDice(2, 1, 0);
        Av.assign(Bv, FloatFunctions.div);
        for (int s = 0; s < cols; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < slices; c++) {
                    assertEquals(a_3d[c][r][s] / b_3d[c][r][s], Av.getQuick(s, r, c), tol);
                }
            }
        }
    }

    @Test
    public void testAssignFloatMatrix2DFloatFloatFunctionIntArrayListIntArrayList() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D B = new DenseFloatMatrix3D(b_3d);
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    sliceList.add(s);
                    rowList.add(r);
                    columnList.add(c);
                }
            }
        }
        A.assign(B, FloatFunctions.div, sliceList, rowList, columnList);
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(a_3d[s][r][c] / b_3d[s][r][c], A.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        B = new DenseFloatMatrix3D(b_3d);
        A.assign(B, FloatFunctions.div, sliceList, rowList, columnList);
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(a_3d[s][r][c] / b_3d[s][r][c], A.getQuick(s, r, c), tol);
                }
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        B = new DenseFloatMatrix3D(b_3d);
        FloatMatrix3D Bv = B.viewDice(2, 1, 0);
        Av.assign(Bv, FloatFunctions.div, columnList, rowList, sliceList);
        for (int s = 0; s < cols; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < slices; c++) {
                    assertEquals(a_3d[c][r][s] / b_3d[c][r][s], Av.getQuick(s, r, c), tol);
                }
            }
        }

        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        Av = A.viewDice(2, 1, 0);
        B = new DenseFloatMatrix3D(b_3d);
        Bv = B.viewDice(2, 1, 0);
        Av.assign(Bv, FloatFunctions.div, columnList, rowList, sliceList);
        for (int s = 0; s < cols; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < slices; c++) {
                    assertEquals(a_3d[c][r][s] / b_3d[c][r][s], Av.getQuick(s, r, c), tol);
                }
            }
        }
    }
    
    @Test
	public void testAssignFloatProcedureFloat() {
		FloatProcedure procedure = new FloatProcedure() {
			public boolean apply(float element) {
				if (Math.abs(element) > 0.1) {
					return true;
				} else {
					return false;
				}
			}
		};
		/* No view */
		// single thread
		ConcurrencyUtils.setNumberOfProcessors(1);
		FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
		FloatMatrix3D B = A.copy();
		A.assign(procedure, -1);
		for (int s = 0; s < slices; s++) {
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) {
					if (Math.abs(B.getQuick(s, r, c)) > 0.1) {
						B.setQuick(s, r, c, -1);
					}
				}
			}
		}
		AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
		// multiple threads
		ConcurrencyUtils.setNumberOfProcessors(nThreads);
		ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
		A = new DenseFloatMatrix3D(a_3d);
		B = A.copy();
		A.assign(procedure, -1);
		for (int s = 0; s < slices; s++) {
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) {
					if (Math.abs(B.getQuick(s, r, c)) > 0.1) {
						B.setQuick(s, r, c, -1);
					}
				}
			}
		}
		AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
		/* View */
		// single thread
		ConcurrencyUtils.setNumberOfProcessors(1);
		A = new DenseFloatMatrix3D(a_3d);
		FloatMatrix3D Av = A.viewDice(2, 1, 0);
		B = A.copy();
		FloatMatrix3D Bv = B.viewDice(2, 1, 0);
		Av.assign(procedure, -1);
		for (int s = 0; s < cols; s++) {
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < slices; c++) {
					if (Math.abs(Bv.getQuick(s, r, c)) > 0.1) {
						Bv.setQuick(s, r, c, -1);
					}
				}
			}
		}
		for (int s = 0; s < cols; s++) {
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < slices; c++) {
					assertEquals(Bv.getQuick(s, r, c), Av.getQuick(s, r, c), tol);
				}
			}
		}
		// multiple threads
		ConcurrencyUtils.setNumberOfProcessors(nThreads);
		ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
		A = new DenseFloatMatrix3D(a_3d);
		Av = A.viewDice(2, 1, 0);
		B = A.copy();
		Bv = B.viewDice(2, 1, 0);
		Av.assign(procedure, -1);
		for (int s = 0; s < cols; s++) {
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < slices; c++) {
					if (Math.abs(Bv.getQuick(s, r, c)) > 0.1) {
						Bv.setQuick(s, r, c, -1);
					}
				}
			}
		}
		for (int s = 0; s < cols; s++) {
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < slices; c++) {
					assertEquals(Bv.getQuick(s, r, c), Av.getQuick(s, r, c), tol);
				}
			}
		}
	}
	
	@Test
	public void testAssignFloatProcedureFloatFunction() {
		FloatProcedure procedure = new FloatProcedure() {
			public boolean apply(float element) {
				if (Math.abs(element) > 0.1) {
					return true;
				} else {
					return false;
				}
			}
		};
		/* No view */
		// single thread
		ConcurrencyUtils.setNumberOfProcessors(1);
		FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
		FloatMatrix3D B = A.copy();
		A.assign(procedure, FloatFunctions.tan);
		for (int s = 0; s < slices; s++) {
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) {
					if (Math.abs(B.getQuick(s, r, c)) > 0.1) {
						B.setQuick(s, r, c, (float)Math.tan(B.getQuick(s, r, c)));
					}
				}
			}
		}
		AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
		// multiple threads
		ConcurrencyUtils.setNumberOfProcessors(nThreads);
		ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
		A = new DenseFloatMatrix3D(a_3d);
		B = A.copy();
		A.assign(procedure, FloatFunctions.tan);
		for (int s = 0; s < slices; s++) {
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < cols; c++) {
					if (Math.abs(B.getQuick(s, r, c)) > 0.1) {
						B.setQuick(s, r, c, (float)Math.tan(B.getQuick(s, r, c)));
					}
				}
			}
		}
		AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
		/* View */
		// single thread
		ConcurrencyUtils.setNumberOfProcessors(1);
		A = new DenseFloatMatrix3D(a_3d);
		FloatMatrix3D Av = A.viewDice(2, 1, 0);
		B = A.copy();
		FloatMatrix3D Bv = B.viewDice(2, 1, 0);
		Av.assign(procedure, FloatFunctions.tan);
		for (int s = 0; s < cols; s++) {
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < slices; c++) {
					if (Math.abs(Bv.getQuick(s, r, c)) > 0.1) {
						Bv.setQuick(s, r, c, (float)Math.tan(Bv.getQuick(s, r, c)));
					}
				}
			}
		}
		for (int s = 0; s < cols; s++) {
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < slices; c++) {
					assertEquals(Bv.getQuick(s, r, c), Av.getQuick(s, r, c), tol);
				}
			}
		}
		// multiple threads
		ConcurrencyUtils.setNumberOfProcessors(nThreads);
		ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
		A = new DenseFloatMatrix3D(a_3d);
		Av = A.viewDice(2, 1, 0);
		B = A.copy();
		Bv = B.viewDice(2, 1, 0);
		Av.assign(procedure, FloatFunctions.tan);
		for (int s = 0; s < cols; s++) {
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < slices; c++) {
					if (Math.abs(Bv.getQuick(s, r, c)) > 0.1) {
						Bv.setQuick(s, r, c, (float)Math.tan(Bv.getQuick(s, r, c)));
					}
				}
			}
		}
		for (int s = 0; s < cols; s++) {
			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < slices; c++) {
					assertEquals(Bv.getQuick(s, r, c), Av.getQuick(s, r, c), tol);
				}
			}
		}
	}

    @Test
    public void testCardinality() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        int card = A.cardinality();
        assertEquals(slices * rows * cols, card);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        card = A.cardinality();
        assertEquals(slices * rows * cols, card);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        card = Av.cardinality();
        assertEquals(slices * rows * cols, card);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        Av = A.viewDice(2, 1, 0);
        card = Av.cardinality();
        assertEquals(slices * rows * cols, card);
    }

    @Test
    public void testDct3() {
        int slices = 64;
        int rows = 128;
        int cols = 64;
        float[] a_1d = new float[slices * rows * cols];
        for (int i = 0; i < slices * rows * cols; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        A.dct3(true);
        A.idct3(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        A.dct3(true);
        A.idct3(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        FloatMatrix3D Av = A.viewPart(slices / 2 - 1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        float[] av_elems = (float[]) Av.copy().elements();
        Av.dct3(true);
        Av.idct3(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(slices / 2 - 1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        av_elems = (float[]) Av.copy().elements();
        Av.dct3(true);
        Av.idct3(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
    }
    
    @Test
    public void testDct2Slices() {
        int slices = 64;
        int rows = 128;
        int cols = 64;
        float[] a_1d = new float[slices * rows * cols];
        for (int i = 0; i < slices * rows * cols; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        A.dct2Slices(true);
        A.idct2Slices(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        A.dct2Slices(true);
        A.idct2Slices(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        FloatMatrix3D Av = A.viewPart(slices / 2 - 1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        float[] av_elems = (float[]) Av.copy().elements();
        Av.dct2Slices(true);
        Av.idct2Slices(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(slices / 2 - 1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        av_elems = (float[]) Av.copy().elements();
        Av.dct2Slices(true);
        Av.idct2Slices(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
    }

    @Test
    public void testDht3() {
        int slices = 64;
        int rows = 128;
        int cols = 64;
        float[] a_1d = new float[slices * rows * cols];
        for (int i = 0; i < slices * rows * cols; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        A.dht3();
        A.idht3(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        A.dht3();
        A.idht3(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        FloatMatrix3D Av = A.viewPart(slices / 2 - 1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        float[] av_elems = (float[]) Av.copy().elements();
        Av.dht3();
        Av.idht3(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(slices / 2 - 1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        av_elems = (float[]) Av.copy().elements();
        Av.dht3();
        Av.idht3(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
    }
    
    @Test
    public void testDht2Slices() {
        int slices = 64;
        int rows = 128;
        int cols = 64;
        float[] a_1d = new float[slices * rows * cols];
        for (int i = 0; i < slices * rows * cols; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        A.dht2Slices();
        A.idht2Slices(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        A.dht2Slices();
        A.idht2Slices(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        FloatMatrix3D Av = A.viewPart(slices / 2 - 1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        float[] av_elems = (float[]) Av.copy().elements();
        Av.dht2Slices();
        Av.idht2Slices(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(slices / 2 - 1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        av_elems = (float[]) Av.copy().elements();
        Av.dht2Slices();
        Av.idht2Slices(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
    }
    
    @Test
    public void testDst3() {
        int slices = 64;
        int rows = 128;
        int cols = 64;
        float[] a_1d = new float[slices * rows * cols];
        for (int i = 0; i < slices * rows * cols; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        A.dst3(true);
        A.idst3(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        A.dst3(true);
        A.idst3(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        FloatMatrix3D Av = A.viewPart(slices / 2 - 1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        float[] av_elems = (float[]) Av.copy().elements();
        Av.dst3(true);
        Av.idst3(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(slices / 2 - 1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        av_elems = (float[]) Av.copy().elements();
        Av.dst3(true);
        Av.idst3(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
    }
    
    @Test
    public void testDst2Slices() {
        int slices = 64;
        int rows = 128;
        int cols = 64;
        float[] a_1d = new float[slices * rows * cols];
        for (int i = 0; i < slices * rows * cols; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        A.dst2Slices(true);
        A.idst2Slices(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        A.dst2Slices(true);
        A.idst2Slices(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        FloatMatrix3D Av = A.viewPart(slices / 2 - 1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        float[] av_elems = (float[]) Av.copy().elements();
        Av.dst2Slices(true);
        Av.idst2Slices(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(slices / 2 - 1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        av_elems = (float[]) Av.copy().elements();
        Av.dst2Slices(true);
        Av.idst2Slices(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
    }
    
    @Test
    public void testFft3() {
        int slices = 64;
        int rows = 128;
        int cols = 64;
        float[] a_1d = new float[slices * rows * cols];
        for (int i = 0; i < slices * rows * cols; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        A.fft3();
        A.ifft3(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        A.fft3();
        A.ifft3(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        FloatMatrix3D Av = A.viewPart(slices / 2 - 1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        float[] av_elems = (float[]) Av.copy().elements();
        Av.fft3();
        Av.ifft3(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(slices / 2 - 1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        av_elems = (float[]) Av.copy().elements();
        Av.fft3();
        Av.ifft3(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
    }
    
    @Test
    public void testGetFft3() {
    	 int slices = 64;
         int rows = 128;
         int cols = 64;
         float[] a_1d = new float[slices * rows * cols];
         for (int i = 0; i < slices * rows * cols; i++) {
             a_1d[i] = (float)Math.random();
         }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        FComplexMatrix3D Ac = A.getFft3();
        Ac.ifft3(true);
        float[] ac_elems = (float[])Ac.elements();
        for (int i = 0; i < slices*rows*cols; i++) {
            assertEquals(a_1d[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        Ac = A.getFft3();
        Ac.ifft3(true);
        ac_elems = (float[])Ac.elements();
        for (int i = 0; i < slices*rows*cols; i++) {
            assertEquals(a_1d[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        FloatMatrix3D Av = A.viewPart(slices / 2  - 1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        float[] av_elems = (float[])Av.copy().elements();
        Ac = Av.getFft3();
        Ac.ifft3(true);
        ac_elems = (float[])Ac.elements();
        for (int i = 0; i < av_elems.length; i++) {
            assertEquals(av_elems[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(slices / 2 -1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        av_elems = (float[])Av.copy().elements();
        Ac = Av.getFft3();
        Ac.ifft3(true);
        ac_elems = (float[])Ac.elements();
        for (int i = 0; i < av_elems.length; i++) {
            assertEquals(av_elems[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
    }
    
    @Test
    public void testGetFft2Slices() {
    	 int slices = 64;
         int rows = 128;
         int cols = 64;
         float[] a_1d = new float[slices * rows * cols];
         for (int i = 0; i < slices * rows * cols; i++) {
             a_1d[i] = (float)Math.random();
         }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        FComplexMatrix3D Ac = A.getFft2Slices();
        Ac.ifft2Slices(true);
        float[] ac_elems = (float[])Ac.elements();
        for (int i = 0; i < slices*rows*cols; i++) {
            assertEquals(a_1d[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        Ac = A.getFft2Slices();
        Ac.ifft2Slices(true);
        ac_elems = (float[])Ac.elements();
        for (int i = 0; i < slices*rows*cols; i++) {
            assertEquals(a_1d[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        FloatMatrix3D Av = A.viewPart(slices / 2  - 1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        float[] av_elems = (float[])Av.copy().elements();
        Ac = Av.getFft2Slices();
        Ac.ifft2Slices(true);
        ac_elems = (float[])Ac.elements();
        for (int i = 0; i < av_elems.length; i++) {
            assertEquals(av_elems[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(slices / 2 -1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        av_elems = (float[])Av.copy().elements();
        Ac = Av.getFft2Slices();
        Ac.ifft2Slices(true);
        ac_elems = (float[])Ac.elements();
        for (int i = 0; i < av_elems.length; i++) {
            assertEquals(av_elems[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
    }
    
    
    @Test
    public void testGetIfft3() {
    	 int slices = 64;
         int rows = 128;
         int cols = 64;
         float[] a_1d = new float[slices * rows * cols];
         for (int i = 0; i < slices * rows * cols; i++) {
             a_1d[i] = (float)Math.random();
         }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        FComplexMatrix3D Ac = A.getIfft3(true);
        Ac.fft3();
        float[] ac_elems = (float[])Ac.elements();
        for (int i = 0; i < slices*rows*cols; i++) {
            assertEquals(a_1d[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        Ac = A.getIfft3(true);
        Ac.fft3();
        ac_elems = (float[])Ac.elements();
        for (int i = 0; i < slices*rows*cols; i++) {
            assertEquals(a_1d[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        FloatMatrix3D Av = A.viewPart(slices / 2  - 1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        float[] av_elems = (float[])Av.copy().elements();
        Ac = Av.getIfft3(true);
        Ac.fft3();
        ac_elems = (float[])Ac.elements();
        for (int i = 0; i < av_elems.length; i++) {
            assertEquals(av_elems[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(slices / 2 -1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        av_elems = (float[])Av.copy().elements();
        Ac = Av.getIfft3(true);
        Ac.fft3();
        ac_elems = (float[])Ac.elements();
        for (int i = 0; i < av_elems.length; i++) {
            assertEquals(av_elems[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
    }
    
    @Test
    public void testGetIfft2Slices() {
    	 int slices = 64;
         int rows = 128;
         int cols = 64;
         float[] a_1d = new float[slices * rows * cols];
         for (int i = 0; i < slices * rows * cols; i++) {
             a_1d[i] = (float)Math.random();
         }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        FComplexMatrix3D Ac = A.getIfft2Slices(true);
        Ac.fft2Slices();
        float[] ac_elems = (float[])Ac.elements();
        for (int i = 0; i < slices*rows*cols; i++) {
            assertEquals(a_1d[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        Ac = A.getIfft2Slices(true);
        Ac.fft2Slices();
        ac_elems = (float[])Ac.elements();
        for (int i = 0; i < slices*rows*cols; i++) {
            assertEquals(a_1d[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        FloatMatrix3D Av = A.viewPart(slices / 2  - 1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        float[] av_elems = (float[])Av.copy().elements();
        Ac = Av.getIfft2Slices(true);
        Ac.fft2Slices();
        ac_elems = (float[])Ac.elements();
        for (int i = 0; i < av_elems.length; i++) {
            assertEquals(av_elems[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(a_1d);
        Av = A.viewPart(slices / 2 -1, rows / 2 - 1, cols / 2 - 1, slices / 2, rows / 2, cols / 2);
        av_elems = (float[])Av.copy().elements();
        Ac = Av.getIfft2Slices(true);
        Ac.fft2Slices();
        ac_elems = (float[])Ac.elements();
        for (int i = 0; i < av_elems.length; i++) {
            assertEquals(av_elems[i], ac_elems[2*i], tol);
            assertEquals(0, ac_elems[2*i+1], tol);
        }
    }
    
    @Test
    public void testEqualsFloat() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        float value = 1;
        A.assign(value);
        boolean eq = A.equals(value);
        assertEquals(true, eq);
        eq = A.equals(2);
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.assign(1);
        eq = A.equals(value);
        assertEquals(true, eq);
        eq = A.equals(2);
        assertEquals(false, eq);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        Av.assign(value);
        eq = Av.equals(value);
        assertEquals(true, eq);
        eq = Av.equals(2);
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        Av = A.viewDice(2, 1, 0);
        Av.assign(value);
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
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D B = new DenseFloatMatrix3D(b_3d);
        boolean eq = A.equals(A);
        assertEquals(true, eq);
        eq = A.equals(B);
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        B = new DenseFloatMatrix3D(b_3d);
        eq = A.equals(A);
        assertEquals(true, eq);
        eq = A.equals(B);
        assertEquals(false, eq);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        B = new DenseFloatMatrix3D(b_3d);
        FloatMatrix3D Bv = B.viewDice(2, 1, 0);
        eq = Av.equals(Av);
        assertEquals(true, eq);
        eq = Av.equals(Bv);
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        Av = A.viewDice(2, 1, 0);
        B = new DenseFloatMatrix3D(b_3d);
        Bv = B.viewDice(2, 1, 0);
        eq = Av.equals(Av);
        assertEquals(true, eq);
        eq = Av.equals(Bv);
        assertEquals(false, eq);

    }

    @Test
    public void testGet() {
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(a_3d[s][r][c], A.get(s, r, c), tol);
                }
            }
        }
    }

    @Test
    public void testGetMaxLocation() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        A.setQuick(slices / 3, rows / 3, cols / 3, 0.7f);
        A.setQuick(slices / 3, rows / 2, cols / 2, 0.7f);
        float[] maxAndLoc = A.getMaxLocation();
        assertEquals(0.7f, maxAndLoc[0], tol);
        assertEquals(slices / 3, (int) maxAndLoc[1]);
        assertEquals(rows / 3, (int) maxAndLoc[2]);
        assertEquals(cols / 3, (int) maxAndLoc[3]);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.setQuick(slices / 3, rows / 3, cols / 3, 0.7f);
        A.setQuick(slices / 3, rows / 2, cols / 2, 0.7f);
        maxAndLoc = A.getMaxLocation();
        assertEquals(0.7f, maxAndLoc[0], tol);
        assertEquals(slices / 3, (int) maxAndLoc[1]);
        assertEquals(rows / 3, (int) maxAndLoc[2]);
        assertEquals(cols / 3, (int) maxAndLoc[3]);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(cols, rows, slices);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        Av.setQuick(slices / 3, rows / 3, cols / 3, 0.7f);
        Av.setQuick(slices / 3, rows / 2, cols / 2, 0.7f);
        maxAndLoc = Av.getMaxLocation();
        assertEquals(0.7f, maxAndLoc[0], tol);
        assertEquals(slices / 3, (int) maxAndLoc[1]);
        assertEquals(rows / 3, (int) maxAndLoc[2]);
        assertEquals(cols / 3, (int) maxAndLoc[3]);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(cols, rows, slices);
        Av = A.viewDice(2, 1, 0);
        Av.setQuick(slices / 3, rows / 3, cols / 3, 0.7f);
        Av.setQuick(slices / 3, rows / 2, cols / 2, 0.7f);
        maxAndLoc = Av.getMaxLocation();
        assertEquals(0.7f, maxAndLoc[0], tol);
        assertEquals(slices / 3, (int) maxAndLoc[1]);
        assertEquals(rows / 3, (int) maxAndLoc[2]);
        assertEquals(cols / 3, (int) maxAndLoc[3]);

    }

    @Test
    public void testGetMinLocation() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        A.setQuick(slices / 3, rows / 3, cols / 3, -0.7f);
        A.setQuick(slices / 3, rows / 2, cols / 2, -0.7f);
        float[] minAndLoc = A.getMinLocation();
        assertEquals(-0.7f, minAndLoc[0], tol);
        assertEquals(slices / 3, (int) minAndLoc[1]);
        assertEquals(rows / 3, (int) minAndLoc[2]);
        assertEquals(cols / 3, (int) minAndLoc[3]);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(slices, rows, cols);
        A.setQuick(slices / 3, rows / 3, cols / 3, -0.7f);
        A.setQuick(slices / 3, rows / 2, cols / 2, -0.7f);
        minAndLoc = A.getMinLocation();
        assertEquals(-0.7f, minAndLoc[0], tol);
        assertEquals(slices / 3, (int) minAndLoc[1]);
        assertEquals(rows / 3, (int) minAndLoc[2]);
        assertEquals(cols / 3, (int) minAndLoc[3]);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(cols, rows, slices);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        Av.setQuick(slices / 3, rows / 3, cols / 3, -0.7f);
        Av.setQuick(slices / 3, rows / 2, cols / 2, -0.7f);
        minAndLoc = Av.getMinLocation();
        assertEquals(-0.7f, minAndLoc[0], tol);
        assertEquals(slices / 3, (int) minAndLoc[1]);
        assertEquals(rows / 3, (int) minAndLoc[2]);
        assertEquals(cols / 3, (int) minAndLoc[3]);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(cols, rows, slices);
        Av = A.viewDice(2, 1, 0);
        Av.setQuick(slices / 3, rows / 3, cols / 3, -0.7f);
        Av.setQuick(slices / 3, rows / 2, cols / 2, -0.7f);
        minAndLoc = Av.getMinLocation();
        assertEquals(-0.7f, minAndLoc[0], tol);
        assertEquals(slices / 3, (int) minAndLoc[1]);
        assertEquals(rows / 3, (int) minAndLoc[2]);
        assertEquals(cols / 3, (int) minAndLoc[3]);

    }

    @Test
    public void testGetNegativeValues() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        A.assign(FloatFunctions.mult(-1));
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        A.getNegativeValues(sliceList, rowList, columnList, valueList);
        assertEquals(slices * rows * cols, sliceList.size());
        assertEquals(slices * rows * cols, rowList.size());
        assertEquals(slices * rows * cols, columnList.size());
        assertEquals(slices * rows * cols, valueList.size());
        int idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(A.getQuick(sliceList.get(idx), rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                    idx++;
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        A.assign(FloatFunctions.mult(-1));
        A.getNegativeValues(sliceList, rowList, columnList, valueList);
        assertEquals(slices * rows * cols, sliceList.size());
        assertEquals(slices * rows * cols, rowList.size());
        assertEquals(slices * rows * cols, columnList.size());
        assertEquals(slices * rows * cols, valueList.size());
        idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(A.getQuick(sliceList.get(idx), rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                    idx++;
                }
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(a_3d);
        A.assign(FloatFunctions.mult(-1));
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        Av.getNegativeValues(sliceList, rowList, columnList, valueList);
        assertEquals(slices * rows * cols, sliceList.size());
        assertEquals(slices * rows * cols, rowList.size());
        assertEquals(slices * rows * cols, columnList.size());
        assertEquals(slices * rows * cols, valueList.size());
        idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(Av.getQuick(sliceList.get(idx), rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                    idx++;
                }
            }
        }

        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        A.assign(FloatFunctions.mult(-1));
        Av = A.viewDice(2, 1, 0);
        Av.getNegativeValues(sliceList, rowList, columnList, valueList);
        assertEquals(slices * rows * cols, sliceList.size());
        assertEquals(slices * rows * cols, rowList.size());
        assertEquals(slices * rows * cols, columnList.size());
        assertEquals(slices * rows * cols, valueList.size());
        idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(Av.getQuick(sliceList.get(idx), rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                    idx++;
                }
            }
        }
    }

    @Test
    public void testGetNonZeros() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        A.getNonZeros(sliceList, rowList, columnList, valueList);
        assertEquals(slices * rows * cols, sliceList.size());
        assertEquals(slices * rows * cols, rowList.size());
        assertEquals(slices * rows * cols, columnList.size());
        assertEquals(slices * rows * cols, valueList.size());
        int idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(A.getQuick(sliceList.get(idx), rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                    idx++;
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        A.getNonZeros(sliceList, rowList, columnList, valueList);
        assertEquals(slices * rows * cols, sliceList.size());
        assertEquals(slices * rows * cols, rowList.size());
        assertEquals(slices * rows * cols, columnList.size());
        assertEquals(slices * rows * cols, valueList.size());
        idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(A.getQuick(sliceList.get(idx), rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                    idx++;
                }
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        Av.getNonZeros(sliceList, rowList, columnList, valueList);
        assertEquals(slices * rows * cols, sliceList.size());
        assertEquals(slices * rows * cols, rowList.size());
        assertEquals(slices * rows * cols, columnList.size());
        assertEquals(slices * rows * cols, valueList.size());
        idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(Av.getQuick(sliceList.get(idx), rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                    idx++;
                }
            }
        }

        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        Av = A.viewDice(2, 1, 0);
        Av.getNonZeros(sliceList, rowList, columnList, valueList);
        assertEquals(slices * rows * cols, sliceList.size());
        assertEquals(slices * rows * cols, rowList.size());
        assertEquals(slices * rows * cols, columnList.size());
        assertEquals(slices * rows * cols, valueList.size());
        idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(Av.getQuick(sliceList.get(idx), rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                    idx++;
                }
            }
        }
    }

    @Test
    public void testGetPositiveValues() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        A.getPositiveValues(sliceList, rowList, columnList, valueList);
        assertEquals(slices * rows * cols, sliceList.size());
        assertEquals(slices * rows * cols, rowList.size());
        assertEquals(slices * rows * cols, columnList.size());
        assertEquals(slices * rows * cols, valueList.size());
        int idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(A.getQuick(sliceList.get(idx), rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                    idx++;
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        A.getPositiveValues(sliceList, rowList, columnList, valueList);
        assertEquals(slices * rows * cols, sliceList.size());
        assertEquals(slices * rows * cols, rowList.size());
        assertEquals(slices * rows * cols, columnList.size());
        assertEquals(slices * rows * cols, valueList.size());
        idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(A.getQuick(sliceList.get(idx), rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                    idx++;
                }
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        Av.getPositiveValues(sliceList, rowList, columnList, valueList);
        assertEquals(slices * rows * cols, sliceList.size());
        assertEquals(slices * rows * cols, rowList.size());
        assertEquals(slices * rows * cols, columnList.size());
        assertEquals(slices * rows * cols, valueList.size());
        idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(Av.getQuick(sliceList.get(idx), rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                    idx++;
                }
            }
        }

        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        Av = A.viewDice(2, 1, 0);
        Av.getPositiveValues(sliceList, rowList, columnList, valueList);
        assertEquals(slices * rows * cols, sliceList.size());
        assertEquals(slices * rows * cols, rowList.size());
        assertEquals(slices * rows * cols, columnList.size());
        assertEquals(slices * rows * cols, valueList.size());
        idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(Av.getQuick(sliceList.get(idx), rowList.get(idx), columnList.get(idx)), valueList.get(idx), tol);
                    idx++;
                }
            }
        }
    }

    @Test
    public void testGetQuick() {
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(a_3d[s][r][c], A.getQuick(s, r, c), tol);
                }
            }
        }
    }

    @Test
    public void testSet() {
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        float elem = (float)Math.random();
        A.set(slices / 2, rows / 2, cols / 2, elem);
        assertEquals(elem, A.getQuick(slices / 2, rows / 2, cols / 2), tol);
    }

    @Test
    public void testSetQuick() {
        FloatMatrix3D A = new DenseFloatMatrix3D(slices, rows, cols);
        float elem = (float)Math.random();
        A.setQuick(slices / 2, rows / 2, cols / 2, elem);
        assertEquals(elem, A.getQuick(slices / 2, rows / 2, cols / 2), tol);
    }

    @Test
    public void testToArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        float[][][] array = A.toArray();
        AssertUtils.assertArrayEquals(a_3d, array, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        array = A.toArray();
        AssertUtils.assertArrayEquals(a_3d, array, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        array = Av.toArray();
        for (int s = 0; s < cols; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < slices; c++) {
                    assertEquals(a_3d[c][r][s], Av.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        Av = A.viewDice(2, 1, 0);
        array = Av.toArray();
        for (int s = 0; s < cols; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < slices; c++) {
                    assertEquals(a_3d[c][r][s], Av.getQuick(s, r, c), tol);
                }
            }
        }
    }

    @Test
    public void testVectorize() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix1D B = A.vectorize();
        int idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int c = 0; c < cols; c++) {
                for (int r = 0; r < rows; r++) {
                    assertEquals(A.getQuick(s, r, c), B.getQuick(idx++), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        B = A.vectorize();
        idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int c = 0; c < cols; c++) {
                for (int r = 0; r < rows; r++) {
                    assertEquals(A.getQuick(s, r, c), B.getQuick(idx++), tol);
                }
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        B = Av.vectorize();
        idx = 0;
        for (int s = 0; s < cols; s++) {
            for (int c = 0; c < slices; c++) {
                for (int r = 0; r < rows; r++) {
                    assertEquals(Av.getQuick(s, r, c), B.getQuick(idx++), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        Av = A.viewDice(2, 1, 0);
        B = Av.vectorize();
        idx = 0;
        for (int s = 0; s < cols; s++) {
            for (int c = 0; c < slices; c++) {
                for (int r = 0; r < rows; r++) {
                    assertEquals(Av.getQuick(s, r, c), B.getQuick(idx++), tol);
                }
            }
        }
    }

    @Test
    public void testViewColumn() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix2D B = A.viewColumn(cols / 2);
        assertEquals(slices, B.rows());
        assertEquals(rows, B.columns());
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                assertEquals(a_3d[s][r][cols / 2], B.getQuick(s, r), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        B = A.viewColumn(cols / 2);
        assertEquals(slices, B.rows());
        assertEquals(rows, B.columns());
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                assertEquals(a_3d[s][r][cols / 2], B.getQuick(s, r), tol);
            }
        }

    }

    @Test
    public void testViewColumnFlip() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D B = A.viewColumnFlip();
        assertEquals(A.size(), B.size());
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(a_3d[s][r][cols - 1 - c], B.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        B = A.viewColumnFlip();
        assertEquals(A.size(), B.size());
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(a_3d[s][r][cols - 1 - c], B.getQuick(s, r, c), tol);
                }
            }
        }
    }

    @Test
    public void testViewDice() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D B = A.viewDice(2, 1, 0);
        assertEquals(A.slices(), B.columns());
        assertEquals(A.rows(), B.rows());
        assertEquals(A.columns(), B.slices());
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(A.getQuick(s, r, c), B.getQuick(c, r, s), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        B = A.viewDice(2, 1, 0);
        assertEquals(A.slices(), B.columns());
        assertEquals(A.rows(), B.rows());
        assertEquals(A.columns(), B.slices());
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(A.getQuick(s, r, c), B.getQuick(c, r, s), tol);
                }
            }
        }
    }

    @Test
    public void testViewPart() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D B = A.viewPart(2, 15, 11, 2, 21, 27);
        for (int s = 0; s < 2; s++) {
            for (int r = 0; r < 21; r++) {
                for (int c = 0; c < 27; c++) {
                    assertEquals(A.getQuick(2 + s, 15 + r, 11 + c), B.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        B = A.viewPart(2, 15, 11, 2, 21, 27);
        for (int s = 0; s < 2; s++) {
            for (int r = 0; r < 21; r++) {
                for (int c = 0; c < 27; c++) {
                    assertEquals(A.getQuick(2 + s, 15 + r, 11 + c), B.getQuick(s, r, c), tol);
                }
            }
        }
    }

    @Test
    public void testViewRow() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix2D B = A.viewRow(rows / 2);
        assertEquals(slices, B.rows());
        assertEquals(cols, B.columns());
        for (int s = 0; s < slices; s++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_3d[s][rows / 2][c], B.getQuick(s, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        B = A.viewRow(rows / 2);
        assertEquals(slices, B.rows());
        assertEquals(cols, B.columns());
        for (int s = 0; s < slices; s++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_3d[s][rows / 2][c], B.getQuick(s, c), tol);
            }
        }
    }

    @Test
    public void testViewRowFlip() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D B = A.viewRowFlip();
        assertEquals(A.size(), B.size());
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(a_3d[s][rows - 1 - r][c], B.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        B = A.viewRowFlip();
        assertEquals(A.size(), B.size());
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(a_3d[s][rows - 1 - r][c], B.getQuick(s, r, c), tol);
                }
            }
        }
    }

    @Test
    public void testViewSelectionFloatMatrix2DProcedure() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        final float value = 2;
        A.setQuick(3, rows / 4, 0, value);
        FloatMatrix3D B = A.viewSelection(new FloatMatrix2DProcedure() {
            public boolean apply(FloatMatrix2D element) {
                if (Math.abs(element.getQuick(rows / 4, 0) - value) <= tol) {
                    return true;
                } else {
                    return false;
                }

            }
        });
        assertEquals(1, B.slices());
        assertEquals(A.rows(), B.rows());
        assertEquals(A.columns(), B.columns());
        assertEquals(A.getQuick(3, rows / 4, 0), B.getQuick(0, rows / 4, 0), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        A.setQuick(3, rows / 4, 0, value);
        B = A.viewSelection(new FloatMatrix2DProcedure() {
            public boolean apply(FloatMatrix2D element) {
                if (Math.abs(element.getQuick(rows / 4, 0) - value) <= tol) {
                    return true;
                } else {
                    return false;
                }

            }
        });
        assertEquals(1, B.slices());
        assertEquals(A.rows(), B.rows());
        assertEquals(A.columns(), B.columns());
        assertEquals(A.getQuick(3, rows / 4, 0), B.getQuick(0, rows / 4, 0), tol);
    }

    @Test
    public void testViewSelectionIntArrayIntArrayIntArray() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        int[] sliceIndexes = new int[] { 2, 3 };
        int[] rowIndexes = new int[] { 5, 11, 22, 37 };
        int[] colIndexes = new int[] { 2, 17, 32, 47, 51 };
        FloatMatrix3D B = A.viewSelection(sliceIndexes, rowIndexes, colIndexes);
        assertEquals(sliceIndexes.length, B.slices());
        assertEquals(rowIndexes.length, B.rows());
        assertEquals(colIndexes.length, B.columns());
        for (int s = 0; s < sliceIndexes.length; s++) {
            for (int r = 0; r < rowIndexes.length; r++) {
                for (int c = 0; c < colIndexes.length; c++) {
                    assertEquals(A.getQuick(sliceIndexes[s], rowIndexes[r], colIndexes[c]), B.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        sliceIndexes = new int[] { 2, 3 };
        rowIndexes = new int[] { 5, 11, 22, 37 };
        colIndexes = new int[] { 2, 17, 32, 47, 51 };
        B = A.viewSelection(sliceIndexes, rowIndexes, colIndexes);
        assertEquals(sliceIndexes.length, B.slices());
        assertEquals(rowIndexes.length, B.rows());
        assertEquals(colIndexes.length, B.columns());
        for (int s = 0; s < sliceIndexes.length; s++) {
            for (int r = 0; r < rowIndexes.length; r++) {
                for (int c = 0; c < colIndexes.length; c++) {
                    assertEquals(A.getQuick(sliceIndexes[s], rowIndexes[r], colIndexes[c]), B.getQuick(s, r, c), tol);
                }
            }
        }
    }

    @Test
    public void testViewSlice() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix2D B = A.viewSlice(slices / 2);
        assertEquals(rows, B.rows());
        assertEquals(cols, B.columns());
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_3d[slices / 2][r][c], B.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        B = A.viewSlice(slices / 2);
        assertEquals(rows, B.rows());
        assertEquals(cols, B.columns());
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_3d[slices / 2][r][c], B.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testViewSliceFlip() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D B = A.viewSliceFlip();
        assertEquals(A.size(), B.size());
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(a_3d[slices - 1 - s][r][c], B.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        B = A.viewSliceFlip();
        assertEquals(A.size(), B.size());
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    assertEquals(a_3d[slices - 1 - s][r][c], B.getQuick(s, r, c), tol);
                }
            }
        }
    }

    @Test
    public void testViewSorted() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D B = A.viewSorted(1, 1);
        for (int s = 0; s < slices - 1; s++) {
            if (B.getQuick(s, 1, 1) > B.getQuick(s + 1, 1, 1)) {
                fail();
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        B = A.viewSorted(1, 1);
        for (int s = 0; s < slices - 1; s++) {
            if (B.getQuick(s, 1, 1) > B.getQuick(s + 1, 1, 1)) {
                fail();
            }
        }
    }

    @Test
    public void testViewStrides() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        int sliceStride = 2;
        int rowStride = 3;
        int colStride = 5;
        FloatMatrix3D B = A.viewStrides(sliceStride, rowStride, colStride);
        for (int s = 0; s < B.slices(); s++) {
            for (int r = 0; r < B.rows(); r++) {
                for (int c = 0; c < B.columns(); c++) {
                    assertEquals(A.getQuick(s * sliceStride, r * rowStride, c * colStride), B.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        B = A.viewStrides(sliceStride, rowStride, colStride);
        for (int s = 0; s < B.slices(); s++) {
            for (int r = 0; r < B.rows(); r++) {
                for (int c = 0; c < B.columns(); c++) {
                    assertEquals(A.getQuick(s * sliceStride, r * rowStride, c * colStride), B.getQuick(s, r, c), tol);
                }
            }
        }
    }

    @Test
    public void testZSum() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix3D A = new DenseFloatMatrix3D(a_3d);
        float aSum = A.zSum();
        float tmpSum = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    tmpSum += a_3d[s][r][c];
                }
            }
        }
        assertEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        aSum = A.zSum();
        tmpSum = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    tmpSum += a_3d[s][r][c];
                }
            }
        }
        assertEquals(tmpSum, aSum, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix3D(a_3d);
        FloatMatrix3D Av = A.viewDice(2, 1, 0);
        aSum = Av.zSum();
        tmpSum = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    tmpSum += a_3d[s][r][c];
                }
            }
        }
        assertEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_3D(nThreadsBegin);
        A = new DenseFloatMatrix3D(a_3d);
        Av = A.viewDice(2, 1, 0);
        aSum = Av.zSum();
        tmpSum = 0;
        for (int s = 0; s < slices; s++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    tmpSum += a_3d[s][r][c];
                }
            }
        }
        assertEquals(tmpSum, aSum, tol);
    }

}
