package cern.colt.matrix.impl;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Random;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import cern.colt.function.FComplexProcedure;
import cern.colt.function.IntIntFComplexFunction;
import cern.colt.list.IntArrayList;
import cern.colt.matrix.FComplexMatrix1D;
import cern.colt.matrix.FComplexMatrix1DProcedure;
import cern.colt.matrix.FComplexMatrix2D;
import cern.colt.matrix.FloatFactory2D;
import cern.colt.matrix.FloatMatrix2D;
import cern.jet.math.FComplex;
import cern.jet.math.FComplexFunctions;
import edu.emory.mathcs.utils.AssertUtils;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public class TestDenseFComplexMatrix2D {
    private static final int rows = 113;

    private static final int cols = 117;

    private static final float tol = 1e-1f;

    private static final int nThreads = 3;

    private static final int nThreadsBegin = 1;

    private static final FloatFactory2D factory = FloatFactory2D.dense;

    private float[][] a_2d, b_2d, a_2dt, b_2dt;

    private float[] a_1d, b_1d, a_1dt;

    private Random rand;

    @Before
    public void setUpBeforeClass() throws Exception {
        rand = new Random(0);

        a_1d = new float[rows * 2 * cols];
        a_2d = new float[rows][2 * cols];
        int idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * cols; c++) {
                a_2d[r][c] = rand.nextFloat();
                a_1d[idx++] = a_2d[r][c];
            }
        }

        b_1d = new float[rows * 2 * cols];
        b_2d = new float[rows][2 * cols];
        idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < 2 * cols; c++) {
                b_2d[r][c] = rand.nextFloat();
                b_1d[idx++] = b_2d[r][c];
            }
        }
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        a_2dt = A.viewDice().toArray();
        a_1dt = (float[]) A.viewDice().copy().elements();
        FComplexMatrix2D B = new DenseFComplexMatrix2D(b_2d);
        b_2dt = B.viewDice().toArray();
    }

    @After
    public void tearDownAfterClass() throws Exception {
        a_1d = null;
        a_1dt = null;
        a_2d = null;
        a_2dt = null;
        b_1d = null;
        b_2d = null;
        b_2dt = null;
        System.gc();
    }

    @Test
    public void testAggregateComplexComplexComplexFunctionComplexComplexFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        float[] aSum = A.aggregate(FComplexFunctions.plus, FComplexFunctions.square);
        float[] tmpSum = new float[2];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmpSum = FComplex.plus(tmpSum, FComplex.square(new float[] { a_2d[r][2 * c], a_2d[r][2 * c + 1] }));
            }
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        aSum = A.aggregate(FComplexFunctions.plus, FComplexFunctions.square);
        tmpSum = new float[2];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmpSum = FComplex.plus(tmpSum, FComplex.square(new float[] { a_2d[r][2 * c], a_2d[r][2 * c + 1] }));
            }
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        aSum = A.aggregate(FComplexFunctions.plus, FComplexFunctions.square);
        tmpSum = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmpSum = FComplex.plus(tmpSum, FComplex.square(new float[] { a_2dt[r][2 * c], a_2dt[r][2 * c + 1] }));
            }
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Av = A.viewDice();
        aSum = A.aggregate(FComplexFunctions.plus, FComplexFunctions.square);
        tmpSum = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmpSum = FComplex.plus(tmpSum, FComplex.square(new float[] { a_2dt[r][2 * c], a_2dt[r][2 * c + 1] }));
            }
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
    }

    @Test
    public void testAggregateComplexMatrix2FComplexComplexComplexFunctionComplexComplexComplexFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D B = new DenseFComplexMatrix2D(b_2d);
        float[] sumMult = A.aggregate(B, FComplexFunctions.plus, FComplexFunctions.mult);
        float[] tmpSumMult = new float[2];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a_2d[r][2 * c], a_2d[r][2 * c + 1] }, new float[] { b_2d[r][2 * c], b_2d[r][2 * c + 1] }));
            }
        }
        AssertUtils.assertArrayEquals(tmpSumMult, sumMult, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        B = new DenseFComplexMatrix2D(b_2d);
        sumMult = A.aggregate(B, FComplexFunctions.plus, FComplexFunctions.mult);
        tmpSumMult = new float[2];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a_2d[r][2 * c], a_2d[r][2 * c + 1] }, new float[] { b_2d[r][2 * c], b_2d[r][2 * c + 1] }));
            }
        }
        AssertUtils.assertArrayEquals(tmpSumMult, sumMult, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        B = new DenseFComplexMatrix2D(b_2d);
        FComplexMatrix2D Bv = B.viewDice();
        sumMult = Av.aggregate(Bv, FComplexFunctions.plus, FComplexFunctions.mult);
        tmpSumMult = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a_2dt[r][2 * c], a_2dt[r][2 * c + 1] }, new float[] { b_2dt[r][2 * c], b_2dt[r][2 * c + 1] }));
            }
        }
        AssertUtils.assertArrayEquals(tmpSumMult, sumMult, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Av = A.viewDice();
        B = new DenseFComplexMatrix2D(b_2d);
        Bv = B.viewDice();
        sumMult = Av.aggregate(Bv, FComplexFunctions.plus, FComplexFunctions.mult);
        tmpSumMult = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a_2dt[r][2 * c], a_2dt[r][2 * c + 1] }, new float[] { b_2dt[r][2 * c], b_2dt[r][2 * c + 1] }));
            }
        }
        AssertUtils.assertArrayEquals(tmpSumMult, sumMult, tol);
    }

    @Test
    public void testAssignComplexComplexFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        A.assign(FComplexFunctions.acos);
        float[] tmp = new float[2];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmp[0] = a_2d[r][2 * c];
                tmp[1] = a_2d[r][2 * c + 1];
                tmp = FComplex.acos(tmp);
                AssertUtils.assertArrayEquals(tmp, A.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        A.assign(FComplexFunctions.acos);
        tmp = new float[2];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmp[0] = a_2d[r][2 * c];
                tmp[1] = a_2d[r][2 * c + 1];
                tmp = FComplex.acos(tmp);
                AssertUtils.assertArrayEquals(tmp, A.getQuick(r, c), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        Av.assign(FComplexFunctions.acos);
        tmp = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp[0] = a_2dt[r][2 * c];
                tmp[1] = a_2dt[r][2 * c + 1];
                tmp = FComplex.acos(tmp);
                AssertUtils.assertArrayEquals(tmp, Av.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Av = A.viewDice();
        Av.assign(FComplexFunctions.acos);
        tmp = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp[0] = a_2dt[r][2 * c];
                tmp[1] = a_2dt[r][2 * c + 1];
                tmp = FComplex.acos(tmp);
                AssertUtils.assertArrayEquals(tmp, Av.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testAssignComplexMatrix2D() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(rows, cols);
        FComplexMatrix2D B = new DenseFComplexMatrix2D(a_2d);
        A.assign(B);
        float[] aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a_1d, aElts, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(rows, cols);
        B = new DenseFComplexMatrix2D(a_2d);
        A.assign(B);
        aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a_1d, aElts, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(rows, cols);
        FComplexMatrix2D Av = A.viewDice();
        B = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Bv = B.viewDice();
        Av.assign(Bv);
        float[] tmp = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp[0] = a_2dt[r][2 * c];
                tmp[1] = a_2dt[r][2 * c + 1];
                AssertUtils.assertArrayEquals(tmp, Av.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(rows, cols);
        Av = A.viewDice();
        B = new DenseFComplexMatrix2D(a_2d);
        Bv = B.viewDice();
        Av.assign(Bv);
        tmp = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp[0] = a_2dt[r][2 * c];
                tmp[1] = a_2dt[r][2 * c + 1];
                AssertUtils.assertArrayEquals(tmp, Av.getQuick(r, c), tol);
            }
        }

    }

    @Test
    public void testAssignComplexMatrix2FComplexComplexComplexFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D B = new DenseFComplexMatrix2D(b_2d);
        A.assign(B, FComplexFunctions.div);
        float[] tmp1 = new float[2];
        float[] tmp2 = new float[2];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmp1[0] = a_2d[r][2 * c];
                tmp1[1] = a_2d[r][2 * c + 1];
                tmp2[0] = b_2d[r][2 * c];
                tmp2[1] = b_2d[r][2 * c + 1];
                tmp1 = FComplex.div(tmp1, tmp2);
                AssertUtils.assertArrayEquals(tmp1, A.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        B = new DenseFComplexMatrix2D(b_2d);
        A.assign(B, FComplexFunctions.div);
        tmp1 = new float[2];
        tmp2 = new float[2];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmp1[0] = a_2d[r][2 * c];
                tmp1[1] = a_2d[r][2 * c + 1];
                tmp2[0] = b_2d[r][2 * c];
                tmp2[1] = b_2d[r][2 * c + 1];
                tmp1 = FComplex.div(tmp1, tmp2);
                AssertUtils.assertArrayEquals(tmp1, A.getQuick(r, c), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        B = new DenseFComplexMatrix2D(b_2d);
        FComplexMatrix2D Bv = B.viewDice();
        Av.assign(Bv, FComplexFunctions.div);
        tmp1 = new float[2];
        tmp2 = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp1[0] = a_2dt[r][2 * c];
                tmp1[1] = a_2dt[r][2 * c + 1];
                tmp2[0] = b_2dt[r][2 * c];
                tmp2[1] = b_2dt[r][2 * c + 1];
                tmp1 = FComplex.div(tmp1, tmp2);
                AssertUtils.assertArrayEquals(tmp1, Av.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Av = A.viewDice();
        B = new DenseFComplexMatrix2D(b_2d);
        Bv = B.viewDice();
        Av.assign(Bv, FComplexFunctions.div);
        tmp1 = new float[2];
        tmp2 = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp1[0] = a_2dt[r][2 * c];
                tmp1[1] = a_2dt[r][2 * c + 1];
                tmp2[0] = b_2dt[r][2 * c];
                tmp2[1] = b_2dt[r][2 * c + 1];
                tmp1 = FComplex.div(tmp1, tmp2);
                AssertUtils.assertArrayEquals(tmp1, Av.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testAssignComplexProcedureComplexComplexFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D B = A.copy();
        A.assign(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, FComplexFunctions.tan);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (FComplex.abs(B.getQuick(r, c)) > 3) {
                    B.setQuick(r, c, FComplex.tan(B.getQuick(r, c)));
                }
            }
        }
        AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        B = A.copy();
        A.assign(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, FComplexFunctions.tan);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (FComplex.abs(B.getQuick(r, c)) > 3) {
                    B.setQuick(r, c, FComplex.tan(B.getQuick(r, c)));
                }
            }
        }
        AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        B = A.copy();
        FComplexMatrix2D Bv = B.viewDice();
        Av.assign(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, FComplexFunctions.tan);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                if (FComplex.abs(Bv.getQuick(r, c)) > 3) {
                    Bv.setQuick(r, c, FComplex.tan(Bv.getQuick(r, c)));
                }
            }
        }
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                AssertUtils.assertArrayEquals(Bv.getQuick(r, c), Av.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Av = A.viewDice();
        B = A.copy();
        Bv = B.viewDice();
        Av.assign(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, FComplexFunctions.tan);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                if (FComplex.abs(Bv.getQuick(r, c)) > 3) {
                    Bv.setQuick(r, c, FComplex.tan(Bv.getQuick(r, c)));
                }
            }
        }
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                AssertUtils.assertArrayEquals(Bv.getQuick(r, c), Av.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testAssignComplexProcedureFloatArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D B = A.copy();
        A.assign(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, new float[] { -1, -1 });
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (FComplex.abs(B.getQuick(r, c)) > 3) {
                    B.setQuick(r, c, new float[] { -1, -1 });
                }
            }
        }
        AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        B = A.copy();
        A.assign(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, new float[] { -1, -1 });
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (FComplex.abs(B.getQuick(r, c)) > 3) {
                    B.setQuick(r, c, new float[] { -1, -1 });
                }
            }
        }
        AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        B = A.copy();
        FComplexMatrix2D Bv = B.viewDice();
        Av.assign(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, new float[] { -1, -1 });
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                if (FComplex.abs(Bv.getQuick(r, c)) > 3) {
                    Bv.setQuick(r, c, new float[] { -1, -1 });
                }
            }
        }
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                AssertUtils.assertArrayEquals(Bv.getQuick(r, c), Av.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Av = A.viewDice();
        B = A.copy();
        Bv = B.viewDice();
        Av.assign(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, new float[] { -1, -1 });
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                if (FComplex.abs(Bv.getQuick(r, c)) > 3) {
                    Bv.setQuick(r, c, new float[] { -1, -1 });
                }
            }
        }
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                AssertUtils.assertArrayEquals(Bv.getQuick(r, c), Av.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testAssignComplexRealFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        A.assign(FComplexFunctions.abs);
        float[] tmp = new float[2];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmp[0] = a_2d[r][2 * c];
                tmp[1] = a_2d[r][2 * c + 1];
                tmp[0] = FComplex.abs(tmp);
                tmp[1] = 0;
                AssertUtils.assertArrayEquals(tmp, A.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        A.assign(FComplexFunctions.abs);
        tmp = new float[2];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmp[0] = a_2d[r][2 * c];
                tmp[1] = a_2d[r][2 * c + 1];
                tmp[0] = FComplex.abs(tmp);
                tmp[1] = 0;
                AssertUtils.assertArrayEquals(tmp, A.getQuick(r, c), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        Av.assign(FComplexFunctions.abs);
        tmp = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp[0] = a_2dt[r][2 * c];
                tmp[1] = a_2dt[r][2 * c + 1];
                tmp[0] = FComplex.abs(tmp);
                tmp[1] = 0;
                AssertUtils.assertArrayEquals(tmp, Av.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Av = A.viewDice();
        Av.assign(FComplexFunctions.abs);
        tmp = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp[0] = a_2dt[r][2 * c];
                tmp[1] = a_2dt[r][2 * c + 1];
                tmp[0] = FComplex.abs(tmp);
                tmp[1] = 0;
                AssertUtils.assertArrayEquals(tmp, Av.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testAssignFloatArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(rows, cols);
        A.assign(a_1d);
        float[] aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a_1d, aElts, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(rows, cols);
        A.assign(a_1d);
        aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a_1d, aElts, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(rows, cols);
        FComplexMatrix2D Av = A.viewDice();
        Av.assign(a_1dt);
        float[] tmp = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp = Av.getQuick(r, c);
                assertEquals(a_1dt[r * 2 * rows + 2 * c], tmp[0], tol);
                assertEquals(a_1dt[r * 2 * rows + 2 * c + 1], tmp[1], tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(rows, cols);
        Av = A.viewDice();
        Av.assign(a_1dt);
        tmp = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp = Av.getQuick(r, c);
                assertEquals(a_1dt[r * 2 * rows + 2 * c], tmp[0], tol);
                assertEquals(a_1dt[r * 2 * rows + 2 * c + 1], tmp[1], tol);
            }
        }
    }

    @Test
    public void testAssignFloatArrayArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(rows, cols);
        A.assign(a_2d);
        float[] aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a_1d, aElts, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(rows, cols);
        A.assign(a_2d);
        aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a_1d, aElts, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(rows, cols);
        FComplexMatrix2D Av = A.viewDice();
        Av.assign(a_2dt);
        float[] tmp = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp = Av.getQuick(r, c);
                assertEquals(a_2dt[r][2 * c], tmp[0], tol);
                assertEquals(a_2dt[r][2 * c + 1], tmp[1], tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(rows, cols);
        Av = A.viewDice();
        Av.assign(a_2dt);
        tmp = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp = Av.getQuick(r, c);
                assertEquals(a_2dt[r][2 * c], tmp[0], tol);
                assertEquals(a_2dt[r][2 * c + 1], tmp[1], tol);
            }
        }
    }

    @Test
    public void testAssignFloatFloat() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(rows, cols);
        float[] value = new float[] { (float)Math.random(), (float)Math.random() };
        A.assign(value[0], value[1]);
        float[] aElt = null;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                aElt = A.getQuick(r, c);
                AssertUtils.assertArrayEquals(value, aElt, tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(rows, cols);
        value = new float[] { (float)Math.random(), (float)Math.random() };
        A.assign(value[0], value[1]);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                aElt = A.getQuick(r, c);
                AssertUtils.assertArrayEquals(value, aElt, tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(rows, cols);
        FComplexMatrix2D Av = A.viewDice();
        Av.assign(value[0], value[1]);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                aElt = Av.getQuick(r, c);
                AssertUtils.assertArrayEquals(value, aElt, tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(rows, cols);
        Av = A.viewDice();
        Av.assign(value[0], value[1]);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                aElt = Av.getQuick(r, c);
                AssertUtils.assertArrayEquals(value, aElt, tol);
            }
        }
    }

    @Test
    public void testAssignImaginary() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(rows, cols);
        FloatMatrix2D Im = factory.random(rows, cols);
        A.assignImaginary(Im);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                AssertUtils.assertArrayEquals(new float[] { 0, Im.getQuick(r, c) }, A.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(rows, cols);
        Im = factory.random(rows, cols);
        A.assignImaginary(Im);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                AssertUtils.assertArrayEquals(new float[] { 0, Im.getQuick(r, c) }, A.getQuick(r, c), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(rows, cols);
        FComplexMatrix2D Av = A.viewDice();
        Im = factory.random(rows, cols);
        FloatMatrix2D Imv = Im.viewDice();
        Av.assignImaginary(Imv);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                AssertUtils.assertArrayEquals(new float[] { 0, Imv.getQuick(r, c) }, Av.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(rows, cols);
        Av = A.viewDice();
        Im = factory.random(rows, cols);
        Imv = Im.viewDice();
        Av.assignImaginary(Imv);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                AssertUtils.assertArrayEquals(new float[] { 0, Imv.getQuick(r, c) }, Av.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testAssignReal() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(rows, cols);
        FloatMatrix2D Re = factory.random(rows, cols);
        A.assignReal(Re);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                AssertUtils.assertArrayEquals(new float[] { Re.getQuick(r, c), 0 }, A.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(rows, cols);
        Re = factory.random(rows, cols);
        A.assignReal(Re);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                AssertUtils.assertArrayEquals(new float[] { Re.getQuick(r, c), 0 }, A.getQuick(r, c), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(rows, cols);
        FComplexMatrix2D Av = A.viewDice();
        Re = factory.random(rows, cols);
        FloatMatrix2D Rev = Re.viewDice();
        Av.assignReal(Rev);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                AssertUtils.assertArrayEquals(new float[] { Rev.getQuick(r, c), 0 }, Av.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(rows, cols);
        Av = A.viewDice();
        Re = factory.random(rows, cols);
        Rev = Re.viewDice();
        Av.assignReal(Rev);
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                AssertUtils.assertArrayEquals(new float[] { Rev.getQuick(r, c), 0 }, Av.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testCardinality() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        int card = A.cardinality();
        assertEquals(rows * cols, card);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        card = A.cardinality();
        assertEquals(rows * cols, card);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        card = Av.cardinality();
        assertEquals(rows * cols, card);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Av = A.viewDice();
        card = Av.cardinality();
        assertEquals(rows * cols, card);
    }

    @Test
    public void testCopy() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D B = A.copy();
        float[] bElts = (float[]) B.elements();
        AssertUtils.assertArrayEquals(a_1d, bElts, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        B = A.copy();
        bElts = (float[]) B.elements();
        AssertUtils.assertArrayEquals(a_1d, bElts, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        B = Av.copy();
        float[] tmp;
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp = B.getQuick(r, c);
                assertEquals(a_2dt[r][2 * c], tmp[0], tol);
                assertEquals(a_2dt[r][2 * c + 1], tmp[1], tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Av = A.viewDice();
        B = Av.copy();
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp = B.getQuick(r, c);
                assertEquals(a_2dt[r][2 * c], tmp[0], tol);
                assertEquals(a_2dt[r][2 * c + 1], tmp[1], tol);
            }
        }

    }

    @Test
    public void testEqualsFloatArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(rows, cols);
        float[] value = new float[] { 1, 1 };
        A.assign(1, 1);
        boolean eq = A.equals(value);
        assertEquals(true, eq);
        eq = A.equals(new float[] { 2, 1 });
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(rows, cols);
        A.assign(1, 1);
        eq = A.equals(value);
        assertEquals(true, eq);
        eq = A.equals(new float[] { 2, 1 });
        assertEquals(false, eq);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(rows, cols);
        FComplexMatrix2D Av = A.viewDice();
        Av.assign(1, 1);
        eq = Av.equals(value);
        assertEquals(true, eq);
        eq = Av.equals(new float[] { 2, 1 });
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(rows, cols);
        Av = A.viewDice();
        Av.assign(1, 1);
        eq = Av.equals(value);
        assertEquals(true, eq);
        eq = Av.equals(new float[] { 2, 1 });
        assertEquals(false, eq);
    }

    @Test
    public void testEqualsObject() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D B = new DenseFComplexMatrix2D(b_2d);
        boolean eq = A.equals(A);
        assertEquals(true, eq);
        eq = A.equals(B);
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        B = new DenseFComplexMatrix2D(b_2d);
        eq = A.equals(A);
        assertEquals(true, eq);
        eq = A.equals(B);
        assertEquals(false, eq);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        B = new DenseFComplexMatrix2D(b_2d);
        FComplexMatrix2D Bv = B.viewDice();
        eq = Av.equals(Av);
        assertEquals(true, eq);
        eq = Av.equals(Bv);
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Av = A.viewDice();
        B = new DenseFComplexMatrix2D(b_2d);
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
        FComplexMatrix2D A = new DenseFComplexMatrix2D(rows, cols / 2);
        A.assign(a_1d);
        A.fft2();
        A.ifft2(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(rows, cols / 2);
        A.assign(a_1d);
        A.fft2();
        A.ifft2(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(rows, cols / 2);
        A.assign(a_1d);
        FComplexMatrix2D Av = A.viewPart(rows / 2 - 1, cols / 4 - 1, rows / 2, cols / 4);
        float[] av_elems = (float[])Av.copy().elements();
        Av.fft2();
        Av.ifft2(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(rows, cols / 2);
        A.assign(a_1d);
        Av = A.viewPart(rows / 2 - 1, cols / 4 - 1, rows / 2, cols / 4);
        av_elems = (float[])Av.copy().elements();
        Av.fft2();
        Av.ifft2(true);
        AssertUtils.assertArrayEquals(av_elems,  (float[]) Av.copy().elements(), tol);
    }
    
    @Test
	public void testFftColumns() {
		int rows = 512;
		int cols = 256;
		float[] a_1d = new float[rows * cols];
		for (int i = 0; i < rows * cols; i++) {
			a_1d[i] = (float)Math.random();
		}
		/* No view */
		// single thread
		ConcurrencyUtils.setNumberOfProcessors(1);
		FComplexMatrix2D A = new DenseFComplexMatrix2D(rows, cols / 2);
		A.assign(a_1d);
		A.fftColumns();
		A.ifftColumns(true);
		AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
		// multiple threads
		ConcurrencyUtils.setNumberOfProcessors(nThreads);
		ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
		A = new DenseFComplexMatrix2D(rows, cols / 2);
		A.assign(a_1d);
		A.fftColumns();
		A.ifftColumns(true);
		AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
		/* View */
		// single thread
		ConcurrencyUtils.setNumberOfProcessors(1);
		A = new DenseFComplexMatrix2D(rows, cols / 2);
		A.assign(a_1d);
		FComplexMatrix2D Av = A.viewPart(rows / 2 - 1, cols / 4 - 1, rows / 2, cols / 4);
		float[] av_elems = (float[]) Av.copy().elements();
		Av.fftColumns();
		Av.ifftColumns(true);
		AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
		// multiple threads
		ConcurrencyUtils.setNumberOfProcessors(nThreads);
		ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
		A = new DenseFComplexMatrix2D(rows, cols / 2);
		A.assign(a_1d);
		Av = A.viewPart(rows / 2 - 1, cols / 4 - 1, rows / 2, cols / 4);
		av_elems = (float[]) Av.copy().elements();
		Av.fftColumns();
		Av.ifftColumns(true);
		AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
	}

    @Test
	public void testFftRows() {
		int rows = 512;
		int cols = 256;
		float[] a_1d = new float[rows * cols];
		for (int i = 0; i < rows * cols; i++) {
			a_1d[i] = (float)Math.random();
		}
		/* No view */
		// single thread
		ConcurrencyUtils.setNumberOfProcessors(1);
		FComplexMatrix2D A = new DenseFComplexMatrix2D(rows, cols / 2);
		A.assign(a_1d);
		A.fftRows();
		A.ifftRows(true);
		AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
		// multiple threads
		ConcurrencyUtils.setNumberOfProcessors(nThreads);
		ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
		A = new DenseFComplexMatrix2D(rows, cols / 2);
		A.assign(a_1d);
		A.fftRows();
		A.ifftRows(true);
		AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
		/* View */
		// single thread
		ConcurrencyUtils.setNumberOfProcessors(1);
		A = new DenseFComplexMatrix2D(rows, cols / 2);
		A.assign(a_1d);
		FComplexMatrix2D Av = A.viewPart(rows / 2 - 1, cols / 4 - 1, rows / 2, cols / 4);
		float[] av_elems = (float[]) Av.copy().elements();
		Av.fftRows();
		Av.ifftRows(true);
		AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
		// multiple threads
		ConcurrencyUtils.setNumberOfProcessors(nThreads);
		ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
		A = new DenseFComplexMatrix2D(rows, cols / 2);
		A.assign(a_1d);
		Av = A.viewPart(rows / 2 - 1, cols / 4 - 1, rows / 2, cols / 4);
		av_elems = (float[]) Av.copy().elements();
		Av.fftRows();
		Av.ifftRows(true);
		AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
	}
    
    @Test
    public void testForEachNonZero() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(rows, cols);
        float[] value = new float[] { 1.5f, 2.7f };
        A.setQuick(0, 0, value);
        value[0] = -3.3f;
        value[1] = 0;
        A.setQuick(3, 5, value);
        value[0] = 222.3f;
        value[1] = -123.9f;
        A.setQuick(11, 22, value);
        value[0] = 0.1123f;
        value[1] = 156.9f;
        A.setQuick(89, 67, value);
        float[] aElts = new float[rows * 2 * cols];
        System.arraycopy((float[]) A.elements(), 0, aElts, 0, rows * 2 * cols);
        A.forEachNonZero(new IntIntFComplexFunction() {
            public float[] apply(int first, int second, float[] third) {
                return FComplex.sqrt(third);
            }
        });
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                AssertUtils.assertArrayEquals(FComplex.sqrt(new float[] { aElts[r * 2 * cols + 2 * c], aElts[r * 2 * cols + 2 * c + 1] }), A.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(rows, cols);
        value = new float[] { 1.5f, 2.7f };
        A.setQuick(0, 0, value);
        value[0] = -3.3f;
        value[1] = 0;
        A.setQuick(3, 5, value);
        value[0] = 222.3f;
        value[1] = -123.9f;
        A.setQuick(11, 22, value);
        value[0] = 0.1123f;
        value[1] = 156.9f;
        A.setQuick(89, 67, value);
        aElts = new float[rows * 2 * cols];
        System.arraycopy((float[]) A.elements(), 0, aElts, 0, rows * 2 * cols);
        A.forEachNonZero(new IntIntFComplexFunction() {
            public float[] apply(int first, int second, float[] third) {
                return FComplex.sqrt(third);
            }
        });
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                AssertUtils.assertArrayEquals(FComplex.sqrt(new float[] { aElts[r * 2 * cols + 2 * c], aElts[r * 2 * cols + 2 * c + 1] }), A.getQuick(r, c), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(rows, cols);
        FComplexMatrix2D Av = A.viewDice();
        value = new float[] { 1.5f, 2.7f };
        Av.setQuick(0, 0, value);
        value[0] = -3.3f;
        value[1] = 0;
        Av.setQuick(3, 5, value);
        value[0] = 222.3f;
        value[1] = -123.9f;
        Av.setQuick(11, 22, value);
        value[0] = 0.1123f;
        value[1] = 156.9f;
        Av.setQuick(89, 67, value);
        aElts = new float[rows * 2 * cols];
        float[] tmp = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp = Av.getQuick(r, c);
                aElts[r * 2 * rows + 2 * c] = tmp[0];
                aElts[r * 2 * rows + 2 * c + 1] = tmp[1];
            }
        }
        Av.forEachNonZero(new IntIntFComplexFunction() {
            public float[] apply(int first, int second, float[] third) {
                return FComplex.sqrt(third);
            }
        });
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp = Av.getQuick(r, c);
                AssertUtils.assertArrayEquals(FComplex.sqrt(new float[] { aElts[r * 2 * rows + 2 * c], aElts[r * 2 * rows + 2 * c + 1] }), tmp, tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(rows, cols);
        Av = A.viewDice();
        value = new float[] { 1.5f, 2.7f };
        Av.setQuick(0, 0, value);
        value[0] = -3.3f;
        value[1] = 0;
        Av.setQuick(3, 5, value);
        value[0] = 222.3f;
        value[1] = -123.9f;
        Av.setQuick(11, 22, value);
        value[0] = 0.1123f;
        value[1] = 156.9f;
        Av.setQuick(89, 67, value);
        aElts = new float[rows * 2 * cols];
        tmp = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp = Av.getQuick(r, c);
                aElts[r * 2 * rows + 2 * c] = tmp[0];
                aElts[r * 2 * rows + 2 * c + 1] = tmp[1];
            }
        }
        Av.forEachNonZero(new IntIntFComplexFunction() {
            public float[] apply(int first, int second, float[] third) {
                return FComplex.sqrt(third);
            }
        });
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmp = Av.getQuick(r, c);
                AssertUtils.assertArrayEquals(FComplex.sqrt(new float[] { aElts[r * 2 * rows + 2 * c], aElts[r * 2 * rows + 2 * c + 1] }), tmp, tol);
            }
        }
    }

    @Test
    public void testGet() {
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        float[] elem;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                elem = A.get(r, c);
                assertEquals(a_2d[r][2 * c], elem[0], tol);
                assertEquals(a_2d[r][2 * c + 1], elem[1], tol);
            }
        }
    }

    @Test
    public void testGetConjugateTranspose() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D B = A.getConjugateTranspose();
        assertEquals(A.rows(), B.columns());
        assertEquals(A.columns(), B.rows());
        float[] aelem = new float[2];
        float[] belem = new float[2];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                aelem = A.getQuick(r, c);
                belem = B.getQuick(c, r);
                assertEquals(aelem[0], belem[0], tol);
                assertEquals(aelem[1], -belem[1], tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        B = A.getConjugateTranspose();
        assertEquals(A.rows(), B.columns());
        assertEquals(A.columns(), B.rows());
        aelem = new float[2];
        belem = new float[2];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                aelem = A.getQuick(r, c);
                belem = B.getQuick(c, r);
                assertEquals(aelem[0], belem[0], tol);
                assertEquals(aelem[1], -belem[1], tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        B = A.getConjugateTranspose();
        FComplexMatrix2D Bv = B.viewDice();
        assertEquals(Av.rows(), Bv.columns());
        assertEquals(Av.columns(), Bv.rows());
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                aelem = Av.getQuick(r, c);
                belem = Bv.getQuick(c, r);
                assertEquals(aelem[0], belem[0], tol);
                assertEquals(aelem[1], -belem[1], tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Av = A.viewDice();
        B = A.getConjugateTranspose();
        Bv = B.viewDice();
        assertEquals(Av.rows(), Bv.columns());
        assertEquals(Av.columns(), Bv.rows());
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                aelem = Av.getQuick(r, c);
                belem = Bv.getQuick(c, r);
                assertEquals(aelem[0], belem[0], tol);
                assertEquals(aelem[1], -belem[1], tol);
            }
        }
    }

    @Test
    public void testGetImaginaryPart() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FloatMatrix2D Im = A.getImaginaryPart();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_2d[r][2 * c + 1], Im.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Im = A.getImaginaryPart();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_2d[r][2 * c + 1], Im.getQuick(r, c), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        Im = Av.getImaginaryPart();
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(a_2dt[r][2 * c + 1], Im.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Av = A.viewDice();
        Im = Av.getImaginaryPart();
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(a_2dt[r][2 * c + 1], Im.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testGetNonZeros() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        IntArrayList rowList = new IntArrayList();
        IntArrayList colList = new IntArrayList();
        ArrayList<float[]> valueList = new ArrayList<float[]>();
        A.getNonZeros(rowList, colList, valueList);
        assertEquals(rows * cols, rowList.size());
        assertEquals(rows * cols, colList.size());
        assertEquals(rows * cols, valueList.size());
        int idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                AssertUtils.assertArrayEquals(A.getQuick(rowList.get(idx), colList.get(idx)), valueList.get(idx), tol);
                idx++;
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        A.getNonZeros(rowList, colList, valueList);
        assertEquals(rows * cols, rowList.size());
        assertEquals(rows * cols, colList.size());
        assertEquals(rows * cols, valueList.size());
        idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                AssertUtils.assertArrayEquals(A.getQuick(rowList.get(idx), colList.get(idx)), valueList.get(idx), tol);
                idx++;
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        Av.getNonZeros(rowList, colList, valueList);
        assertEquals(rows * cols, rowList.size());
        assertEquals(rows * cols, colList.size());
        assertEquals(rows * cols, valueList.size());
        idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                AssertUtils.assertArrayEquals(Av.getQuick(rowList.get(idx), colList.get(idx)), valueList.get(idx), tol);
                idx++;
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Av = A.viewDice();
        Av.getNonZeros(rowList, colList, valueList);
        assertEquals(rows * cols, rowList.size());
        assertEquals(rows * cols, colList.size());
        assertEquals(rows * cols, valueList.size());
        idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                AssertUtils.assertArrayEquals(Av.getQuick(rowList.get(idx), colList.get(idx)), valueList.get(idx), tol);
                idx++;
            }
        }
    }

    @Test
    public void testGetQuick() {
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        float[] elem;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                elem = A.getQuick(r, c);
                assertEquals(a_2d[r][2 * c], elem[0], tol);
                assertEquals(a_2d[r][2 * c + 1], elem[1], tol);
            }
        }
    }

    @Test
    public void testGetRealPart() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FloatMatrix2D R = A.getRealPart();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_2d[r][2 * c], R.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        R = A.getRealPart();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                assertEquals(a_2d[r][2 * c], R.getQuick(r, c), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        R = Av.getRealPart();
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(a_2dt[r][2 * c], R.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Av = A.viewDice();
        R = Av.getRealPart();
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                assertEquals(a_2dt[r][2 * c], R.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testSet() {
        FComplexMatrix2D A = new DenseFComplexMatrix2D(rows, cols);
        float[] elem = new float[] { (float)Math.random(), (float)Math.random() };
        A.set(rows / 2, cols / 2, elem);
        float[] aElem = A.getQuick(rows / 2, cols / 2);
        AssertUtils.assertArrayEquals(elem, aElem, tol);
    }

    @Test
    public void testSetQuickIntIntFloatArray() {
        FComplexMatrix2D A = new DenseFComplexMatrix2D(rows, cols);
        float[] elem = new float[] { (float)Math.random(), (float)Math.random() };
        A.setQuick(rows / 2, cols / 2, elem);
        float[] aElem = A.getQuick(rows / 2, cols / 2);
        AssertUtils.assertArrayEquals(elem, aElem, tol);
    }

    @Test
    public void testSetQuickIntIntFloatFloat() {
        FComplexMatrix2D A = new DenseFComplexMatrix2D(rows, cols);
        float[] elem = new float[] { (float)Math.random(), (float)Math.random() };
        A.setQuick(rows / 2, cols / 2, elem[0], elem[1]);
        float[] aElem = A.getQuick(rows / 2, cols / 2);
        AssertUtils.assertArrayEquals(elem, aElem, tol);
    }

    @Test
    public void testToArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        float[][] array = A.toArray();
        AssertUtils.assertArrayEquals(a_2d, array, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        array = A.toArray();
        AssertUtils.assertArrayEquals(a_2d, array, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        array = Av.toArray();
        AssertUtils.assertArrayEquals(a_2dt, array, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Av = A.viewDice();
        array = Av.toArray();
        AssertUtils.assertArrayEquals(a_2dt, array, tol);

    }

    @Test
    public void testToString() {
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        String s = A.toString();
        System.out.println(s);

    }

    @Test
    public void testVectorize() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix1D B = A.vectorize();
        int idx = 0;
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                AssertUtils.assertArrayEquals(A.getQuick(r, c), B.getQuick(idx++), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        B = A.vectorize();
        idx = 0;
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                AssertUtils.assertArrayEquals(A.getQuick(r, c), B.getQuick(idx++), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        B = Av.vectorize();
        idx = 0;
        for (int c = 0; c < rows; c++) {
            for (int r = 0; r < cols; r++) {
                AssertUtils.assertArrayEquals(Av.getQuick(r, c), B.getQuick(idx++), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Av = A.viewDice();
        B = Av.vectorize();
        idx = 0;
        for (int c = 0; c < rows; c++) {
            for (int r = 0; r < cols; r++) {
                AssertUtils.assertArrayEquals(Av.getQuick(r, c), B.getQuick(idx++), tol);
            }
        }
    }

    @Test
    public void testViewColumn() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix1D B = A.viewColumn(cols / 2);
        assertEquals(rows, B.size());
        float[] tmp;
        for (int i = 0; i < rows; i++) {
            tmp = B.getQuick(i);
            assertEquals(a_2d[i][2 * (cols / 2)], tmp[0], tol);
            assertEquals(a_2d[i][2 * (cols / 2) + 1], tmp[1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        B = A.viewColumn(cols / 2);
        assertEquals(rows, B.size());
        for (int i = 0; i < rows; i++) {
            tmp = B.getQuick(i);
            assertEquals(a_2d[i][2 * (cols / 2)], tmp[0], tol);
            assertEquals(a_2d[i][2 * (cols / 2) + 1], tmp[1], tol);
        }
    }

    @Test
    public void testViewColumnFlip() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D B = A.viewColumnFlip();
        assertEquals(A.size(), B.size());
        float[] tmp;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmp = B.getQuick(r, c);
                assertEquals(a_2d[r][2 * (cols - 1 - c)], tmp[0], tol);
                assertEquals(a_2d[r][2 * (cols - 1 - c) + 1], tmp[1], tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        B = A.viewColumnFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmp = B.getQuick(r, c);
                assertEquals(a_2d[r][2 * (cols - 1 - c)], tmp[0], tol);
                assertEquals(a_2d[r][2 * (cols - 1 - c) + 1], tmp[1], tol);
            }
        }
    }

    @Test
    public void testViewDice() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D B = A.viewDice();
        assertEquals(A.rows(), B.columns());
        assertEquals(A.columns(), B.rows());
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                AssertUtils.assertArrayEquals(A.getQuick(r, c), B.getQuick(c, r), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        B = A.viewDice();
        assertEquals(A.rows(), B.columns());
        assertEquals(A.columns(), B.rows());
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                AssertUtils.assertArrayEquals(A.getQuick(r, c), B.getQuick(c, r), tol);
            }
        }
    }

    @Test
    public void testViewPart() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D B = A.viewPart(15, 11, 21, 27);
        for (int r = 0; r < 21; r++) {
            for (int c = 0; c < 27; c++) {
                AssertUtils.assertArrayEquals(A.getQuick(15 + r, 11 + c), B.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        B = A.viewPart(15, 11, 21, 27);
        for (int r = 0; r < 21; r++) {
            for (int c = 0; c < 27; c++) {
                AssertUtils.assertArrayEquals(A.getQuick(15 + r, 11 + c), B.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testViewRow() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix1D B = A.viewRow(rows / 2);
        assertEquals(cols, B.size());
        float[] tmp;
        for (int i = 0; i < cols; i++) {
            tmp = B.getQuick(i);
            assertEquals(a_2d[rows / 2][2 * i], tmp[0], tol);
            assertEquals(a_2d[rows / 2][2 * i + 1], tmp[1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        B = A.viewRow(rows / 2);
        assertEquals(cols, B.size());
        for (int i = 0; i < cols; i++) {
            tmp = B.getQuick(i);
            assertEquals(a_2d[rows / 2][2 * i], tmp[0], tol);
            assertEquals(a_2d[rows / 2][2 * i + 1], tmp[1], tol);
        }
    }

    @Test
    public void testViewRowFlip() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D B = A.viewRowFlip();
        assertEquals(A.size(), B.size());
        float[] tmp;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmp = B.getQuick(r, c);
                assertEquals(a_2d[rows - 1 - r][2 * c], tmp[0], tol);
                assertEquals(a_2d[rows - 1 - r][2 * c + 1], tmp[1], tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        B = A.viewRowFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmp = B.getQuick(r, c);
                assertEquals(a_2d[rows - 1 - r][2 * c], tmp[0], tol);
                assertEquals(a_2d[rows - 1 - r][2 * c + 1], tmp[1], tol);
            }
        }
    }

    @Test
    public void testViewSelectionComplexMatrix1DProcedure() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        final float[] value = new float[] { 2, 3 };
        A.setQuick(rows / 4, 0, value);
        A.setQuick(rows / 2, 0, value);
        FComplexMatrix2D B = A.viewSelection(new FComplexMatrix1DProcedure() {
            public boolean apply(FComplexMatrix1D element) {
                return FComplex.isEqual(element.getQuick(0), value, tol);
            }
        });
        assertEquals(2, B.rows());
        assertEquals(A.columns(), B.columns());
        AssertUtils.assertArrayEquals(A.getQuick(rows / 4, 0), B.getQuick(0, 0), tol);
        AssertUtils.assertArrayEquals(A.getQuick(rows / 2, 0), B.getQuick(1, 0), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        A.setQuick(rows / 4, 0, value);
        A.setQuick(rows / 2, 0, value);
        B = A.viewSelection(new FComplexMatrix1DProcedure() {
            public boolean apply(FComplexMatrix1D element) {
                return FComplex.isEqual(element.getQuick(0), value, tol);
            }
        });
        assertEquals(2, B.rows());
        assertEquals(A.columns(), B.columns());
        AssertUtils.assertArrayEquals(A.getQuick(rows / 4, 0), B.getQuick(0, 0), tol);
        AssertUtils.assertArrayEquals(A.getQuick(rows / 2, 0), B.getQuick(1, 0), tol);
    }

    @Test
    public void testViewSelectionIntArrayIntArray() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        int[] rowIndexes = new int[] { 5, 11, 22, 37, 101 };
        int[] colIndexes = new int[] { 2, 17, 32, 47, 99, 111 };
        FComplexMatrix2D B = A.viewSelection(rowIndexes, colIndexes);
        assertEquals(rowIndexes.length, B.rows());
        assertEquals(colIndexes.length, B.columns());
        for (int r = 0; r < rowIndexes.length; r++) {
            for (int c = 0; c < colIndexes.length; c++) {
                AssertUtils.assertArrayEquals(A.getQuick(rowIndexes[r], colIndexes[c]), B.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        rowIndexes = new int[] { 5, 11, 22, 37, 101 };
        colIndexes = new int[] { 2, 17, 32, 47, 99, 111 };
        B = A.viewSelection(rowIndexes, colIndexes);
        assertEquals(rowIndexes.length, B.rows());
        assertEquals(colIndexes.length, B.columns());
        for (int r = 0; r < rowIndexes.length; r++) {
            for (int c = 0; c < colIndexes.length; c++) {
                AssertUtils.assertArrayEquals(A.getQuick(rowIndexes[r], colIndexes[c]), B.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testViewStrides() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        int rowStride = 3;
        int colStride = 5;
        FComplexMatrix2D B = A.viewStrides(rowStride, colStride);
        for (int r = 0; r < B.rows(); r++) {
            for (int c = 0; c < B.columns(); c++) {
                AssertUtils.assertArrayEquals(A.getQuick(r * rowStride, c * colStride), B.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        B = A.viewStrides(rowStride, colStride);
        for (int r = 0; r < B.rows(); r++) {
            for (int c = 0; c < B.columns(); c++) {
                AssertUtils.assertArrayEquals(A.getQuick(r * rowStride, c * colStride), B.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testZMultComplexMatrix1FComplexMatrix1D() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix1D y = new DenseFComplexMatrix1D(A.columns());
        for (int i = 0; i < y.size(); i++) {
            y.set(i, rand.nextFloat(), rand.nextFloat());
        }
        FComplexMatrix1D z = new DenseFComplexMatrix1D(A.rows());
        A.zMult(y, z);
        float[][] tmpMatVec = new float[A.rows()][2];
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                tmpMatVec[r] = FComplex.plus(tmpMatVec[r], FComplex.mult(A.getQuick(r, c), y.getQuick(c)));
            }
        }
        for (int r = 0; r < A.rows(); r++) {
            AssertUtils.assertArrayEquals(tmpMatVec[r], z.getQuick(r), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        y = new DenseFComplexMatrix1D(A.columns());
        for (int i = 0; i < y.size(); i++) {
            y.set(i, rand.nextFloat(), rand.nextFloat());
        }
        z = new DenseFComplexMatrix1D(A.rows());
        A.zMult(y, z);
        tmpMatVec = new float[A.rows()][2];
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                tmpMatVec[r] = FComplex.plus(tmpMatVec[r], FComplex.mult(A.getQuick(r, c), y.getQuick(c)));
            }
        }
        for (int r = 0; r < A.rows(); r++) {
            AssertUtils.assertArrayEquals(tmpMatVec[r], z.getQuick(r), tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        y = new DenseFComplexMatrix1D(Av.columns());
        for (int i = 0; i < y.size(); i++) {
            y.set(i, rand.nextFloat(), rand.nextFloat());
        }
        z = new DenseFComplexMatrix1D(Av.rows());
        Av.zMult(y, z);
        tmpMatVec = new float[Av.rows()][2];
        for (int r = 0; r < Av.rows(); r++) {
            for (int c = 0; c < Av.columns(); c++) {
                tmpMatVec[r] = FComplex.plus(tmpMatVec[r], FComplex.mult(Av.getQuick(r, c), y.getQuick(c)));
            }
        }
        for (int r = 0; r < Av.rows(); r++) {
            AssertUtils.assertArrayEquals(tmpMatVec[r], z.getQuick(r), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Av = A.viewDice();
        y = new DenseFComplexMatrix1D(Av.columns());
        for (int i = 0; i < y.size(); i++) {
            y.set(i, rand.nextFloat(), rand.nextFloat());
        }
        z = new DenseFComplexMatrix1D(Av.rows());
        Av.zMult(y, z);
        tmpMatVec = new float[Av.rows()][2];
        for (int r = 0; r < Av.rows(); r++) {
            for (int c = 0; c < Av.columns(); c++) {
                tmpMatVec[r] = FComplex.plus(tmpMatVec[r], FComplex.mult(Av.getQuick(r, c), y.getQuick(c)));
            }
        }
        for (int r = 0; r < Av.rows(); r++) {
            AssertUtils.assertArrayEquals(tmpMatVec[r], z.getQuick(r), tol);
        }
    }

    @Test
    public void testZMultComplexMatrix2FComplexMatrix2D() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D B = new DenseFComplexMatrix2D(b_2d);
        B = B.viewDice().copy();
        FComplexMatrix2D C = new DenseFComplexMatrix2D(rows, rows);
        A.zMult(B, C);
        float[][][] tmpMatMat = new float[rows][rows][2];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < rows; c++) {
                for (int k = 0; k < cols; k++) {
                    tmpMatMat[c][r] = FComplex.plus(tmpMatMat[c][r], FComplex.mult(A.getQuick(c, k), B.getQuick(k, r)));
                }
            }
        }
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < rows; c++) {
                AssertUtils.assertArrayEquals(tmpMatMat[r][c], C.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        B = new DenseFComplexMatrix2D(b_2d);
        B = B.viewDice().copy();
        C = new DenseFComplexMatrix2D(rows, rows);
        A.zMult(B, C);
        tmpMatMat = new float[rows][rows][2];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < rows; c++) {
                for (int k = 0; k < cols; k++) {
                    tmpMatMat[c][r] = FComplex.plus(tmpMatMat[c][r], FComplex.mult(A.getQuick(c, k), B.getQuick(k, r)));
                }
            }
        }
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < rows; c++) {
                AssertUtils.assertArrayEquals(tmpMatMat[r][c], C.getQuick(r, c), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        B = new DenseFComplexMatrix2D(b_2d);
        B = B.viewDice().copy();
        FComplexMatrix2D Bv = B.viewDice();
        C = new DenseFComplexMatrix2D(cols, cols);
        FComplexMatrix2D Cv = C.viewDice();
        Av.zMult(Bv, Cv);
        tmpMatMat = new float[cols][cols][2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < cols; c++) {
                for (int k = 0; k < rows; k++) {
                    tmpMatMat[c][r] = FComplex.plus(tmpMatMat[c][r], FComplex.mult(Av.getQuick(c, k), Bv.getQuick(k, r)));
                }
            }
        }
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < cols; c++) {
                AssertUtils.assertArrayEquals(tmpMatMat[r][c], Cv.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Av = A.viewDice();
        B = new DenseFComplexMatrix2D(b_2d);
        B = B.viewDice().copy();
        Bv = B.viewDice();
        C = new DenseFComplexMatrix2D(cols, cols);
        Cv = C.viewDice();
        Av.zMult(Bv, Cv);
        tmpMatMat = new float[cols][cols][2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < cols; c++) {
                for (int k = 0; k < rows; k++) {
                    tmpMatMat[c][r] = FComplex.plus(tmpMatMat[c][r], FComplex.mult(Av.getQuick(c, k), Bv.getQuick(k, r)));
                }
            }
        }
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < cols; c++) {
                AssertUtils.assertArrayEquals(tmpMatMat[r][c], Cv.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testZSum() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix2D A = new DenseFComplexMatrix2D(a_2d);
        float[] aSum = A.zSum();
        float[] tmpSum = new float[2];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmpSum[0] += a_2d[r][2 * c];
                tmpSum[1] += a_2d[r][2 * c + 1];
            }
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        aSum = A.zSum();
        tmpSum = new float[2];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                tmpSum[0] += a_2d[r][2 * c];
                tmpSum[1] += a_2d[r][2 * c + 1];
            }
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix2D(a_2d);
        FComplexMatrix2D Av = A.viewDice();
        aSum = Av.zSum();
        tmpSum = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmpSum[0] += a_2dt[r][2 * c];
                tmpSum[1] += a_2dt[r][2 * c + 1];
            }
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_2D(nThreadsBegin);
        A = new DenseFComplexMatrix2D(a_2d);
        Av = A.viewDice();
        aSum = Av.zSum();
        tmpSum = new float[2];
        for (int r = 0; r < cols; r++) {
            for (int c = 0; c < rows; c++) {
                tmpSum[0] += a_2dt[r][2 * c];
                tmpSum[1] += a_2dt[r][2 * c + 1];
            }
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
    }

}
