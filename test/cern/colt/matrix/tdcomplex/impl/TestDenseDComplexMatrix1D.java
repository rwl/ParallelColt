package cern.colt.matrix.tdcomplex.impl;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Random;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import cern.colt.function.tdcomplex.DComplexProcedure;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.DComplexMatrix3D;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix1D;
import cern.colt.matrix.tdouble.DoubleFactory1D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.jet.math.tdcomplex.DComplex;
import cern.jet.math.tdcomplex.DComplexFunctions;
import edu.emory.mathcs.utils.AssertUtils;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public class TestDenseDComplexMatrix1D {
    private static final int size = 2 * 17 * 5;

    private static final double tol = 1e-10;

    private static final int nThreads = 3;

    private static final int nThreadsBegin = 1;

    private double[] a, b;

    private static final DoubleFactory1D factory = DoubleFactory1D.dense;

    @Before
    public void setUpBeforeClass() throws Exception {
        // generate test matrices
        Random r = new Random(0);

        a = new double[2 * size];
        for (int i = 0; i < a.length; i++) {
            a[i] = r.nextDouble();
        }

        b = new double[2 * size];
        for (int i = 0; i < b.length; i++) {
            b[i] = r.nextDouble();
        }
    }

    @After
    public void tearDownAfterClass() throws Exception {
        a = null;
        b = null;
        System.gc();
    }

    @Test
    public void testAggregateComplexComplexFunctionComplexFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        double[] aSum = A.aggregate(DComplexFunctions.plus, DComplexFunctions.square);
        double[] tmpSum = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSum = DComplex.plus(tmpSum, DComplex.square(new double[] { a[i], a[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        aSum = A.aggregate(DComplexFunctions.plus, DComplexFunctions.square);
        tmpSum = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSum = DComplex.plus(tmpSum, DComplex.square(new double[] { a[i], a[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        aSum = Av.aggregate(DComplexFunctions.plus, DComplexFunctions.square);
        tmpSum = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSum = DComplex.plus(tmpSum, DComplex.square(new double[] { a[i], a[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        aSum = Av.aggregate(DComplexFunctions.plus, DComplexFunctions.square);
        tmpSum = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSum = DComplex.plus(tmpSum, DComplex.square(new double[] { a[i], a[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);

    }

    @Test
    public void testAggregateComplexMatrix1DComplexComplexFunctionComplexComplexFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D B = new DenseDComplexMatrix1D(b);
        double[] sumMult = A.aggregate(B, DComplexFunctions.plus, DComplexFunctions.mult);
        double[] tmpSumMult = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSumMult = DComplex.plus(tmpSumMult, DComplex.mult(new double[] { a[i], a[i + 1] }, new double[] { b[i], b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, sumMult, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        B = new DenseDComplexMatrix1D(b);
        sumMult = A.aggregate(B, DComplexFunctions.plus, DComplexFunctions.mult);
        tmpSumMult = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSumMult = DComplex.plus(tmpSumMult, DComplex.mult(new double[] { a[i], a[i + 1] }, new double[] { b[i], b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, sumMult, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        B = new DenseDComplexMatrix1D(b);
        DComplexMatrix1D Bv = B.viewFlip();
        sumMult = Av.aggregate(Bv, DComplexFunctions.plus, DComplexFunctions.mult);
        tmpSumMult = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSumMult = DComplex.plus(tmpSumMult, DComplex.mult(new double[] { a[i], a[i + 1] }, new double[] { b[i], b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, sumMult, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseDComplexMatrix1D(b);
        Bv = B.viewFlip();
        sumMult = Av.aggregate(Bv, DComplexFunctions.plus, DComplexFunctions.mult);
        tmpSumMult = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSumMult = DComplex.plus(tmpSumMult, DComplex.mult(new double[] { a[i], a[i + 1] }, new double[] { b[i], b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, sumMult, tol);
    }

    @Test
    public void testAssignComplexComplexFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        A.assign(DComplexFunctions.acos);
        double[] aElts = (double[]) A.elements();
        double[] tmp = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            tmp = DComplex.acos(tmp);
            assertEquals(tmp[0], aElts[i], tol);
            assertEquals(tmp[1], aElts[i + 1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        A.assign(DComplexFunctions.acos);
        aElts = (double[]) A.elements();
        tmp = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            tmp = DComplex.acos(tmp);
            assertEquals(tmp[0], aElts[i], tol);
            assertEquals(tmp[1], aElts[i + 1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        Av.assign(DComplexFunctions.acos);
        tmp = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            tmp = DComplex.acos(tmp);
            AssertUtils.assertArrayEquals(tmp, Av.getQuick(size - 1 - (i / 2)), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        Av.assign(DComplexFunctions.acos);
        tmp = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            tmp = DComplex.acos(tmp);
            AssertUtils.assertArrayEquals(tmp, Av.getQuick(size - 1 - (i / 2)), tol);
        }
    }

    @Test
    public void testAssignComplexMatrix1D() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(size);
        DComplexMatrix1D B = new DenseDComplexMatrix1D(a);
        A.assign(B);
        double[] aElts = (double[]) A.elements();
        AssertUtils.assertArrayEquals(a, aElts, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(size);
        B = new DenseDComplexMatrix1D(a);
        A.assign(B);
        aElts = (double[]) A.elements();
        AssertUtils.assertArrayEquals(a, aElts, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(size);
        DComplexMatrix1D Av = A.viewFlip();
        B = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Bv = B.viewFlip();
        Av.assign(Bv);
        double[] tmp = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            AssertUtils.assertArrayEquals(tmp, Av.getQuick(size - 1 - (i / 2)), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(size);
        Av = A.viewFlip();
        B = new DenseDComplexMatrix1D(a);
        Bv = B.viewFlip();
        Av.assign(Bv);
        tmp = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            AssertUtils.assertArrayEquals(tmp, Av.getQuick(size - 1 - (i / 2)), tol);
        }
    }

    @Test
    public void testAssignComplexMatrix1DComplexComplexFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D B = new DenseDComplexMatrix1D(b);
        A.assign(B, DComplexFunctions.div);
        double[] aElts = (double[]) A.elements();
        double[] tmp1 = new double[2];
        double[] tmp2 = new double[2];
        for (int i = 0; i < aElts.length; i = i + 2) {
            tmp1[0] = a[i];
            tmp1[1] = a[i + 1];
            tmp2[0] = b[i];
            tmp2[1] = b[i + 1];
            tmp1 = DComplex.div(tmp1, tmp2);
            assertEquals(tmp1[0], aElts[i], tol);
            assertEquals(tmp1[1], aElts[i + 1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        B = new DenseDComplexMatrix1D(b);
        A.assign(B, DComplexFunctions.div);
        aElts = (double[]) A.elements();
        tmp1 = new double[2];
        tmp2 = new double[2];
        for (int i = 0; i < aElts.length; i = i + 2) {
            tmp1[0] = a[i];
            tmp1[1] = a[i + 1];
            tmp2[0] = b[i];
            tmp2[1] = b[i + 1];
            tmp1 = DComplex.div(tmp1, tmp2);
            assertEquals(tmp1[0], aElts[i], tol);
            assertEquals(tmp1[1], aElts[i + 1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        B = new DenseDComplexMatrix1D(b);
        DComplexMatrix1D Bv = B.viewFlip();
        Av.assign(Bv, DComplexFunctions.div);
        tmp1 = new double[2];
        tmp2 = new double[2];
        for (int i = 0; i < aElts.length; i = i + 2) {
            tmp1[0] = a[i];
            tmp1[1] = a[i + 1];
            tmp2[0] = b[i];
            tmp2[1] = b[i + 1];
            tmp1 = DComplex.div(tmp1, tmp2);
            AssertUtils.assertArrayEquals(tmp1, Av.getQuick(size - 1 - (i / 2)), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseDComplexMatrix1D(b);
        Bv = B.viewFlip();
        Av.assign(Bv, DComplexFunctions.div);
        tmp1 = new double[2];
        tmp2 = new double[2];
        for (int i = 0; i < aElts.length; i = i + 2) {
            tmp1[0] = a[i];
            tmp1[1] = a[i + 1];
            tmp2[0] = b[i];
            tmp2[1] = b[i + 1];
            tmp1 = DComplex.div(tmp1, tmp2);
            AssertUtils.assertArrayEquals(tmp1, Av.getQuick(size - 1 - (i / 2)), tol);
        }
    }

    @Test
    public void testAssignComplexProcedureComplexComplexFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D B = A.copy();
        A.assign(new DComplexProcedure() {
            public boolean apply(double[] element) {
                if (DComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, DComplexFunctions.tan);
        for (int i = 0; i < size; i++) {
            if (DComplex.abs(B.getQuick(i)) > 3) {
                B.setQuick(i, DComplex.tan(B.getQuick(i)));
            }
        }
        AssertUtils.assertArrayEquals((double[]) B.elements(), (double[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        B = A.copy();
        A.assign(new DComplexProcedure() {
            public boolean apply(double[] element) {
                if (DComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, DComplexFunctions.tan);
        for (int i = 0; i < size; i++) {
            if (DComplex.abs(B.getQuick(i)) > 3) {
                B.setQuick(i, DComplex.tan(B.getQuick(i)));
            }
        }
        AssertUtils.assertArrayEquals((double[]) B.elements(), (double[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        B = A.copy();
        DComplexMatrix1D Bv = B.viewFlip();
        Av.assign(new DComplexProcedure() {
            public boolean apply(double[] element) {
                if (DComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, DComplexFunctions.tan);
        for (int i = 0; i < size; i++) {
            if (DComplex.abs(Bv.getQuick(i)) > 3) {
                Bv.setQuick(i, DComplex.tan(Bv.getQuick(i)));
            }
        }
        for (int i = 0; i < B.size(); i++) {
            AssertUtils.assertArrayEquals(Bv.getQuick(i), Av.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        B = A.copy();
        Bv = B.viewFlip();
        Av.assign(new DComplexProcedure() {
            public boolean apply(double[] element) {
                if (DComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, DComplexFunctions.tan);
        for (int i = 0; i < size; i++) {
            if (DComplex.abs(Bv.getQuick(i)) > 3) {
                Bv.setQuick(i, DComplex.tan(Bv.getQuick(i)));
            }
        }
        for (int i = 0; i < B.size(); i++) {
            AssertUtils.assertArrayEquals(Bv.getQuick(i), Av.getQuick(i), tol);
        }
    }

    @Test
    public void testAssignComplexProcedureDoubleArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D B = A.copy();
        A.assign(new DComplexProcedure() {
            public boolean apply(double[] element) {
                if (DComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, new double[] { -1, -1 });
        for (int i = 0; i < size; i++) {
            if (DComplex.abs(B.getQuick(i)) > 3) {
                B.setQuick(i, new double[] { -1, -1 });
            }
        }
        AssertUtils.assertArrayEquals((double[]) B.elements(), (double[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        B = A.copy();
        A.assign(new DComplexProcedure() {
            public boolean apply(double[] element) {
                if (DComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, new double[] { -1, -1 });
        for (int i = 0; i < size; i++) {
            if (DComplex.abs(B.getQuick(i)) > 3) {
                B.setQuick(i, new double[] { -1, -1 });
            }
        }
        AssertUtils.assertArrayEquals((double[]) B.elements(), (double[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        B = A.copy();
        DComplexMatrix1D Bv = B.viewFlip();
        Av.assign(new DComplexProcedure() {
            public boolean apply(double[] element) {
                if (DComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, new double[] { -1, -1 });
        for (int i = 0; i < size; i++) {
            if (DComplex.abs(Bv.getQuick(i)) > 3) {
                Bv.setQuick(i, new double[] { -1, -1 });
            }
        }
        for (int i = 0; i < B.size(); i++) {
            AssertUtils.assertArrayEquals(Bv.getQuick(i), Av.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        B = A.copy();
        Bv = B.viewFlip();
        Av.assign(new DComplexProcedure() {
            public boolean apply(double[] element) {
                if (DComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, new double[] { -1, -1 });
        for (int i = 0; i < size; i++) {
            if (DComplex.abs(Bv.getQuick(i)) > 3) {
                Bv.setQuick(i, new double[] { -1, -1 });
            }
        }
        for (int i = 0; i < B.size(); i++) {
            AssertUtils.assertArrayEquals(Bv.getQuick(i), Av.getQuick(i), tol);
        }
    }

    @Test
    public void testAssignComplexRealFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        A.assign(DComplexFunctions.abs);
        double[] aElts = (double[]) A.elements();
        double[] tmp1 = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp1[0] = a[i];
            tmp1[1] = a[i + 1];
            tmp1[0] = DComplex.abs(tmp1);
            assertEquals(tmp1[0], aElts[i], tol);
            assertEquals(0.0, aElts[i + 1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        A.assign(DComplexFunctions.abs);
        aElts = (double[]) A.elements();
        tmp1 = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp1[0] = a[i];
            tmp1[1] = a[i + 1];
            tmp1[0] = DComplex.abs(tmp1);
            assertEquals(tmp1[0], aElts[i], tol);
            assertEquals(0.0, aElts[i + 1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        Av.assign(DComplexFunctions.abs);
        tmp1 = new double[2];
        double[] tmp2 = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp1[0] = a[i];
            tmp1[1] = a[i + 1];
            tmp1[0] = DComplex.abs(tmp1);
            tmp2 = Av.getQuick(size - 1 - i / 2);
            assertEquals(tmp1[0], tmp2[0], tol);
            assertEquals(0.0, tmp2[1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        Av.assign(DComplexFunctions.abs);
        tmp1 = new double[2];
        tmp2 = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp1[0] = a[i];
            tmp1[1] = a[i + 1];
            tmp1[0] = DComplex.abs(tmp1);
            tmp2 = Av.getQuick(size - 1 - i / 2);
            assertEquals(tmp1[0], tmp2[0], tol);
            assertEquals(0.0, tmp2[1], tol);
        }
    }

    @Test
    public void testAssignDoubleArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(size);
        A.assign(a);
        double[] aElts = (double[]) A.elements();
        AssertUtils.assertArrayEquals(a, aElts, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(size);
        A.assign(a);
        aElts = (double[]) A.elements();
        AssertUtils.assertArrayEquals(a, aElts, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(size);
        DComplexMatrix1D Av = A.viewFlip();
        Av.assign(a);
        double[] tmp = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            AssertUtils.assertArrayEquals(tmp, Av.getQuick(i / 2), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(size);
        Av = A.viewFlip();
        Av.assign(a);
        tmp = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            AssertUtils.assertArrayEquals(tmp, Av.getQuick(i / 2), tol);
        }

    }

    @Test
    public void testAssignDoubleDouble() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(size);
        double[] value = new double[] { Math.random(), Math.random() };
        A.assign(value[0], value[1]);
        double[] aElts = (double[]) A.elements();
        for (int i = 0; i < aElts.length; i = i + 2) {
            assertEquals(value[0], aElts[i], tol);
            assertEquals(value[1], aElts[i + 1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(size);
        A.assign(value[0], value[1]);
        aElts = (double[]) A.elements();
        for (int i = 0; i < aElts.length; i = i + 2) {
            assertEquals(value[0], aElts[i], tol);
            assertEquals(value[1], aElts[i + 1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(size);
        DComplexMatrix1D Av = A.viewFlip();
        value = new double[] { Math.random(), Math.random() };
        Av.assign(value[0], value[1]);
        double[] tmp = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp = Av.getQuick(i / 2);
            AssertUtils.assertArrayEquals(value, tmp, tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(size);
        Av = A.viewFlip();
        value = new double[] { Math.random(), Math.random() };
        Av.assign(value[0], value[1]);
        tmp = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp = Av.getQuick(i / 2);
            AssertUtils.assertArrayEquals(value, tmp, tol);
        }

    }

    @Test
    public void testAssignImaginary() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(size);
        DoubleMatrix1D Im = factory.random(size);
        A.assignImaginary(Im);
        int idx = 0;
        double[] aElts = (double[]) A.elements();
        for (int i = 0; i < aElts.length; i = i + 2) {
            assertEquals(0, aElts[i], tol);
            assertEquals(Im.getQuick(idx++), aElts[i + 1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(size);
        Im = factory.random(size);
        A.assignImaginary(Im);
        aElts = (double[]) A.elements();
        idx = 0;
        for (int i = 0; i < aElts.length; i = i + 2) {
            assertEquals(0, aElts[i], tol);
            assertEquals(Im.getQuick(idx++), aElts[i + 1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(size);
        DComplexMatrix1D Av = A.viewFlip();
        Im = factory.random(size);
        Av.assignImaginary(Im);
        idx = 0;
        double[] tmp = new double[2];
        for (int i = 0; i < Av.size(); i++) {
            tmp = Av.getQuick(i);
            assertEquals(0, tmp[0], tol);
            assertEquals(Im.getQuick(idx++), tmp[1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(size);
        Av = A.viewFlip();
        Im = factory.random(size);
        Av.assignImaginary(Im);
        idx = 0;
        tmp = new double[2];
        for (int i = 0; i < Av.size(); i++) {
            tmp = Av.getQuick(i);
            assertEquals(0, tmp[0], tol);
            assertEquals(Im.getQuick(idx++), tmp[1], tol);
        }

    }

    @Test
    public void testAssignReal() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(size);
        DoubleMatrix1D Re = factory.random(size);
        A.assignReal(Re);
        int idx = 0;
        double[] aElts = (double[]) A.elements();
        for (int i = 0; i < aElts.length; i = i + 2) {
            assertEquals(Re.getQuick(idx++), aElts[i], tol);
            assertEquals(0, aElts[i + 1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(size);
        Re = factory.random(size);
        A.assignReal(Re);
        idx = 0;
        aElts = (double[]) A.elements();
        for (int i = 0; i < aElts.length; i = i + 2) {
            assertEquals(Re.getQuick(idx++), aElts[i], tol);
            assertEquals(0, aElts[i + 1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(size);
        DComplexMatrix1D Av = A.viewFlip();
        Re = factory.random(size);
        Av.assignReal(Re);
        idx = 0;
        double[] tmp = new double[2];
        for (int i = 0; i < Av.size(); i++) {
            tmp = Av.getQuick(i);
            assertEquals(Re.getQuick(idx++), tmp[0], tol);
            assertEquals(0, tmp[1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(size);
        Av = A.viewFlip();
        Re = factory.random(size);
        Av.assignReal(Re);
        idx = 0;
        tmp = new double[2];
        for (int i = 0; i < Av.size(); i++) {
            tmp = Av.getQuick(i);
            assertEquals(Re.getQuick(idx++), tmp[0], tol);
            assertEquals(0, tmp[1], tol);
        }
    }

    @Test
    public void testCardinality() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        int card = A.cardinality();
        assertEquals(a.length / 2, card);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        card = A.cardinality();
        assertEquals(a.length / 2, card);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        card = Av.cardinality();
        assertEquals(a.length / 2, card);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        card = Av.cardinality();
        assertEquals(a.length / 2, card);
    }

    @Test
    public void testCopy() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D B = A.copy();
        double[] bElts = (double[]) B.elements();
        AssertUtils.assertArrayEquals(a, bElts, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        B = A.copy();
        bElts = (double[]) B.elements();
        AssertUtils.assertArrayEquals(a, bElts, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        B = Av.copy();
        for (int i = 0; i < Av.size(); i++) {
            AssertUtils.assertArrayEquals(Av.get(i), B.get(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        B = Av.copy();
        for (int i = 0; i < Av.size(); i++) {
            AssertUtils.assertArrayEquals(Av.get(i), B.get(i), tol);
        }
    }

    @Test
    public void testEqualsDoubleArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(size);
        double[] value = new double[] { 1, 1 };
        A.assign(1, 1);
        boolean eq = A.equals(value);
        assertEquals(true, eq);
        eq = A.equals(new double[] { 2, 1 });
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(size);
        value = new double[] { 1, 1 };
        A.assign(1, 1);
        eq = A.equals(value);
        assertEquals(true, eq);
        eq = A.equals(new double[] { 2, 1 });
        assertEquals(false, eq);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(size);
        DComplexMatrix1D Av = A.viewFlip();
        value = new double[] { 1, 1 };
        Av.assign(1, 1);
        eq = A.equals(value);
        assertEquals(true, eq);
        eq = Av.equals(new double[] { 2, 1 });
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(size);
        Av = A.viewFlip();
        value = new double[] { 1, 1 };
        Av.assign(1, 1);
        eq = A.equals(value);
        assertEquals(true, eq);
        eq = Av.equals(new double[] { 2, 1 });
        assertEquals(false, eq);
    }
    

    @Test
    public void testEqualsObject() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D B = new DenseDComplexMatrix1D(b);
        boolean eq = A.equals(A);
        assertEquals(true, eq);
        eq = A.equals(B);
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        B = new DenseDComplexMatrix1D(b);
        eq = A.equals(A);
        assertEquals(true, eq);
        eq = A.equals(B);
        assertEquals(false, eq);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        B = new DenseDComplexMatrix1D(b);
        DComplexMatrix1D Bv = B.viewFlip();
        eq = Av.equals(Av);
        assertEquals(true, eq);
        eq = Av.equals(Bv);
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseDComplexMatrix1D(b);
        Bv = B.viewFlip();
        eq = Av.equals(Av);
        assertEquals(true, eq);
        eq = Av.equals(Bv);
        assertEquals(false, eq);
    }
    
    @Test
    public void testFft() {
        int size = 2048;
        double[] a_1d = new double[size];
        for (int i = 0; i < size; i++) {
            a_1d[i] = Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(size / 2);
        A.assign(a_1d);
        A.fft();
        A.ifft(true);
        AssertUtils.assertArrayEquals(a_1d, (double[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(nThreadsBegin);
        A = new DenseDComplexMatrix1D(size / 2);
        A.assign(a_1d);
        A.fft();
        A.ifft(true);
        AssertUtils.assertArrayEquals(a_1d, (double[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
        A = new DenseDComplexMatrix1D(size / 2);
        A.assign(a_1d);
        DComplexMatrix1D Av = A.viewPart(size / 4 - 1, size / 4);
        double[] av_elems = (double[]) Av.copy().elements();
        Av.fft();
        Av.ifft(true);
        AssertUtils.assertArrayEquals(av_elems, (double[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(nThreadsBegin);
        A = new DenseDComplexMatrix1D(size / 2);
        A.assign(a_1d);
        Av = A.viewPart(size / 4 - 1, size / 4);
        av_elems = (double[]) Av.copy().elements();
        Av.fft();
        Av.ifft(true);
        AssertUtils.assertArrayEquals(av_elems, (double[]) Av.copy().elements(), tol);
    }

    @Test
    public void testGet() {
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        double[] elem;
        for (int i = 0; i < A.size(); i++) {
            elem = A.get(i);
            assertEquals(a[2 * i], elem[0], tol);
            assertEquals(a[2 * i + 1], elem[1], tol);
        }
    }

    @Test
    public void testGetImaginaryPart() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        DoubleMatrix1D Im = A.getImaginaryPart();
        for (int i = 0; i < size; i++) {
            assertEquals(a[2 * i + 1], Im.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Im = A.getImaginaryPart();
        for (int i = 0; i < size; i++) {
            assertEquals(a[2 * i + 1], Im.getQuick(i), tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        Im = Av.getImaginaryPart();
        for (int i = 0; i < size; i++) {
            assertEquals(a[2 * i + 1], Im.getQuick(size - 1 - i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        Im = Av.getImaginaryPart();
        for (int i = 0; i < size; i++) {
            assertEquals(a[2 * i + 1], Im.getQuick(size - 1 - i), tol);
        }
    }

    @Test
    public void testGetNonZerosIntArrayListArrayListOfdouble() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        IntArrayList indexList = new IntArrayList();
        ArrayList<double[]> valueList = new ArrayList<double[]>();
        A.getNonZeros(indexList, valueList);
        assertEquals(size, indexList.size());
        assertEquals(size, valueList.size());
        for (int i = 0; i < size; i++) {
            AssertUtils.assertArrayEquals(A.getQuick(indexList.get(i)), valueList.get(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        A.getNonZeros(indexList, valueList);
        assertEquals(size, indexList.size());
        assertEquals(size, valueList.size());
        for (int i = 0; i < size; i++) {
            AssertUtils.assertArrayEquals(A.getQuick(indexList.get(i)), valueList.get(i), tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        Av.getNonZeros(indexList, valueList);
        assertEquals(size, indexList.size());
        assertEquals(size, valueList.size());
        for (int i = 0; i < size; i++) {
            AssertUtils.assertArrayEquals(Av.getQuick(indexList.get(i)), valueList.get(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        Av.getNonZeros(indexList, valueList);
        assertEquals(size, indexList.size());
        assertEquals(size, valueList.size());
        for (int i = 0; i < size; i++) {
            AssertUtils.assertArrayEquals(Av.getQuick(indexList.get(i)), valueList.get(i), tol);
        }
    }

    @Test
    public void testGetQuick() {
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        double[] elem;
        for (int i = 0; i < A.size(); i++) {
            elem = A.getQuick(i);
            assertEquals(a[2 * i], elem[0], tol);
            assertEquals(a[2 * i + 1], elem[1], tol);
        }
    }

    @Test
    public void testGetRealPart() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        DoubleMatrix1D R = A.getRealPart();
        for (int i = 0; i < size; i++) {
            assertEquals(a[2 * i], R.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        R = A.getRealPart();
        for (int i = 0; i < size; i++) {
            assertEquals(a[2 * i], R.getQuick(i), tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        R = Av.getRealPart();
        for (int i = 0; i < size; i++) {
            assertEquals(a[2 * i], R.getQuick(size - 1 - i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        R = Av.getRealPart();
        for (int i = 0; i < size; i++) {
            assertEquals(a[2 * i], R.getQuick(size - 1 - i), tol);
        }
    }

    @Test
    public void testReshapeIntInt() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        int rows = 10;
        int cols = 17;
        DComplexMatrix2D B = A.reshape(rows, cols);
        int idx = 0;
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                AssertUtils.assertArrayEquals(A.getQuick(idx++), B.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        B = A.reshape(rows, cols);
        idx = 0;
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                AssertUtils.assertArrayEquals(A.getQuick(idx++), B.getQuick(r, c), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        B = Av.reshape(rows, cols);
        idx = 0;
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                AssertUtils.assertArrayEquals(Av.getQuick(idx++), B.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        B = Av.reshape(rows, cols);
        idx = 0;
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                AssertUtils.assertArrayEquals(Av.getQuick(idx++), B.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testReshapeIntIntInt() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        int slices = 2;
        int rows = 5;
        int cols = 17;
        DComplexMatrix3D B = A.reshape(slices, rows, cols);
        int idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int c = 0; c < cols; c++) {
                for (int r = 0; r < rows; r++) {
                    AssertUtils.assertArrayEquals(A.getQuick(idx++), B.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        B = A.reshape(slices, rows, cols);
        idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int c = 0; c < cols; c++) {
                for (int r = 0; r < rows; r++) {
                    AssertUtils.assertArrayEquals(A.getQuick(idx++), B.getQuick(s, r, c), tol);
                }
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        B = Av.reshape(slices, rows, cols);
        idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int c = 0; c < cols; c++) {
                for (int r = 0; r < rows; r++) {
                    AssertUtils.assertArrayEquals(Av.getQuick(idx++), B.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        B = Av.reshape(slices, rows, cols);
        idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int c = 0; c < cols; c++) {
                for (int r = 0; r < rows; r++) {
                    AssertUtils.assertArrayEquals(Av.getQuick(idx++), B.getQuick(s, r, c), tol);
                }
            }
        }
    }

    @Test
    public void testSetIntDoubleDouble() {
        DComplexMatrix1D A = new DenseDComplexMatrix1D(size);
        double[] elem = new double[] { Math.random(), Math.random() };
        A.set(size / 2, elem[0], elem[1]);
        double[] aElem = A.getQuick(size / 2);
        AssertUtils.assertArrayEquals(elem, aElem, tol);
    }

    @Test
    public void testSetQuickIntDoubleArray() {
        DComplexMatrix1D A = new DenseDComplexMatrix1D(size);
        double[] elem = new double[] { Math.random(), Math.random() };
        A.setQuick(size / 2, elem);
        double[] aElem = A.getQuick(size / 2);
        AssertUtils.assertArrayEquals(elem, aElem, tol);
    }

    @Test
    public void testSetQuickIntDoubleDouble() {
        DComplexMatrix1D A = new DenseDComplexMatrix1D(size);
        double[] elem = new double[] { Math.random(), Math.random() };
        A.setQuick(size / 2, elem[0], elem[1]);
        double[] aElem = A.getQuick(size / 2);
        AssertUtils.assertArrayEquals(elem, aElem, tol);
    }

    @Test
    public void testSwap() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D B = new DenseDComplexMatrix1D(b);
        A.swap(B);
        double[] aElts = (double[]) A.elements();
        double[] bElts = (double[]) B.elements();
        double[] tmp = new double[2];
        for (int i = 0; i < aElts.length; i = i + 2) {
            tmp[0] = b[i];
            tmp[1] = b[i + 1];
            assertEquals(tmp[0], aElts[i], tol);
            assertEquals(tmp[1], aElts[i + 1], tol);
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            assertEquals(tmp[0], bElts[i], tol);
            assertEquals(tmp[1], bElts[i + 1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        B = new DenseDComplexMatrix1D(b);
        A.swap(B);
        aElts = (double[]) A.elements();
        bElts = (double[]) B.elements();
        tmp = new double[2];
        for (int i = 0; i < aElts.length; i = i + 2) {
            tmp[0] = b[i];
            tmp[1] = b[i + 1];
            assertEquals(tmp[0], aElts[i], tol);
            assertEquals(tmp[1], aElts[i + 1], tol);
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            assertEquals(tmp[0], bElts[i], tol);
            assertEquals(tmp[1], bElts[i + 1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        B = new DenseDComplexMatrix1D(b);
        DComplexMatrix1D Bv = B.viewFlip();
        Av.swap(Bv);
        tmp = new double[2];
        for (int i = 0; i < aElts.length; i = i + 2) {
            tmp[0] = b[i];
            tmp[1] = b[i + 1];
            AssertUtils.assertArrayEquals(tmp, Av.getQuick(size - 1 - i / 2), tol);
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            AssertUtils.assertArrayEquals(tmp, Bv.getQuick(size - 1 - i / 2), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseDComplexMatrix1D(b);
        Bv = B.viewFlip();
        Av.swap(Bv);
        tmp = new double[2];
        for (int i = 0; i < aElts.length; i = i + 2) {
            tmp[0] = b[i];
            tmp[1] = b[i + 1];
            AssertUtils.assertArrayEquals(tmp, Av.getQuick(size - 1 - i / 2), tol);
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            AssertUtils.assertArrayEquals(tmp, Bv.getQuick(size - 1 - i / 2), tol);
        }
    }

    @Test
    public void testToArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        double[] array = A.toArray();
        AssertUtils.assertArrayEquals(a, array, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        array = A.toArray();
        AssertUtils.assertArrayEquals(a, array, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        array = Av.toArray();
        for (int i = 0; i < array.length / 2; i++) {
            assertEquals(a[2 * i], array[array.length - 2 - 2 * i], tol);
            assertEquals(a[2 * i + 1], array[array.length - 1 - 2 * i], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        array = Av.toArray();
        for (int i = 0; i < array.length / 2; i++) {
            assertEquals(a[2 * i], array[array.length - 2 - 2 * i], tol);
            assertEquals(a[2 * i + 1], array[array.length - 1 - 2 * i], tol);
        }
    }

    @Test
    public void testToArrayDoubleArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        double[] b = new double[2 * size];
        A.toArray(b);
        for (int i = 0; i < a.length; i = i + 2) {
            assertEquals(a[i], b[i], tol);
            assertEquals(a[i + 1], b[i + 1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        A.toArray(b);
        for (int i = 0; i < a.length; i = i + 2) {
            assertEquals(a[i], b[i], tol);
            assertEquals(a[i + 1], b[i + 1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        b = new double[2 * size];
        Av.toArray(b);
        for (int i = 0; i < a.length / 2; i++) {
            assertEquals(a[2 * i], b[a.length - 2 - 2 * i], tol);
            assertEquals(a[2 * i + 1], b[a.length - 1 - 2 * i], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        b = new double[2 * size];
        Av.toArray(b);
        for (int i = 0; i < a.length / 2; i++) {
            assertEquals(a[2 * i], b[a.length - 2 - 2 * i], tol);
            assertEquals(a[2 * i + 1], b[a.length - 1 - 2 * i], tol);
        }

    }

    @Test
    public void testToString() {
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        String s = A.toString();
        System.out.println(s);
    }

    @Test
    public void testViewFlip() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D B = A.viewFlip();
        for (int i = 0; i < size; i++) {
            AssertUtils.assertArrayEquals(A.getQuick(size - 1 - i), B.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        B = A.viewFlip();
        for (int i = 0; i < size; i++) {
            AssertUtils.assertArrayEquals(A.getQuick(size - 1 - i), B.getQuick(i), tol);
        }
    }

    @Test
    public void testViewPart() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D B = A.viewPart(15, 11);
        for (int i = 0; i < 11; i++) {
            AssertUtils.assertArrayEquals(new double[] { a[30 + 2 * i], a[30 + 1 + 2 * i] }, B.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        B = A.viewPart(15, 11);
        for (int i = 0; i < 11; i++) {
            AssertUtils.assertArrayEquals(new double[] { a[30 + 2 * i], a[30 + 1 + 2 * i] }, B.getQuick(i), tol);
        }
    }

    @Test
    public void testViewSelectionComplexProcedure() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D B = A.viewSelection(new DComplexProcedure() {
            public boolean apply(double[] element) {
                if (element[0] < element[1]) {
                    return true;
                } else {
                    return false;
                }
            }
        });
        for (int i = 0; i < B.size(); i++) {
            double[] el = B.getQuick(i);
            if (el[0] >= el[1]) {
                fail();
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        B = A.viewSelection(new DComplexProcedure() {
            public boolean apply(double[] element) {
                if (element[0] < element[1]) {
                    return true;
                } else {
                    return false;
                }
            }
        });
        for (int i = 0; i < B.size(); i++) {
            double[] el = B.getQuick(i);
            if (el[0] >= el[1]) {
                fail();
            }
        }
    }

    @Test
    public void testViewSelectionIntArray() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        int[] indexes = new int[] { 5, 11, 22, 37, 101 };
        DComplexMatrix1D B = A.viewSelection(indexes);
        for (int i = 0; i < indexes.length; i++) {
            AssertUtils.assertArrayEquals(A.getQuick(indexes[i]), B.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        indexes = new int[] { 5, 11, 22, 37, 101 };
        B = A.viewSelection(indexes);
        for (int i = 0; i < indexes.length; i++) {
            AssertUtils.assertArrayEquals(A.getQuick(indexes[i]), B.getQuick(i), tol);
        }
    }

    @Test
    public void testViewStrides() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        int stride = 3;
        DComplexMatrix1D B = A.viewStrides(stride);
        for (int i = 0; i < B.size(); i++) {
            AssertUtils.assertArrayEquals(A.getQuick(i * stride), B.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        B = A.viewStrides(stride);
        for (int i = 0; i < B.size(); i++) {
            AssertUtils.assertArrayEquals(A.getQuick(i * stride), B.getQuick(i), tol);
        }
    }

    @Test
    public void testZDotProductComplexMatrix1D() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D B = new DenseDComplexMatrix1D(b);
        double[] product = A.zDotProduct(B);
        double[] tmpSumMult = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSumMult = DComplex.plus(tmpSumMult, DComplex.mult(new double[] { a[i], a[i + 1] }, new double[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        B = new DenseDComplexMatrix1D(b);
        product = A.zDotProduct(B);
        tmpSumMult = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSumMult = DComplex.plus(tmpSumMult, DComplex.mult(new double[] { a[i], a[i + 1] }, new double[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        B = new DenseDComplexMatrix1D(b);
        DComplexMatrix1D Bv = B.viewFlip();
        product = Av.zDotProduct(Bv);
        tmpSumMult = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSumMult = DComplex.plus(tmpSumMult, DComplex.mult(new double[] { a[i], a[i + 1] }, new double[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseDComplexMatrix1D(b);
        Bv = B.viewFlip();
        product = Av.zDotProduct(Bv);
        tmpSumMult = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSumMult = DComplex.plus(tmpSumMult, DComplex.mult(new double[] { a[i], a[i + 1] }, new double[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
    }

    @Test
    public void testZDotProductComplexMatrix1DIntInt() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D B = new DenseDComplexMatrix1D(b);
        double[] product = A.zDotProduct(B, 5, B.size() - 10);
        double[] tmpSumMult = new double[2];
        for (int i = 10; i < a.length - 10; i = i + 2) {
            tmpSumMult = DComplex.plus(tmpSumMult, DComplex.mult(new double[] { a[i], a[i + 1] }, new double[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        B = new DenseDComplexMatrix1D(b);
        product = A.zDotProduct(B, 5, B.size() - 10);
        tmpSumMult = new double[2];
        for (int i = 10; i < a.length - 10; i = i + 2) {
            tmpSumMult = DComplex.plus(tmpSumMult, DComplex.mult(new double[] { a[i], a[i + 1] }, new double[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        B = new DenseDComplexMatrix1D(b);
        DComplexMatrix1D Bv = B.viewFlip();
        product = Av.zDotProduct(Bv, 5, Bv.size() - 10);
        tmpSumMult = new double[2];
        for (int i = 10; i < a.length - 10; i = i + 2) {
            tmpSumMult = DComplex.plus(tmpSumMult, DComplex.mult(new double[] { a[i], a[i + 1] }, new double[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseDComplexMatrix1D(b);
        Bv = B.viewFlip();
        product = Av.zDotProduct(Bv, 5, Bv.size() - 10);
        tmpSumMult = new double[2];
        for (int i = 10; i < a.length - 10; i = i + 2) {
            tmpSumMult = DComplex.plus(tmpSumMult, DComplex.mult(new double[] { a[i], a[i + 1] }, new double[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
    }

    @Test
    public void testZDotProductComplexMatrix1DIntIntIntArrayList() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D B = new DenseDComplexMatrix1D(b);
        IntArrayList indexList = new IntArrayList();
        ArrayList<double[]> valueList = new ArrayList<double[]>();
        B.getNonZeros(indexList, valueList);
        double[] product = A.zDotProduct(B, 5, B.size() - 10, indexList);
        double[] tmpSumMult = new double[2];
        for (int i = 10; i < a.length - 10; i = i + 2) {
            tmpSumMult = DComplex.plus(tmpSumMult, DComplex.mult(new double[] { a[i], a[i + 1] }, new double[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        B = new DenseDComplexMatrix1D(b);
        indexList = new IntArrayList();
        valueList = new ArrayList<double[]>();
        B.getNonZeros(indexList, valueList);
        product = A.zDotProduct(B, 5, B.size() - 10, indexList);
        tmpSumMult = new double[2];
        for (int i = 10; i < a.length - 10; i = i + 2) {
            tmpSumMult = DComplex.plus(tmpSumMult, DComplex.mult(new double[] { a[i], a[i + 1] }, new double[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        B = new DenseDComplexMatrix1D(b);
        DComplexMatrix1D Bv = B.viewFlip();
        indexList = new IntArrayList();
        valueList = new ArrayList<double[]>();
        Bv.getNonZeros(indexList, valueList);
        product = Av.zDotProduct(Bv, 5, Bv.size() - 10, indexList);
        tmpSumMult = new double[2];
        for (int i = 10; i < a.length - 10; i = i + 2) {
            tmpSumMult = DComplex.plus(tmpSumMult, DComplex.mult(new double[] { a[i], a[i + 1] }, new double[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseDComplexMatrix1D(b);
        Bv = B.viewFlip();
        indexList = new IntArrayList();
        valueList = new ArrayList<double[]>();
        Bv.getNonZeros(indexList, valueList);
        product = Av.zDotProduct(Bv, 5, Bv.size() - 10, indexList);
        tmpSumMult = new double[2];
        for (int i = 10; i < a.length - 10; i = i + 2) {
            tmpSumMult = DComplex.plus(tmpSumMult, DComplex.mult(new double[] { a[i], a[i + 1] }, new double[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
    }

    @Test
    public void testZSum() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        DComplexMatrix1D A = new DenseDComplexMatrix1D(a);
        double[] aSum = A.zSum();
        double[] tmpSum = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSum[0] += a[i];
            tmpSum[1] += a[i + 1];
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        aSum = A.zSum();
        tmpSum = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSum[0] += a[i];
            tmpSum[1] += a[i + 1];
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseDComplexMatrix1D(a);
        DComplexMatrix1D Av = A.viewFlip();
        aSum = Av.zSum();
        tmpSum = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSum[0] += a[i];
            tmpSum[1] += a[i + 1];
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseDComplexMatrix1D(a);
        Av = A.viewFlip();
        aSum = Av.zSum();
        tmpSum = new double[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSum[0] += a[i];
            tmpSum[1] += a[i + 1];
        }
    }

}
