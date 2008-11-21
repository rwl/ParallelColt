package cern.colt.matrix.tfcomplex.impl;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Random;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import cern.colt.function.tfcomplex.FComplexProcedure;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfcomplex.FComplexMatrix3D;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix1D;
import cern.colt.matrix.tfloat.FloatFactory1D;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.jet.math.tfcomplex.FComplex;
import cern.jet.math.tfcomplex.FComplexFunctions;
import edu.emory.mathcs.utils.AssertUtils;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public class TestDenseFComplexMatrix1D {
    private static final int size = 2 * 17 * 5;

    private static final float tol = 1e-4f;

    private static final int nThreads = 3;

    private static final int nThreadsBegin = 1;

    private float[] a, b;

    private static final FloatFactory1D factory = FloatFactory1D.dense;

    @Before
    public void setUpBeforeClass() throws Exception {
        // generate test matrices
        Random r = new Random(0);

        a = new float[2 * size];
        for (int i = 0; i < a.length; i++) {
            a[i] = r.nextFloat();
        }

        b = new float[2 * size];
        for (int i = 0; i < b.length; i++) {
            b[i] = r.nextFloat();
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
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        float[] aSum = A.aggregate(FComplexFunctions.plus, FComplexFunctions.square);
        float[] tmpSum = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSum = FComplex.plus(tmpSum, FComplex.square(new float[] { a[i], a[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        aSum = A.aggregate(FComplexFunctions.plus, FComplexFunctions.square);
        tmpSum = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSum = FComplex.plus(tmpSum, FComplex.square(new float[] { a[i], a[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        aSum = Av.aggregate(FComplexFunctions.plus, FComplexFunctions.square);
        tmpSum = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSum = FComplex.plus(tmpSum, FComplex.square(new float[] { a[i], a[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        Av = A.viewFlip();
        aSum = Av.aggregate(FComplexFunctions.plus, FComplexFunctions.square);
        tmpSum = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSum = FComplex.plus(tmpSum, FComplex.square(new float[] { a[i], a[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);

    }

    @Test
    public void testAggregateComplexMatrix1FComplexComplexFunctionComplexComplexFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D B = new DenseFComplexMatrix1D(b);
        float[] sumMult = A.aggregate(B, FComplexFunctions.plus, FComplexFunctions.mult);
        float[] tmpSumMult = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a[i], a[i + 1] }, new float[] { b[i], b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, sumMult, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        B = new DenseFComplexMatrix1D(b);
        sumMult = A.aggregate(B, FComplexFunctions.plus, FComplexFunctions.mult);
        tmpSumMult = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a[i], a[i + 1] }, new float[] { b[i], b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, sumMult, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        B = new DenseFComplexMatrix1D(b);
        FComplexMatrix1D Bv = B.viewFlip();
        sumMult = Av.aggregate(Bv, FComplexFunctions.plus, FComplexFunctions.mult);
        tmpSumMult = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a[i], a[i + 1] }, new float[] { b[i], b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, sumMult, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseFComplexMatrix1D(b);
        Bv = B.viewFlip();
        sumMult = Av.aggregate(Bv, FComplexFunctions.plus, FComplexFunctions.mult);
        tmpSumMult = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a[i], a[i + 1] }, new float[] { b[i], b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, sumMult, tol);
    }

    @Test
    public void testAssignComplexComplexFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        A.assign(FComplexFunctions.acos);
        float[] aElts = (float[]) A.elements();
        float[] tmp = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            tmp = FComplex.acos(tmp);
            assertEquals(tmp[0], aElts[i], tol);
            assertEquals(tmp[1], aElts[i + 1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        A.assign(FComplexFunctions.acos);
        aElts = (float[]) A.elements();
        tmp = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            tmp = FComplex.acos(tmp);
            assertEquals(tmp[0], aElts[i], tol);
            assertEquals(tmp[1], aElts[i + 1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        Av.assign(FComplexFunctions.acos);
        tmp = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            tmp = FComplex.acos(tmp);
            AssertUtils.assertArrayEquals(tmp, Av.getQuick(size - 1 - (i / 2)), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        Av = A.viewFlip();
        Av.assign(FComplexFunctions.acos);
        tmp = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            tmp = FComplex.acos(tmp);
            AssertUtils.assertArrayEquals(tmp, Av.getQuick(size - 1 - (i / 2)), tol);
        }
    }

    @Test
    public void testAssignComplexMatrix1D() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(size);
        FComplexMatrix1D B = new DenseFComplexMatrix1D(a);
        A.assign(B);
        float[] aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a, aElts, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(size);
        B = new DenseFComplexMatrix1D(a);
        A.assign(B);
        aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a, aElts, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(size);
        FComplexMatrix1D Av = A.viewFlip();
        B = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Bv = B.viewFlip();
        Av.assign(Bv);
        float[] tmp = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            AssertUtils.assertArrayEquals(tmp, Av.getQuick(size - 1 - (i / 2)), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(size);
        Av = A.viewFlip();
        B = new DenseFComplexMatrix1D(a);
        Bv = B.viewFlip();
        Av.assign(Bv);
        tmp = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            AssertUtils.assertArrayEquals(tmp, Av.getQuick(size - 1 - (i / 2)), tol);
        }
    }

    @Test
    public void testAssignComplexMatrix1FComplexComplexFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D B = new DenseFComplexMatrix1D(b);
        A.assign(B, FComplexFunctions.div);
        float[] aElts = (float[]) A.elements();
        float[] tmp1 = new float[2];
        float[] tmp2 = new float[2];
        for (int i = 0; i < aElts.length; i = i + 2) {
            tmp1[0] = a[i];
            tmp1[1] = a[i + 1];
            tmp2[0] = b[i];
            tmp2[1] = b[i + 1];
            tmp1 = FComplex.div(tmp1, tmp2);
            assertEquals(tmp1[0], aElts[i], tol);
            assertEquals(tmp1[1], aElts[i + 1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        B = new DenseFComplexMatrix1D(b);
        A.assign(B, FComplexFunctions.div);
        aElts = (float[]) A.elements();
        tmp1 = new float[2];
        tmp2 = new float[2];
        for (int i = 0; i < aElts.length; i = i + 2) {
            tmp1[0] = a[i];
            tmp1[1] = a[i + 1];
            tmp2[0] = b[i];
            tmp2[1] = b[i + 1];
            tmp1 = FComplex.div(tmp1, tmp2);
            assertEquals(tmp1[0], aElts[i], tol);
            assertEquals(tmp1[1], aElts[i + 1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        B = new DenseFComplexMatrix1D(b);
        FComplexMatrix1D Bv = B.viewFlip();
        Av.assign(Bv, FComplexFunctions.div);
        tmp1 = new float[2];
        tmp2 = new float[2];
        for (int i = 0; i < aElts.length; i = i + 2) {
            tmp1[0] = a[i];
            tmp1[1] = a[i + 1];
            tmp2[0] = b[i];
            tmp2[1] = b[i + 1];
            tmp1 = FComplex.div(tmp1, tmp2);
            AssertUtils.assertArrayEquals(tmp1, Av.getQuick(size - 1 - (i / 2)), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseFComplexMatrix1D(b);
        Bv = B.viewFlip();
        Av.assign(Bv, FComplexFunctions.div);
        tmp1 = new float[2];
        tmp2 = new float[2];
        for (int i = 0; i < aElts.length; i = i + 2) {
            tmp1[0] = a[i];
            tmp1[1] = a[i + 1];
            tmp2[0] = b[i];
            tmp2[1] = b[i + 1];
            tmp1 = FComplex.div(tmp1, tmp2);
            AssertUtils.assertArrayEquals(tmp1, Av.getQuick(size - 1 - (i / 2)), tol);
        }
    }

    @Test
    public void testAssignComplexProcedureComplexComplexFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D B = A.copy();
        A.assign(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, FComplexFunctions.tan);
        for (int i = 0; i < size; i++) {
            if (FComplex.abs(B.getQuick(i)) > 3) {
                B.setQuick(i, FComplex.tan(B.getQuick(i)));
            }
        }
        AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
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
        for (int i = 0; i < size; i++) {
            if (FComplex.abs(B.getQuick(i)) > 3) {
                B.setQuick(i, FComplex.tan(B.getQuick(i)));
            }
        }
        AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        B = A.copy();
        FComplexMatrix1D Bv = B.viewFlip();
        Av.assign(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, FComplexFunctions.tan);
        for (int i = 0; i < size; i++) {
            if (FComplex.abs(Bv.getQuick(i)) > 3) {
                Bv.setQuick(i, FComplex.tan(Bv.getQuick(i)));
            }
        }
        for (int i = 0; i < B.size(); i++) {
            AssertUtils.assertArrayEquals(Bv.getQuick(i), Av.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        Av = A.viewFlip();
        B = A.copy();
        Bv = B.viewFlip();
        Av.assign(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, FComplexFunctions.tan);
        for (int i = 0; i < size; i++) {
            if (FComplex.abs(Bv.getQuick(i)) > 3) {
                Bv.setQuick(i, FComplex.tan(Bv.getQuick(i)));
            }
        }
        for (int i = 0; i < B.size(); i++) {
            AssertUtils.assertArrayEquals(Bv.getQuick(i), Av.getQuick(i), tol);
        }
    }

    @Test
    public void testAssignComplexProcedureFloatArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D B = A.copy();
        A.assign(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, new float[] { -1, -1 });
        for (int i = 0; i < size; i++) {
            if (FComplex.abs(B.getQuick(i)) > 3) {
                B.setQuick(i, new float[] { -1, -1 });
            }
        }
        AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
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
        for (int i = 0; i < size; i++) {
            if (FComplex.abs(B.getQuick(i)) > 3) {
                B.setQuick(i, new float[] { -1, -1 });
            }
        }
        AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        B = A.copy();
        FComplexMatrix1D Bv = B.viewFlip();
        Av.assign(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, new float[] { -1, -1 });
        for (int i = 0; i < size; i++) {
            if (FComplex.abs(Bv.getQuick(i)) > 3) {
                Bv.setQuick(i, new float[] { -1, -1 });
            }
        }
        for (int i = 0; i < B.size(); i++) {
            AssertUtils.assertArrayEquals(Bv.getQuick(i), Av.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        Av = A.viewFlip();
        B = A.copy();
        Bv = B.viewFlip();
        Av.assign(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, new float[] { -1, -1 });
        for (int i = 0; i < size; i++) {
            if (FComplex.abs(Bv.getQuick(i)) > 3) {
                Bv.setQuick(i, new float[] { -1, -1 });
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
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        A.assign(FComplexFunctions.abs);
        float[] aElts = (float[]) A.elements();
        float[] tmp1 = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp1[0] = a[i];
            tmp1[1] = a[i + 1];
            tmp1[0] = FComplex.abs(tmp1);
            assertEquals(tmp1[0], aElts[i], tol);
            assertEquals(0.0, aElts[i + 1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        A.assign(FComplexFunctions.abs);
        aElts = (float[]) A.elements();
        tmp1 = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp1[0] = a[i];
            tmp1[1] = a[i + 1];
            tmp1[0] = FComplex.abs(tmp1);
            assertEquals(tmp1[0], aElts[i], tol);
            assertEquals(0.0, aElts[i + 1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        Av.assign(FComplexFunctions.abs);
        tmp1 = new float[2];
        float[] tmp2 = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp1[0] = a[i];
            tmp1[1] = a[i + 1];
            tmp1[0] = FComplex.abs(tmp1);
            tmp2 = Av.getQuick(size - 1 - i / 2);
            assertEquals(tmp1[0], tmp2[0], tol);
            assertEquals(0.0, tmp2[1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        Av = A.viewFlip();
        Av.assign(FComplexFunctions.abs);
        tmp1 = new float[2];
        tmp2 = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp1[0] = a[i];
            tmp1[1] = a[i + 1];
            tmp1[0] = FComplex.abs(tmp1);
            tmp2 = Av.getQuick(size - 1 - i / 2);
            assertEquals(tmp1[0], tmp2[0], tol);
            assertEquals(0.0, tmp2[1], tol);
        }
    }

    @Test
    public void testAssignFloatArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(size);
        A.assign(a);
        float[] aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a, aElts, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(size);
        A.assign(a);
        aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a, aElts, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(size);
        FComplexMatrix1D Av = A.viewFlip();
        Av.assign(a);
        float[] tmp = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            AssertUtils.assertArrayEquals(tmp, Av.getQuick(i / 2), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(size);
        Av = A.viewFlip();
        Av.assign(a);
        tmp = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp[0] = a[i];
            tmp[1] = a[i + 1];
            AssertUtils.assertArrayEquals(tmp, Av.getQuick(i / 2), tol);
        }

    }

    @Test
    public void testAssignFloatFloat() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(size);
        float[] value = new float[] { (float)Math.random(), (float)Math.random() };
        A.assign(value[0], value[1]);
        float[] aElts = (float[]) A.elements();
        for (int i = 0; i < aElts.length; i = i + 2) {
            assertEquals(value[0], aElts[i], tol);
            assertEquals(value[1], aElts[i + 1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(size);
        A.assign(value[0], value[1]);
        aElts = (float[]) A.elements();
        for (int i = 0; i < aElts.length; i = i + 2) {
            assertEquals(value[0], aElts[i], tol);
            assertEquals(value[1], aElts[i + 1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(size);
        FComplexMatrix1D Av = A.viewFlip();
        value = new float[] { (float)Math.random(), (float)Math.random() };
        Av.assign(value[0], value[1]);
        float[] tmp = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmp = Av.getQuick(i / 2);
            AssertUtils.assertArrayEquals(value, tmp, tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(size);
        Av = A.viewFlip();
        value = new float[] { (float)Math.random(), (float)Math.random() };
        Av.assign(value[0], value[1]);
        tmp = new float[2];
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
        FComplexMatrix1D A = new DenseFComplexMatrix1D(size);
        FloatMatrix1D Im = factory.random(size);
        A.assignImaginary(Im);
        int idx = 0;
        float[] aElts = (float[]) A.elements();
        for (int i = 0; i < aElts.length; i = i + 2) {
            assertEquals(0, aElts[i], tol);
            assertEquals(Im.getQuick(idx++), aElts[i + 1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(size);
        Im = factory.random(size);
        A.assignImaginary(Im);
        aElts = (float[]) A.elements();
        idx = 0;
        for (int i = 0; i < aElts.length; i = i + 2) {
            assertEquals(0, aElts[i], tol);
            assertEquals(Im.getQuick(idx++), aElts[i + 1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(size);
        FComplexMatrix1D Av = A.viewFlip();
        Im = factory.random(size);
        Av.assignImaginary(Im);
        idx = 0;
        float[] tmp = new float[2];
        for (int i = 0; i < Av.size(); i++) {
            tmp = Av.getQuick(i);
            assertEquals(0, tmp[0], tol);
            assertEquals(Im.getQuick(idx++), tmp[1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(size);
        Av = A.viewFlip();
        Im = factory.random(size);
        Av.assignImaginary(Im);
        idx = 0;
        tmp = new float[2];
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
        FComplexMatrix1D A = new DenseFComplexMatrix1D(size);
        FloatMatrix1D Re = factory.random(size);
        A.assignReal(Re);
        int idx = 0;
        float[] aElts = (float[]) A.elements();
        for (int i = 0; i < aElts.length; i = i + 2) {
            assertEquals(Re.getQuick(idx++), aElts[i], tol);
            assertEquals(0, aElts[i + 1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(size);
        Re = factory.random(size);
        A.assignReal(Re);
        idx = 0;
        aElts = (float[]) A.elements();
        for (int i = 0; i < aElts.length; i = i + 2) {
            assertEquals(Re.getQuick(idx++), aElts[i], tol);
            assertEquals(0, aElts[i + 1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(size);
        FComplexMatrix1D Av = A.viewFlip();
        Re = factory.random(size);
        Av.assignReal(Re);
        idx = 0;
        float[] tmp = new float[2];
        for (int i = 0; i < Av.size(); i++) {
            tmp = Av.getQuick(i);
            assertEquals(Re.getQuick(idx++), tmp[0], tol);
            assertEquals(0, tmp[1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(size);
        Av = A.viewFlip();
        Re = factory.random(size);
        Av.assignReal(Re);
        idx = 0;
        tmp = new float[2];
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
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        int card = A.cardinality();
        assertEquals(a.length / 2, card);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        card = A.cardinality();
        assertEquals(a.length / 2, card);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        card = Av.cardinality();
        assertEquals(a.length / 2, card);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        Av = A.viewFlip();
        card = Av.cardinality();
        assertEquals(a.length / 2, card);
    }

    @Test
    public void testCopy() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D B = A.copy();
        float[] bElts = (float[]) B.elements();
        AssertUtils.assertArrayEquals(a, bElts, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        B = A.copy();
        bElts = (float[]) B.elements();
        AssertUtils.assertArrayEquals(a, bElts, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        B = Av.copy();
        for (int i = 0; i < Av.size(); i++) {
            AssertUtils.assertArrayEquals(Av.get(i), B.get(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        Av = A.viewFlip();
        B = Av.copy();
        for (int i = 0; i < Av.size(); i++) {
            AssertUtils.assertArrayEquals(Av.get(i), B.get(i), tol);
        }
    }

    @Test
    public void testEqualsFloatArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(size);
        float[] value = new float[] { 1, 1 };
        A.assign(1, 1);
        boolean eq = A.equals(value);
        assertEquals(true, eq);
        eq = A.equals(new float[] { 2, 1 });
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(size);
        value = new float[] { 1, 1 };
        A.assign(1, 1);
        eq = A.equals(value);
        assertEquals(true, eq);
        eq = A.equals(new float[] { 2, 1 });
        assertEquals(false, eq);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(size);
        FComplexMatrix1D Av = A.viewFlip();
        value = new float[] { 1, 1 };
        Av.assign(1, 1);
        eq = A.equals(value);
        assertEquals(true, eq);
        eq = Av.equals(new float[] { 2, 1 });
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(size);
        Av = A.viewFlip();
        value = new float[] { 1, 1 };
        Av.assign(1, 1);
        eq = A.equals(value);
        assertEquals(true, eq);
        eq = Av.equals(new float[] { 2, 1 });
        assertEquals(false, eq);
    }

    @Test
    public void testEqualsObject() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D B = new DenseFComplexMatrix1D(b);
        boolean eq = A.equals(A);
        assertEquals(true, eq);
        eq = A.equals(B);
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        B = new DenseFComplexMatrix1D(b);
        eq = A.equals(A);
        assertEquals(true, eq);
        eq = A.equals(B);
        assertEquals(false, eq);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        B = new DenseFComplexMatrix1D(b);
        FComplexMatrix1D Bv = B.viewFlip();
        eq = Av.equals(Av);
        assertEquals(true, eq);
        eq = Av.equals(Bv);
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseFComplexMatrix1D(b);
        Bv = B.viewFlip();
        eq = Av.equals(Av);
        assertEquals(true, eq);
        eq = Av.equals(Bv);
        assertEquals(false, eq);
    }
    
    @Test
    public void testFft() {
        int size = 2048;
        float[] a_1d = new float[size];
        for (int i = 0; i < size; i++) {
            a_1d[i] = (float)Math.random();
        }
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(size / 2);
        A.assign(a_1d);
        A.fft();
        A.ifft(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(nThreadsBegin);
        A = new DenseFComplexMatrix1D(size / 2);
        A.assign(a_1d);
        A.fft();
        A.ifft(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
        A = new DenseFComplexMatrix1D(size / 2);
        A.assign(a_1d);
        FComplexMatrix1D Av = A.viewPart(size / 4 - 1, size / 4);
        float[] av_elems = (float[]) Av.copy().elements();
        Av.fft();
        Av.ifft(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(nThreadsBegin);
        A = new DenseFComplexMatrix1D(size / 2);
        A.assign(a_1d);
        Av = A.viewPart(size / 4 - 1, size / 4);
        av_elems = (float[]) Av.copy().elements();
        Av.fft();
        Av.ifft(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
    }

    @Test
    public void testGet() {
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        float[] elem;
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
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        FloatMatrix1D Im = A.getImaginaryPart();
        for (int i = 0; i < size; i++) {
            assertEquals(a[2 * i + 1], Im.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        Im = A.getImaginaryPart();
        for (int i = 0; i < size; i++) {
            assertEquals(a[2 * i + 1], Im.getQuick(i), tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        Im = Av.getImaginaryPart();
        for (int i = 0; i < size; i++) {
            assertEquals(a[2 * i + 1], Im.getQuick(size - 1 - i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        Av = A.viewFlip();
        Im = Av.getImaginaryPart();
        for (int i = 0; i < size; i++) {
            assertEquals(a[2 * i + 1], Im.getQuick(size - 1 - i), tol);
        }
    }

    @Test
    public void testGetNonZerosIntArrayListArrayListOffloat() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        IntArrayList indexList = new IntArrayList();
        ArrayList<float[]> valueList = new ArrayList<float[]>();
        A.getNonZeros(indexList, valueList);
        assertEquals(size, indexList.size());
        assertEquals(size, valueList.size());
        for (int i = 0; i < size; i++) {
            AssertUtils.assertArrayEquals(A.getQuick(indexList.get(i)), valueList.get(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        A.getNonZeros(indexList, valueList);
        assertEquals(size, indexList.size());
        assertEquals(size, valueList.size());
        for (int i = 0; i < size; i++) {
            AssertUtils.assertArrayEquals(A.getQuick(indexList.get(i)), valueList.get(i), tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        Av.getNonZeros(indexList, valueList);
        assertEquals(size, indexList.size());
        assertEquals(size, valueList.size());
        for (int i = 0; i < size; i++) {
            AssertUtils.assertArrayEquals(Av.getQuick(indexList.get(i)), valueList.get(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
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
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        float[] elem;
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
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        FloatMatrix1D R = A.getRealPart();
        for (int i = 0; i < size; i++) {
            assertEquals(a[2 * i], R.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        R = A.getRealPart();
        for (int i = 0; i < size; i++) {
            assertEquals(a[2 * i], R.getQuick(i), tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        R = Av.getRealPart();
        for (int i = 0; i < size; i++) {
            assertEquals(a[2 * i], R.getQuick(size - 1 - i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
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
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        int rows = 10;
        int cols = 17;
        FComplexMatrix2D B = A.reshape(rows, cols);
        int idx = 0;
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                AssertUtils.assertArrayEquals(A.getQuick(idx++), B.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
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
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
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
        A = new DenseFComplexMatrix1D(a);
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
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        int slices = 2;
        int rows = 5;
        int cols = 17;
        FComplexMatrix3D B = A.reshape(slices, rows, cols);
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
        A = new DenseFComplexMatrix1D(a);
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
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
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
        A = new DenseFComplexMatrix1D(a);
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
    public void testSetIntFloatFloat() {
        FComplexMatrix1D A = new DenseFComplexMatrix1D(size);
        float[] elem = new float[] { (float)Math.random(), (float)Math.random() };
        A.set(size / 2, elem[0], elem[1]);
        float[] aElem = A.getQuick(size / 2);
        AssertUtils.assertArrayEquals(elem, aElem, tol);
    }

    @Test
    public void testSetQuickIntFloatArray() {
        FComplexMatrix1D A = new DenseFComplexMatrix1D(size);
        float[] elem = new float[] { (float)Math.random(), (float)Math.random() };
        A.setQuick(size / 2, elem);
        float[] aElem = A.getQuick(size / 2);
        AssertUtils.assertArrayEquals(elem, aElem, tol);
    }

    @Test
    public void testSetQuickIntFloatFloat() {
        FComplexMatrix1D A = new DenseFComplexMatrix1D(size);
        float[] elem = new float[] { (float)Math.random(), (float)Math.random() };
        A.setQuick(size / 2, elem[0], elem[1]);
        float[] aElem = A.getQuick(size / 2);
        AssertUtils.assertArrayEquals(elem, aElem, tol);
    }

    @Test
    public void testSwap() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D B = new DenseFComplexMatrix1D(b);
        A.swap(B);
        float[] aElts = (float[]) A.elements();
        float[] bElts = (float[]) B.elements();
        float[] tmp = new float[2];
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
        A = new DenseFComplexMatrix1D(a);
        B = new DenseFComplexMatrix1D(b);
        A.swap(B);
        aElts = (float[]) A.elements();
        bElts = (float[]) B.elements();
        tmp = new float[2];
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
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        B = new DenseFComplexMatrix1D(b);
        FComplexMatrix1D Bv = B.viewFlip();
        Av.swap(Bv);
        tmp = new float[2];
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
        A = new DenseFComplexMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseFComplexMatrix1D(b);
        Bv = B.viewFlip();
        Av.swap(Bv);
        tmp = new float[2];
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
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        float[] array = A.toArray();
        AssertUtils.assertArrayEquals(a, array, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        array = A.toArray();
        AssertUtils.assertArrayEquals(a, array, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        array = Av.toArray();
        for (int i = 0; i < array.length / 2; i++) {
            assertEquals(a[2 * i], array[array.length - 2 - 2 * i], tol);
            assertEquals(a[2 * i + 1], array[array.length - 1 - 2 * i], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        Av = A.viewFlip();
        array = Av.toArray();
        for (int i = 0; i < array.length / 2; i++) {
            assertEquals(a[2 * i], array[array.length - 2 - 2 * i], tol);
            assertEquals(a[2 * i + 1], array[array.length - 1 - 2 * i], tol);
        }
    }

    @Test
    public void testToArrayFloatArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        float[] b = new float[2 * size];
        A.toArray(b);
        for (int i = 0; i < a.length; i = i + 2) {
            assertEquals(a[i], b[i], tol);
            assertEquals(a[i + 1], b[i + 1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        A.toArray(b);
        for (int i = 0; i < a.length; i = i + 2) {
            assertEquals(a[i], b[i], tol);
            assertEquals(a[i + 1], b[i + 1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        b = new float[2 * size];
        Av.toArray(b);
        for (int i = 0; i < a.length / 2; i++) {
            assertEquals(a[2 * i], b[a.length - 2 - 2 * i], tol);
            assertEquals(a[2 * i + 1], b[a.length - 1 - 2 * i], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        Av = A.viewFlip();
        b = new float[2 * size];
        Av.toArray(b);
        for (int i = 0; i < a.length / 2; i++) {
            assertEquals(a[2 * i], b[a.length - 2 - 2 * i], tol);
            assertEquals(a[2 * i + 1], b[a.length - 1 - 2 * i], tol);
        }

    }

    @Test
    public void testToString() {
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        String s = A.toString();
        System.out.println(s);
    }

    @Test
    public void testViewFlip() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D B = A.viewFlip();
        for (int i = 0; i < size; i++) {
            AssertUtils.assertArrayEquals(A.getQuick(size - 1 - i), B.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        B = A.viewFlip();
        for (int i = 0; i < size; i++) {
            AssertUtils.assertArrayEquals(A.getQuick(size - 1 - i), B.getQuick(i), tol);
        }
    }

    @Test
    public void testViewPart() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D B = A.viewPart(15, 11);
        for (int i = 0; i < 11; i++) {
            AssertUtils.assertArrayEquals(new float[] { a[30 + 2 * i], a[30 + 1 + 2 * i] }, B.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        B = A.viewPart(15, 11);
        for (int i = 0; i < 11; i++) {
            AssertUtils.assertArrayEquals(new float[] { a[30 + 2 * i], a[30 + 1 + 2 * i] }, B.getQuick(i), tol);
        }
    }

    @Test
    public void testViewSelectionComplexProcedure() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D B = A.viewSelection(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (element[0] < element[1]) {
                    return true;
                } else {
                    return false;
                }
            }
        });
        for (int i = 0; i < B.size(); i++) {
            float[] el = B.getQuick(i);
            if (el[0] >= el[1]) {
                fail();
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        B = A.viewSelection(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (element[0] < element[1]) {
                    return true;
                } else {
                    return false;
                }
            }
        });
        for (int i = 0; i < B.size(); i++) {
            float[] el = B.getQuick(i);
            if (el[0] >= el[1]) {
                fail();
            }
        }
    }

    @Test
    public void testViewSelectionIntArray() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        int[] indexes = new int[] { 5, 11, 22, 37, 101 };
        FComplexMatrix1D B = A.viewSelection(indexes);
        for (int i = 0; i < indexes.length; i++) {
            AssertUtils.assertArrayEquals(A.getQuick(indexes[i]), B.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
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
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        int stride = 3;
        FComplexMatrix1D B = A.viewStrides(stride);
        for (int i = 0; i < B.size(); i++) {
            AssertUtils.assertArrayEquals(A.getQuick(i * stride), B.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
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
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D B = new DenseFComplexMatrix1D(b);
        float[] product = A.zDotProduct(B);
        float[] tmpSumMult = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a[i], a[i + 1] }, new float[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        B = new DenseFComplexMatrix1D(b);
        product = A.zDotProduct(B);
        tmpSumMult = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a[i], a[i + 1] }, new float[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        B = new DenseFComplexMatrix1D(b);
        FComplexMatrix1D Bv = B.viewFlip();
        product = Av.zDotProduct(Bv);
        tmpSumMult = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a[i], a[i + 1] }, new float[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseFComplexMatrix1D(b);
        Bv = B.viewFlip();
        product = Av.zDotProduct(Bv);
        tmpSumMult = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a[i], a[i + 1] }, new float[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
    }

    @Test
    public void testZDotProductComplexMatrix1DIntInt() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D B = new DenseFComplexMatrix1D(b);
        float[] product = A.zDotProduct(B, 5, B.size() - 10);
        float[] tmpSumMult = new float[2];
        for (int i = 10; i < a.length - 10; i = i + 2) {
            tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a[i], a[i + 1] }, new float[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        B = new DenseFComplexMatrix1D(b);
        product = A.zDotProduct(B, 5, B.size() - 10);
        tmpSumMult = new float[2];
        for (int i = 10; i < a.length - 10; i = i + 2) {
            tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a[i], a[i + 1] }, new float[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        B = new DenseFComplexMatrix1D(b);
        FComplexMatrix1D Bv = B.viewFlip();
        product = Av.zDotProduct(Bv, 5, Bv.size() - 10);
        tmpSumMult = new float[2];
        for (int i = 10; i < a.length - 10; i = i + 2) {
            tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a[i], a[i + 1] }, new float[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseFComplexMatrix1D(b);
        Bv = B.viewFlip();
        product = Av.zDotProduct(Bv, 5, Bv.size() - 10);
        tmpSumMult = new float[2];
        for (int i = 10; i < a.length - 10; i = i + 2) {
            tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a[i], a[i + 1] }, new float[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
    }

    @Test
    public void testZDotProductComplexMatrix1DIntIntIntArrayList() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D B = new DenseFComplexMatrix1D(b);
        IntArrayList indexList = new IntArrayList();
        ArrayList<float[]> valueList = new ArrayList<float[]>();
        B.getNonZeros(indexList, valueList);
        float[] product = A.zDotProduct(B, 5, B.size() - 10, indexList);
        float[] tmpSumMult = new float[2];
        for (int i = 10; i < a.length - 10; i = i + 2) {
            tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a[i], a[i + 1] }, new float[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        B = new DenseFComplexMatrix1D(b);
        indexList = new IntArrayList();
        valueList = new ArrayList<float[]>();
        B.getNonZeros(indexList, valueList);
        product = A.zDotProduct(B, 5, B.size() - 10, indexList);
        tmpSumMult = new float[2];
        for (int i = 10; i < a.length - 10; i = i + 2) {
            tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a[i], a[i + 1] }, new float[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        B = new DenseFComplexMatrix1D(b);
        FComplexMatrix1D Bv = B.viewFlip();
        indexList = new IntArrayList();
        valueList = new ArrayList<float[]>();
        Bv.getNonZeros(indexList, valueList);
        product = Av.zDotProduct(Bv, 5, Bv.size() - 10, indexList);
        tmpSumMult = new float[2];
        for (int i = 10; i < a.length - 10; i = i + 2) {
            tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a[i], a[i + 1] }, new float[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseFComplexMatrix1D(b);
        Bv = B.viewFlip();
        indexList = new IntArrayList();
        valueList = new ArrayList<float[]>();
        Bv.getNonZeros(indexList, valueList);
        product = Av.zDotProduct(Bv, 5, Bv.size() - 10, indexList);
        tmpSumMult = new float[2];
        for (int i = 10; i < a.length - 10; i = i + 2) {
            tmpSumMult = FComplex.plus(tmpSumMult, FComplex.mult(new float[] { a[i], a[i + 1] }, new float[] { b[i], -b[i + 1] }));
        }
        AssertUtils.assertArrayEquals(tmpSumMult, product, tol);
    }

    @Test
    public void testZSum() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FComplexMatrix1D A = new DenseFComplexMatrix1D(a);
        float[] aSum = A.zSum();
        float[] tmpSum = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSum[0] += a[i];
            tmpSum[1] += a[i + 1];
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        aSum = A.zSum();
        tmpSum = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSum[0] += a[i];
            tmpSum[1] += a[i + 1];
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFComplexMatrix1D(a);
        FComplexMatrix1D Av = A.viewFlip();
        aSum = Av.zSum();
        tmpSum = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSum[0] += a[i];
            tmpSum[1] += a[i + 1];
        }
        AssertUtils.assertArrayEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFComplexMatrix1D(a);
        Av = A.viewFlip();
        aSum = Av.zSum();
        tmpSum = new float[2];
        for (int i = 0; i < a.length; i = i + 2) {
            tmpSum[0] += a[i];
            tmpSum[1] += a[i + 1];
        }
    }

}
