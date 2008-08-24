package cern.colt.matrix.impl;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.Arrays;
import java.util.Random;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import cern.colt.function.FloatProcedure;
import cern.colt.list.FloatArrayList;
import cern.colt.list.IntArrayList;
import cern.colt.matrix.FComplexMatrix1D;
import cern.colt.matrix.FloatMatrix1D;
import cern.colt.matrix.FloatMatrix2D;
import cern.colt.matrix.FloatMatrix3D;
import cern.jet.math.FloatFunctions;
import edu.emory.mathcs.utils.AssertUtils;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public class TestDenseFloatMatrix1D {
    private static final int size = 2 * 17 * 5;

    private static final float tol = 1e-4f;

    private static final int nThreads = 3;

    private static final int nThreadsBegin = 1;

    float[] a, b;

    @Before
    public void setUpBeforeClass() throws Exception {
        // generate test matrices
        Random r = new Random(0);

        a = new float[size];
        for (int i = 0; i < a.length; i++) {
            a[i] = r.nextFloat();
        }

        b = new float[size];
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
    public void testAggregateFloatFloatFunctionFloatFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        float aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.square);
        float tmpSum = 0;
        for (int i = 0; i < a.length; i++) {
            tmpSum += a[i] * a[i];
        }
        assertEquals(tmpSum, aSum, tol);
        // multithreaded
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        aSum = A.aggregate(FloatFunctions.plus, FloatFunctions.square);
        tmpSum = 0;
        for (int i = 0; i < a.length; i++) {
            tmpSum += a[i] * a[i];
        }
        assertEquals(tmpSum, aSum, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.square);
        tmpSum = 0;
        for (int i = 0; i < a.length; i++) {
            tmpSum += a[i] * a[i];
        }
        assertEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        Av = A.viewFlip();
        aSum = Av.aggregate(FloatFunctions.plus, FloatFunctions.square);
        tmpSum = 0;
        for (int i = 0; i < a.length; i++) {
            tmpSum += a[i] * a[i];
        }
        assertEquals(tmpSum, aSum, tol);
    }

    @Test
    public void testAggregateFloatMatrix1DFloatFloatFunctionFloatFloatFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatMatrix1D B = new DenseFloatMatrix1D(b);
        float sumMult = A.aggregate(B, FloatFunctions.plus, FloatFunctions.mult);
        float tmpSumMult = 0;
        for (int i = 0; i < a.length; i++) {
            tmpSumMult += a[i] * b[i];
        }
        assertEquals(tmpSumMult, sumMult, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        B = new DenseFloatMatrix1D(b);
        sumMult = A.aggregate(B, FloatFunctions.plus, FloatFunctions.mult);
        tmpSumMult = 0;
        for (int i = 0; i < a.length; i++) {
            tmpSumMult += a[i] * b[i];
        }
        assertEquals(tmpSumMult, sumMult, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        B = new DenseFloatMatrix1D(b);
        FloatMatrix1D Bv = B.viewFlip();
        sumMult = Av.aggregate(Bv, FloatFunctions.plus, FloatFunctions.mult);
        tmpSumMult = 0;
        for (int i = 0; i < a.length; i++) {
            tmpSumMult += a[i] * b[i];
        }
        assertEquals(tmpSumMult, sumMult, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseFloatMatrix1D(b);
        Bv = B.viewFlip();
        sumMult = Av.aggregate(Bv, FloatFunctions.plus, FloatFunctions.mult);
        tmpSumMult = 0;
        for (int i = 0; i < a.length; i++) {
            tmpSumMult += a[i] * b[i];
        }
        assertEquals(tmpSumMult, sumMult, tol);

    }

    @Test
    public void testAssignFloatProcedureFloatFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatMatrix1D B = A.copy();
        A.assign(new FloatProcedure() {
            public boolean apply(float element) {
                if (Math.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        }, FloatFunctions.tan);
        for (int i = 0; i < size; i++) {
            if (Math.abs(B.getQuick(i)) > 0.1) {
                B.setQuick(i, (float)Math.tan(B.getQuick(i)));
            }
        }
        AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        B = A.copy();
        A.assign(new FloatProcedure() {
            public boolean apply(float element) {
                if ((float)Math.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        }, FloatFunctions.tan);
        for (int i = 0; i < size; i++) {
            if ((float)Math.abs(B.getQuick(i)) > 0.1) {
                B.setQuick(i, (float)Math.tan(B.getQuick(i)));
            }
        }
        AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        B = A.copy();
        Av.assign(new FloatProcedure() {
            public boolean apply(float element) {
                if ((float)Math.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        }, FloatFunctions.tan);
        for (int i = 0; i < size; i++) {
            if ((float)Math.abs(B.getQuick(i)) > 0.1) {
                B.setQuick(i, (float)Math.tan(B.getQuick(i)));
            }
        }
        for (int i = 0; i < size; i++) {
            assertEquals(B.getQuick(i), Av.getQuick(size - 1 - i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        Av = A.viewFlip();
        B = A.copy();
        Av.assign(new FloatProcedure() {
            public boolean apply(float element) {
                if ((float)Math.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        }, FloatFunctions.tan);
        for (int i = 0; i < size; i++) {
            if ((float)Math.abs(B.getQuick(i)) > 0.1) {
                B.setQuick(i, (float)Math.tan(B.getQuick(i)));
            }
        }
        for (int i = 0; i < size; i++) {
            assertEquals(B.getQuick(i), Av.getQuick(size - 1 - i), tol);
        }
    }

    @Test
    public void testAssignFloatProcedureFloat() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatMatrix1D B = A.copy();
        A.assign(new FloatProcedure() {
            public boolean apply(float element) {
                if ((float)Math.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        }, -1);
        for (int i = 0; i < size; i++) {
            if ((float)Math.abs(B.getQuick(i)) > 0.1) {
                B.setQuick(i, -1);
            }
        }
        AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        B = A.copy();
        A.assign(new FloatProcedure() {
            public boolean apply(float element) {
                if ((float)Math.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        }, -1);
        for (int i = 0; i < size; i++) {
            if ((float)Math.abs(B.getQuick(i)) > 0.1) {
                B.setQuick(i, -1);
            }
        }
        AssertUtils.assertArrayEquals((float[]) B.elements(), (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        B = A.copy();
        Av.assign(new FloatProcedure() {
            public boolean apply(float element) {
                if ((float)Math.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        }, -1);
        for (int i = 0; i < size; i++) {
            if ((float)Math.abs(B.getQuick(i)) > 0.1) {
                B.setQuick(i, -1);
            }
        }
        for (int i = 0; i < size; i++) {
            assertEquals(B.getQuick(i), Av.getQuick(size - 1 - i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        Av = A.viewFlip();
        B = A.copy();
        Av.assign(new FloatProcedure() {
            public boolean apply(float element) {
                if ((float)Math.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        }, -1);
        for (int i = 0; i < size; i++) {
            if ((float)Math.abs(B.getQuick(i)) > 0.1) {
                B.setQuick(i, -1);
            }
        }
        for (int i = 0; i < size; i++) {
            assertEquals(B.getQuick(i), Av.getQuick(size - 1 - i), tol);
        }
    }

    @Test
    public void testAssignFloat() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(size);
        float value = (float)Math.random();
        A.assign(value);
        float[] aElts = (float[]) A.elements();
        for (int i = 0; i < aElts.length; i++) {
            assertEquals(value, aElts[i], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        A.assign(value);
        aElts = (float[]) A.elements();
        for (int i = 0; i < aElts.length; i++) {
            assertEquals(value, aElts[i], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(size);
        FloatMatrix1D Av = A.viewFlip();
        value = (float)Math.random();
        Av.assign(value);
        for (int i = 0; i < size; i++) {
            assertEquals(value, Av.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        Av = A.viewFlip();
        value = (float)Math.random();
        Av.assign(value);
        for (int i = 0; i < size; i++) {
            assertEquals(value, Av.getQuick(i), tol);
        }
    }

    @Test
    public void testAssignFloatArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(size);
        A.assign(a);
        float[] aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a, aElts, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        A.assign(a);
        aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a, aElts, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(size);
        FloatMatrix1D Av = A.viewFlip();
        Av.assign(a);
        for (int i = 0; i < size; i++) {
            assertEquals(a[i], Av.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        Av = A.viewFlip();
        Av.assign(a);
        for (int i = 0; i < size; i++) {
            assertEquals(a[i], Av.getQuick(i), tol);
        }
    }

    @Test
    public void testAssignFloatFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        A.assign(FloatFunctions.acos);
        float[] aElts = (float[]) A.elements();
        float tmp;
        for (int i = 0; i < a.length; i++) {
            tmp = (float)Math.acos(a[i]);
            assertEquals(tmp, aElts[i], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        A.assign(FloatFunctions.acos);
        aElts = (float[]) A.elements();
        for (int i = 0; i < a.length; i++) {
            tmp = (float)Math.acos(a[i]);
            assertEquals(tmp, aElts[i], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        Av.assign(FloatFunctions.acos);
        for (int i = 0; i < a.length; i++) {
            tmp = (float)Math.acos(a[i]);
            assertEquals(tmp, Av.getQuick(size - 1 - i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        Av = A.viewFlip();
        Av.assign(FloatFunctions.acos);
        for (int i = 0; i < a.length; i++) {
            tmp = (float)Math.acos(a[i]);
            assertEquals(tmp, Av.getQuick(size - 1 - i), tol);
        }
    }

    @Test
    public void testAssignFloatMatrix1D() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(size);
        FloatMatrix1D B = new DenseFloatMatrix1D(a);
        A.assign(B);
        float[] aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a, aElts, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        B = new DenseFloatMatrix1D(a);
        A.assign(B);
        aElts = (float[]) A.elements();
        AssertUtils.assertArrayEquals(a, aElts, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(size);
        FloatMatrix1D Av = A.viewFlip();
        B = new DenseFloatMatrix1D(a);
        FloatMatrix1D Bv = B.viewFlip();
        Av.assign(Bv);
        for (int i = 0; i < a.length; i++) {
            assertEquals(a[size - 1 - i], Av.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        Av = A.viewFlip();
        B = new DenseFloatMatrix1D(a);
        Bv = B.viewFlip();
        Av.assign(Bv);
        for (int i = 0; i < a.length; i++) {
            assertEquals(a[size - 1 - i], Av.getQuick(i), tol);
        }
    }

    @Test
    public void testAssignFloatMatrix1DFloatFloatFunction() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatMatrix1D B = new DenseFloatMatrix1D(b);
        A.assign(B, FloatFunctions.div);
        float[] aElts = (float[]) A.elements();
        float tmp;
        for (int i = 0; i < aElts.length; i++) {
            tmp = a[i] / b[i];
            assertEquals(tmp, aElts[i], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        B = new DenseFloatMatrix1D(b);
        A.assign(B, FloatFunctions.div);
        aElts = (float[]) A.elements();
        for (int i = 0; i < aElts.length; i++) {
            tmp = a[i] / b[i];
            assertEquals(tmp, aElts[i], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        B = new DenseFloatMatrix1D(b);
        FloatMatrix1D Bv = B.viewFlip();
        Av.assign(Bv, FloatFunctions.div);
        for (int i = 0; i < aElts.length; i++) {
            tmp = a[i] / b[i];
            assertEquals(tmp, Av.getQuick(size - 1 - i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseFloatMatrix1D(b);
        Bv = B.viewFlip();
        Av.assign(Bv, FloatFunctions.div);
        for (int i = 0; i < aElts.length; i++) {
            tmp = a[i] / b[i];
            assertEquals(tmp, Av.getQuick(size - 1 - i), tol);
        }
    }

    @Test
    public void testCardinality() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        int card = A.cardinality();
        assertEquals(a.length, card);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        card = A.cardinality();
        assertEquals(a.length, card);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        card = Av.cardinality();
        assertEquals(a.length, card);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        Av = A.viewFlip();
        card = Av.cardinality();
        assertEquals(a.length, card);
    }

    @Test
    public void testDct() {
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
        FloatMatrix1D A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        A.dct(true);
        A.idct(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(nThreadsBegin);

        A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        A.dct(true);
        A.idct(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
        A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        FloatMatrix1D Av = A.viewPart(size / 2 - 1, size / 2);
        float[] av_elems = (float[]) Av.copy().elements();
        Av.dct(true);
        Av.idct(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        Av = A.viewPart(size / 2 - 1, size / 2);
        av_elems = (float[]) Av.copy().elements();
        Av.dct(true);
        Av.idct(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
    }
    
    @Test
    public void testDht() {
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
        FloatMatrix1D A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        A.dht();
        A.idht(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(nThreadsBegin);

        A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        A.dht();
        A.idht(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
        A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        FloatMatrix1D Av = A.viewPart(size / 2 - 1, size / 2);
        float[] av_elems = (float[]) Av.copy().elements();
        Av.dht();
        Av.idht(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        Av = A.viewPart(size / 2 - 1, size / 2);
        av_elems = (float[]) Av.copy().elements();
        Av.dht();
        Av.idht(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
    }


    @Test
    public void testDst() {
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
        FloatMatrix1D A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        A.dst(true);
        A.idst(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        A.dst(true);
        A.idst(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
        A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        FloatMatrix1D Av = A.viewPart(size / 2 - 1, size / 2);
        float[] av_elems = (float[]) Av.copy().elements();
        Av.dst(true);
        Av.idst(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        Av = A.viewPart(size / 2 - 1, size / 2);
        av_elems = (float[]) Av.copy().elements();
        Av.dst(true);
        Av.idst(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
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
        FloatMatrix1D A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        A.fft();
        A.ifft(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        A.fft();
        A.ifft(true);
        AssertUtils.assertArrayEquals(a_1d, (float[]) A.elements(), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
        A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        FloatMatrix1D Av = A.viewPart(size / 2 - 1, size / 2);
        float[] av_elems = (float[]) Av.copy().elements();
        Av.fft();
        Av.ifft(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        Av = A.viewPart(size / 2 - 1, size / 2);
        av_elems = (float[]) Av.copy().elements();
        Av.fft();
        Av.ifft(true);
        AssertUtils.assertArrayEquals(av_elems, (float[]) Av.copy().elements(), tol);
    }

    @Test
    public void testGetFft() {
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
        FloatMatrix1D A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        FComplexMatrix1D Ac = A.getFft();
        Ac.ifft(true);
        float[] ac_elems = (float[]) Ac.elements();
        for (int i = 0; i < size; i++) {
            assertEquals(a_1d[i], ac_elems[2 * i], tol);
            assertEquals(0, ac_elems[2 * i + 1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        Ac = A.getFft();
        Ac.ifft(true);
        ac_elems = (float[]) Ac.elements();
        for (int i = 0; i < size; i++) {
            assertEquals(a_1d[i], ac_elems[2 * i], tol);
            assertEquals(0, ac_elems[2 * i + 1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
        A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        FloatMatrix1D Av = A.viewPart(size / 2 - 1, size / 2);
        float[] av_elems = (float[]) Av.copy().elements();
        Ac = Av.getFft();
        Ac.ifft(true);
        ac_elems = (float[]) Ac.elements();
        for (int i = 0; i < av_elems.length; i++) {
            assertEquals(av_elems[i], ac_elems[2 * i], tol);
            assertEquals(0, ac_elems[2 * i + 1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        Av = A.viewPart(size / 2 - 1, size / 2);
        av_elems = (float[]) Av.copy().elements();
        Ac = Av.getFft();
        Ac.ifft(true);
        ac_elems = (float[]) Ac.elements();
        for (int i = 0; i < av_elems.length; i++) {
            assertEquals(av_elems[i], ac_elems[2 * i], tol);
            assertEquals(0, ac_elems[2 * i + 1], tol);
        }
    }

    @Test
    public void testGetIfft() {
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
        FloatMatrix1D A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        FComplexMatrix1D Ac = A.getIfft(true);
        Ac.fft();
        float[] ac_elems = (float[]) Ac.elements();
        for (int i = 0; i < size; i++) {
            assertEquals(a_1d[i], ac_elems[2 * i], tol);
            assertEquals(0, ac_elems[2 * i + 1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        Ac = A.getIfft(true);
        Ac.fft();
        ac_elems = (float[]) Ac.elements();
        for (int i = 0; i < size; i++) {
            assertEquals(a_1d[i], ac_elems[2 * i], tol);
            assertEquals(0, ac_elems[2 * i + 1], tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(Integer.MAX_VALUE);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(Integer.MAX_VALUE);
        A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        FloatMatrix1D Av = A.viewPart(size / 2 - 1, size / 2);
        float[] av_elems = (float[]) Av.copy().elements();
        Ac = Av.getIfft(true);
        Ac.fft();
        ac_elems = (float[]) Ac.elements();
        for (int i = 0; i < av_elems.length; i++) {
            assertEquals(av_elems[i], ac_elems[2 * i], tol);
            assertEquals(0, ac_elems[2 * i + 1], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_2Threads(nThreadsBegin);
        ConcurrencyUtils.setThreadsBeginN_1D_FFT_4Threads(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        A.assign(a_1d);
        Av = A.viewPart(size / 2 - 1, size / 2);
        av_elems = (float[]) Av.copy().elements();
        Ac = Av.getIfft(true);
        Ac.fft();
        ac_elems = (float[]) Ac.elements();
        for (int i = 0; i < av_elems.length; i++) {
            assertEquals(av_elems[i], ac_elems[2 * i], tol);
            assertEquals(0, ac_elems[2 * i + 1], tol);
        }
    }

    @Test
    public void testEqualsFloat() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(size);
        float value = 1;
        A.assign(1);
        boolean eq = A.equals(value);
        assertEquals(true, eq);
        eq = A.equals(2);
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        A.assign(1);
        eq = A.equals(value);
        assertEquals(true, eq);
        eq = A.equals(2);
        assertEquals(false, eq);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(size);
        FloatMatrix1D Av = A.viewFlip();
        value = 1;
        Av.assign(1);
        eq = Av.equals(value);
        assertEquals(true, eq);
        eq = Av.equals(2);
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        Av = A.viewFlip();
        value = 1;
        Av.assign(1);
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
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatMatrix1D B = new DenseFloatMatrix1D(b);
        boolean eq = A.equals(A);
        assertEquals(true, eq);
        eq = A.equals(B);
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        B = new DenseFloatMatrix1D(b);
        eq = A.equals(A);
        assertEquals(true, eq);
        eq = A.equals(B);
        assertEquals(false, eq);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        B = new DenseFloatMatrix1D(b);
        FloatMatrix1D Bv = B.viewFlip();
        eq = Av.equals(Av);
        assertEquals(true, eq);
        eq = Av.equals(Bv);
        assertEquals(false, eq);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseFloatMatrix1D(b);
        Bv = B.viewFlip();
        eq = Av.equals(Av);
        assertEquals(true, eq);
        eq = Av.equals(Bv);
        assertEquals(false, eq);
    }

    @Test
    public void testGet() {
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        for (int i = 0; i < A.size(); i++) {
            assertEquals(a[i], A.getQuick(i), tol);
        }
    }

    @Test
    public void testGetMaxLocation() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(size);
        A.setQuick(size / 3, 0.7f);
        A.setQuick(size / 2, 0.7f);
        float[] maxAndLoc = A.getMaxLocation();
        assertEquals(0.7f, maxAndLoc[0], tol);
        assertEquals(size / 3, (int) maxAndLoc[1]);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        A.setQuick(size / 3, 0.7f);
        A.setQuick(size / 2, 0.7f);
        maxAndLoc = A.getMaxLocation();
        assertEquals(0.7f, maxAndLoc[0], tol);
        assertEquals(size / 3, (int) maxAndLoc[1]);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(size);
        FloatMatrix1D Av = A.viewFlip();
        Av.setQuick(size / 3, 0.7f);
        Av.setQuick(size / 2, 0.7f);
        maxAndLoc = Av.getMaxLocation();
        assertEquals(0.7f, maxAndLoc[0], tol);
        assertEquals(size / 3, (int) maxAndLoc[1]);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        Av = A.viewFlip();
        Av.setQuick(size / 3, 0.7f);
        Av.setQuick(size / 2, 0.7f);
        maxAndLoc = Av.getMaxLocation();
        assertEquals(0.7f, maxAndLoc[0], tol);
        assertEquals(size / 3, (int) maxAndLoc[1]);

    }

    @Test
    public void testGetMinLocation() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(size);
        A.setQuick(size / 3, -0.7f);
        A.setQuick(size / 2, -0.7f);
        float[] minAndLoc = A.getMinLocation();
        assertEquals(-0.7f, minAndLoc[0], tol);
        assertEquals(size / 3, (int) minAndLoc[1]);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        A.setQuick(size / 3, -0.7f);
        A.setQuick(size / 2, -0.7f);
        minAndLoc = A.getMinLocation();
        assertEquals(-0.7f, minAndLoc[0], tol);
        assertEquals(size / 3, (int) minAndLoc[1]);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(size);
        FloatMatrix1D Av = A.viewFlip();
        Av.setQuick(size / 3, -0.7f);
        Av.setQuick(size / 2, -0.7f);
        minAndLoc = Av.getMinLocation();
        assertEquals(-0.7f, minAndLoc[0], tol);
        assertEquals(size / 3, (int) minAndLoc[1]);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        Av = A.viewFlip();
        Av.setQuick(size / 3, -0.7f);
        Av.setQuick(size / 2, -0.7f);
        minAndLoc = Av.getMinLocation();
        assertEquals(-0.7f, minAndLoc[0], tol);
        assertEquals(size / 3, (int) minAndLoc[1]);
    }

    @Test
    public void testGetNonZerosIntArrayListFloatArrayList() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        IntArrayList indexList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        A.getNonZeros(indexList, valueList);
        assertEquals(size, indexList.size());
        assertEquals(size, valueList.size());
        for (int i = 0; i < size; i++) {
            assertEquals(A.getQuick(indexList.get(i)), valueList.get(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        A.getNonZeros(indexList, valueList);
        assertEquals(size, indexList.size());
        assertEquals(size, valueList.size());
        for (int i = 0; i < size; i++) {
            assertEquals(A.getQuick(indexList.get(i)), valueList.get(i), tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        Av.getNonZeros(indexList, valueList);
        assertEquals(size, indexList.size());
        assertEquals(size, valueList.size());
        for (int i = 0; i < size; i++) {
            assertEquals(Av.getQuick(indexList.get(i)), valueList.get(i), tol);
        }

        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        Av = A.viewFlip();
        Av.getNonZeros(indexList, valueList);
        assertEquals(size, indexList.size());
        assertEquals(size, valueList.size());
        for (int i = 0; i < size; i++) {
            assertEquals(Av.getQuick(indexList.get(i)), valueList.get(i), tol);
        }
    }

    @Test
    public void testGetPositiveValuesIntArrayListFloatArrayList() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        IntArrayList indexList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        A.getPositiveValues(indexList, valueList);
        assertEquals(size, indexList.size());
        assertEquals(size, valueList.size());
        for (int i = 0; i < size; i++) {
            assertEquals(A.getQuick(indexList.get(i)), valueList.get(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        A.getPositiveValues(indexList, valueList);
        assertEquals(size, indexList.size());
        assertEquals(size, valueList.size());
        for (int i = 0; i < size; i++) {
            assertEquals(A.getQuick(indexList.get(i)), valueList.get(i), tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        Av.getPositiveValues(indexList, valueList);
        assertEquals(size, indexList.size());
        assertEquals(size, valueList.size());
        for (int i = 0; i < size; i++) {
            assertEquals(Av.getQuick(indexList.get(i)), valueList.get(i), tol);
        }

        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        Av = A.viewFlip();
        Av.getPositiveValues(indexList, valueList);
        assertEquals(size, indexList.size());
        assertEquals(size, valueList.size());
        for (int i = 0; i < size; i++) {
            assertEquals(Av.getQuick(indexList.get(i)), valueList.get(i), tol);
        }
    }
    
    @Test
    public void testGetNegativeValuesIntArrayListFloatArrayList() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        A.assign(FloatFunctions.mult(-1));
        IntArrayList indexList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        A.getNegativeValues(indexList, valueList);
        assertEquals(size, indexList.size());
        assertEquals(size, valueList.size());
        for (int i = 0; i < size; i++) {
            assertEquals(A.getQuick(indexList.get(i)), valueList.get(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        A.assign(FloatFunctions.mult(-1));
        A.getNegativeValues(indexList, valueList);
        assertEquals(size, indexList.size());
        assertEquals(size, valueList.size());
        for (int i = 0; i < size; i++) {
            assertEquals(A.getQuick(indexList.get(i)), valueList.get(i), tol);
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        A.assign(FloatFunctions.mult(-1));
        FloatMatrix1D Av = A.viewFlip();
        Av.getNegativeValues(indexList, valueList);
        assertEquals(size, indexList.size());
        assertEquals(size, valueList.size());
        for (int i = 0; i < size; i++) {
            assertEquals(Av.getQuick(indexList.get(i)), valueList.get(i), tol);
        }

        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        A.assign(FloatFunctions.mult(-1));
        Av = A.viewFlip();
        Av.getNegativeValues(indexList, valueList);
        assertEquals(size, indexList.size());
        assertEquals(size, valueList.size());
        for (int i = 0; i < size; i++) {
            assertEquals(Av.getQuick(indexList.get(i)), valueList.get(i), tol);
        }
    }
    
    @Test
    public void testGetNonZerosIntArrayListFloatArrayListInt() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(size);
        A.setQuick(size / 4, 1);
        A.setQuick(size / 2, 2);
        A.setQuick(size - 1, 3);
        IntArrayList indexList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        A.getNonZeros(indexList, valueList, 2);
        assertEquals(2, indexList.size());
        assertEquals(2, valueList.size());
        assertEquals(size / 4, indexList.get(0));
        assertEquals(size / 2, indexList.get(1));
        assertEquals(1, valueList.get(0), tol);
        assertEquals(2, valueList.get(1), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        A.setQuick(size / 4, 1);
        A.setQuick(size / 2, 2);
        A.setQuick(size - 1, 3);
        indexList = new IntArrayList();
        valueList = new FloatArrayList();
        A.getNonZeros(indexList, valueList, 2);
        assertEquals(2, indexList.size());
        assertEquals(2, valueList.size());
        assertEquals(size / 4, indexList.get(0));
        assertEquals(size / 2, indexList.get(1));
        assertEquals(1, valueList.get(0), tol);
        assertEquals(2, valueList.get(1), tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(size);
        FloatMatrix1D Av = A.viewFlip();
        Av.setQuick(size / 4, 1);
        Av.setQuick(size / 2, 2);
        Av.setQuick(size - 1, 3);
        indexList = new IntArrayList();
        valueList = new FloatArrayList();
        Av.getNonZeros(indexList, valueList, 2);
        assertEquals(2, indexList.size());
        assertEquals(2, valueList.size());
        assertEquals(size / 4, indexList.get(0));
        assertEquals(size / 2, indexList.get(1));
        assertEquals(1, valueList.get(0), tol);
        assertEquals(2, valueList.get(1), tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(size);
        Av = A.viewFlip();
        Av.setQuick(size / 4, 1);
        Av.setQuick(size / 2, 2);
        Av.setQuick(size - 1, 3);
        indexList = new IntArrayList();
        valueList = new FloatArrayList();
        Av.getNonZeros(indexList, valueList, 2);
        assertEquals(2, indexList.size());
        assertEquals(2, valueList.size());
        assertEquals(size / 4, indexList.get(0));
        assertEquals(size / 2, indexList.get(1));
        assertEquals(1, valueList.get(0), tol);
        assertEquals(2, valueList.get(1), tol);
    }

    @Test
    public void testGetQuick() {
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        for (int i = 0; i < A.size(); i++) {
            assertEquals(a[i], A.getQuick(i), tol);
        }
    }

    @Test
    public void testReshapeIntInt() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        int rows = 10;
        int cols = 17;
        FloatMatrix2D B = A.reshape(rows, cols);
        int idx = 0;
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                assertEquals(A.getQuick(idx++), B.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        B = A.reshape(rows, cols);
        idx = 0;
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                assertEquals(A.getQuick(idx++), B.getQuick(r, c), tol);
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        B = Av.reshape(rows, cols);
        idx = 0;
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                assertEquals(Av.getQuick(idx++), B.getQuick(r, c), tol);
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        Av = A.viewFlip();
        B = Av.reshape(rows, cols);
        idx = 0;
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                assertEquals(Av.getQuick(idx++), B.getQuick(r, c), tol);
            }
        }
    }

    @Test
    public void testReshapeIntIntInt() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        int slices = 2;
        int rows = 5;
        int cols = 17;
        FloatMatrix3D B = A.reshape(slices, rows, cols);
        int idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int c = 0; c < cols; c++) {
                for (int r = 0; r < rows; r++) {
                    assertEquals(A.getQuick(idx++), B.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        B = A.reshape(slices, rows, cols);
        idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int c = 0; c < cols; c++) {
                for (int r = 0; r < rows; r++) {
                    assertEquals(A.getQuick(idx++), B.getQuick(s, r, c), tol);
                }
            }
        }
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        B = Av.reshape(slices, rows, cols);
        idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int c = 0; c < cols; c++) {
                for (int r = 0; r < rows; r++) {
                    assertEquals(Av.getQuick(idx++), B.getQuick(s, r, c), tol);
                }
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        Av = A.viewFlip();
        B = Av.reshape(slices, rows, cols);
        idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int c = 0; c < cols; c++) {
                for (int r = 0; r < rows; r++) {
                    assertEquals(Av.getQuick(idx++), B.getQuick(s, r, c), tol);
                }
            }
        }
    }

    @Test
    public void testSet() {
        FloatMatrix1D A = new DenseFloatMatrix1D(size);
        float elem = (float)Math.random();
        A.set(size / 2, elem);
        assertEquals(elem, A.getQuick(size / 2), tol);
    }

    @Test
    public void testSetQuick() {
        FloatMatrix1D A = new DenseFloatMatrix1D(size);
        float elem = (float)Math.random();
        A.setQuick(size / 2, elem);
        assertEquals(elem, A.getQuick(size / 2), tol);
    }

    @Test
    public void testSwap() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatMatrix1D B = new DenseFloatMatrix1D(b);
        A.swap(B);
        float[] aElts = (float[]) A.elements();
        float[] bElts = (float[]) B.elements();
        AssertUtils.assertArrayEquals(b, aElts, tol);
        AssertUtils.assertArrayEquals(a, bElts, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        B = new DenseFloatMatrix1D(b);
        A.swap(B);
        aElts = (float[]) A.elements();
        bElts = (float[]) B.elements();
        AssertUtils.assertArrayEquals(b, aElts, tol);
        AssertUtils.assertArrayEquals(a, bElts, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        B = new DenseFloatMatrix1D(b);
        FloatMatrix1D Bv = B.viewFlip();
        Av.swap(Bv);
        for (int i = 0; i < size; i++) {
            assertEquals(a[i], Bv.getQuick(size - 1 - i), tol);
            assertEquals(b[i], Av.getQuick(size - 1 - i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseFloatMatrix1D(b);
        Bv = B.viewFlip();
        Av.swap(Bv);
        for (int i = 0; i < size; i++) {
            assertEquals(a[i], Bv.getQuick(size - 1 - i), tol);
            assertEquals(b[i], Av.getQuick(size - 1 - i), tol);
        }
    }

    @Test
    public void testToArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        float[] array = A.toArray();
        AssertUtils.assertArrayEquals(a, array, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        array = A.toArray();
        AssertUtils.assertArrayEquals(a, array, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        array = Av.toArray();
        for (int i = 0; i < size; i++) {
            assertEquals(a[i], array[size - 1 - i], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        Av = A.viewFlip();
        array = Av.toArray();
        for (int i = 0; i < size; i++) {
            assertEquals(a[i], array[size - 1 - i], tol);
        }
    }

    @Test
    public void testToArrayFloatArray() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        float[] b = new float[size];
        A.toArray(b);
        AssertUtils.assertArrayEquals(a, b, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        A.toArray(b);
        AssertUtils.assertArrayEquals(a, b, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        b = new float[size];
        Av.toArray(b);
        for (int i = 0; i < size; i++) {
            assertEquals(a[i], b[size - 1 - i], tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        Av = A.viewFlip();
        b = new float[size];
        Av.toArray(b);
        for (int i = 0; i < size; i++) {
            assertEquals(a[i], b[size - 1 - i], tol);
        }
    }

    @Test
    public void testViewFlip() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatMatrix1D B = A.viewFlip();
        for (int i = 0; i < size; i++) {
            assertEquals(A.getQuick(size - 1 - i), B.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        B = A.viewFlip();
        for (int i = 0; i < size; i++) {
            assertEquals(A.getQuick(size - 1 - i), B.getQuick(i), tol);
        }
    }

    @Test
    public void testViewPart() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatMatrix1D B = A.viewPart(15, 11);
        for (int i = 0; i < 11; i++) {
            assertEquals(a[15 + i], B.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        B = A.viewPart(15, 11);
        for (int i = 0; i < 11; i++) {
            assertEquals(a[15 + i], B.getQuick(i), tol);
        }
    }

    @Test
    public void testViewSelectionFloatProcedure() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatMatrix1D B = A.viewSelection(new FloatProcedure() {
            public boolean apply(float element) {
                return element % 2 == 0;
            }
        });
        for (int i = 0; i < B.size(); i++) {
            float el = B.getQuick(i);
            if (el % 2 != 0) {
                fail();
            }
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        B = A.viewSelection(new FloatProcedure() {
            public boolean apply(float element) {
                return element % 2 == 0;
            }
        });
        for (int i = 0; i < B.size(); i++) {
            float el = B.getQuick(i);
            if (el % 2 != 0) {
                fail();
            }
        }
    }

    @Test
    public void testViewSelectionIntArray() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        int[] indexes = new int[] { 5, 11, 22, 37, 101 };
        FloatMatrix1D B = A.viewSelection(indexes);
        for (int i = 0; i < indexes.length; i++) {
            assertEquals(A.getQuick(indexes[i]), B.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        indexes = new int[] { 5, 11, 22, 37, 101 };
        B = A.viewSelection(indexes);
        for (int i = 0; i < indexes.length; i++) {
            assertEquals(A.getQuick(indexes[i]), B.getQuick(i), tol);
        }
    }

    @Test
    public void testViewSorted() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatMatrix1D B = A.viewSorted();
        float[] b = Arrays.copyOf(a, a.length);
        Arrays.sort(b);
        for (int i = 0; i < b.length; i++) {
            assertEquals(b[i], B.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        B = A.viewSorted();
        b = Arrays.copyOf(a, a.length);
        Arrays.sort(b);
        for (int i = 0; i < b.length; i++) {
            assertEquals(b[i], B.getQuick(i), tol);
        }
    }

    @Test
    public void testViewStrides() {
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        int stride = 3;
        FloatMatrix1D B = A.viewStrides(stride);
        for (int i = 0; i < B.size(); i++) {
            assertEquals(A.getQuick(i * stride), B.getQuick(i), tol);
        }
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        B = A.viewStrides(stride);
        for (int i = 0; i < B.size(); i++) {
            assertEquals(A.getQuick(i * stride), B.getQuick(i), tol);
        }
    }

    @Test
    public void testZDotProductFloatMatrix1D() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatMatrix1D B = new DenseFloatMatrix1D(b);
        float product = A.zDotProduct(B);
        float tmpSumMult = 0;
        for (int i = 0; i < a.length; i++) {
            tmpSumMult += a[i] * b[i];
        }
        assertEquals(tmpSumMult, product, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        B = new DenseFloatMatrix1D(b);
        product = A.zDotProduct(B);
        tmpSumMult = 0;
        for (int i = 0; i < a.length; i++) {
            tmpSumMult += a[i] * b[i];
        }
        assertEquals(tmpSumMult, product, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        B = new DenseFloatMatrix1D(b);
        FloatMatrix1D Bv = B.viewFlip();
        product = Av.zDotProduct(Bv);
        tmpSumMult = 0;
        for (int i = 0; i < a.length; i++) {
            tmpSumMult += a[i] * b[i];
        }
        assertEquals(tmpSumMult, product, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseFloatMatrix1D(b);
        Bv = B.viewFlip();
        product = Av.zDotProduct(Bv);
        tmpSumMult = 0;
        for (int i = 0; i < a.length; i++) {
            tmpSumMult += a[i] * b[i];
        }
        assertEquals(tmpSumMult, product, tol);
    }

    @Test
    public void testZDotProductFloatMatrix1DIntInt() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatMatrix1D B = new DenseFloatMatrix1D(b);
        float product = A.zDotProduct(B, 5, B.size() - 10);
        float tmpSumMult = 0;
        for (int i = 5; i < a.length - 5; i++) {
            tmpSumMult += a[i] * b[i];
        }
        assertEquals(tmpSumMult, product, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        B = new DenseFloatMatrix1D(b);
        product = A.zDotProduct(B, 5, B.size() - 10);
        tmpSumMult = 0;
        for (int i = 5; i < a.length - 5; i++) {
            tmpSumMult += a[i] * b[i];
        }
        assertEquals(tmpSumMult, product, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        B = new DenseFloatMatrix1D(b);
        FloatMatrix1D Bv = B.viewFlip();
        product = Av.zDotProduct(Bv, 5, Bv.size() - 10);
        tmpSumMult = 0;
        for (int i = 5; i < a.length - 5; i++) {
            tmpSumMult += a[i] * b[i];
        }
        assertEquals(tmpSumMult, product, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseFloatMatrix1D(b);
        Bv = B.viewFlip();
        product = Av.zDotProduct(Bv, 5, Bv.size() - 10);
        tmpSumMult = 0;
        for (int i = 5; i < a.length - 5; i++) {
            tmpSumMult += a[i] * b[i];
        }
    }

    @Test
    public void testZDotProductFloatMatrix1DIntIntIntArrayList() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        FloatMatrix1D B = new DenseFloatMatrix1D(b);
        IntArrayList indexList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        B.getNonZeros(indexList, valueList);
        float product = A.zDotProduct(B, 5, B.size() - 10, indexList);
        float tmpSumMult = 0;
        for (int i = 5; i < a.length - 5; i++) {
            tmpSumMult += a[i] * b[i];
        }
        assertEquals(tmpSumMult, product, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        B = new DenseFloatMatrix1D(b);
        indexList = new IntArrayList();
        valueList = new FloatArrayList();
        B.getNonZeros(indexList, valueList);
        product = A.zDotProduct(B, 5, B.size() - 10, indexList);
        tmpSumMult = 0;
        for (int i = 5; i < a.length - 5; i++) {
            tmpSumMult += a[i] * b[i];
        }
        assertEquals(tmpSumMult, product, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        B = new DenseFloatMatrix1D(b);
        FloatMatrix1D Bv = B.viewFlip();
        indexList = new IntArrayList();
        valueList = new FloatArrayList();
        Bv.getNonZeros(indexList, valueList);
        product = Av.zDotProduct(Bv, 5, Bv.size() - 10, indexList);
        tmpSumMult = 0;
        for (int i = 5; i < a.length - 5; i++) {
            tmpSumMult += a[i] * b[i];
        }
        assertEquals(tmpSumMult, product, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        Av = A.viewFlip();
        B = new DenseFloatMatrix1D(b);
        Bv = B.viewFlip();
        indexList = new IntArrayList();
        valueList = new FloatArrayList();
        Bv.getNonZeros(indexList, valueList);
        product = Av.zDotProduct(Bv, 5, Bv.size() - 10, indexList);
        tmpSumMult = 0;
        for (int i = 5; i < a.length - 5; i++) {
            tmpSumMult += a[i] * b[i];
        }
        assertEquals(tmpSumMult, product, tol);
    }

    @Test
    public void testZSum() {
        /* No view */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        FloatMatrix1D A = new DenseFloatMatrix1D(a);
        float aSum = A.zSum();
        float tmpSum = 0;
        for (int i = 0; i < a.length; i++) {
            tmpSum += a[i];
        }
        assertEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        tmpSum = 0;
        for (int i = 0; i < a.length; i++) {
            tmpSum += a[i];
        }
        assertEquals(tmpSum, aSum, tol);
        /* View */
        // single thread
        ConcurrencyUtils.setNumberOfProcessors(1);
        A = new DenseFloatMatrix1D(a);
        FloatMatrix1D Av = A.viewFlip();
        aSum = Av.zSum();
        tmpSum = 0;
        for (int i = 0; i < a.length; i++) {
            tmpSum += a[i];
        }
        assertEquals(tmpSum, aSum, tol);
        // multiple threads
        ConcurrencyUtils.setNumberOfProcessors(nThreads);
        ConcurrencyUtils.setThreadsBeginN_1D(nThreadsBegin);
        A = new DenseFloatMatrix1D(a);
        Av = A.viewFlip();
        aSum = Av.zSum();
        tmpSum = 0;
        for (int i = 0; i < a.length; i++) {
            tmpSum += a[i];
        }
        assertEquals(tmpSum, aSum, tol);
    }

}
