package cern.colt.matrix.tlong;

import java.util.Random;

import junit.framework.TestCase;
import cern.colt.function.tlong.IntIntLongFunction;
import cern.colt.function.tlong.LongProcedure;
import cern.colt.list.tint.IntArrayList;
import cern.colt.list.tlong.LongArrayList;
import cern.colt.matrix.tlong.impl.DenseLongMatrix1D;
import cern.jet.math.tlong.LongFunctions;
import edu.emory.mathcs.utils.pc.ConcurrencyUtils;

public abstract class LongMatrix2DTest extends TestCase {
    /**
     * Matrix to test
     */
    protected LongMatrix2D A;

    /**
     * Matrix of the same size as A
     */
    protected LongMatrix2D B;

    /**
     * Matrix of the size A.columns() x A.rows()
     */
    protected LongMatrix2D Bt;

    protected int NROWS = 13;

    protected int NCOLUMNS = 17;

    protected Random rand = new Random(0);

    /**
     * Constructor for LongMatrix2DTest
     */
    public LongMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void setUp() throws Exception {
        createMatrices();
        populateMatrices();
    }

    protected abstract void createMatrices() throws Exception;

    protected void populateMatrices() {
        ConcurrencyUtils.setThreadsBeginN_2D(1);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                A.setQuick(r, c, Math.max(1, rand.nextLong() % A.rows()));
            }
        }

        for (int r = 0; r < B.rows(); r++) {
            for (int c = 0; c < B.columns(); c++) {
                B.setQuick(r, c, Math.max(1, rand.nextLong() % B.rows()));
            }
        }

        for (int r = 0; r < Bt.rows(); r++) {
            for (int c = 0; c < Bt.columns(); c++) {
                Bt.setQuick(r, c, Math.max(1, rand.nextLong() % Bt.rows()));
            }
        }
    }

    protected void tearDown() throws Exception {
        A = B = Bt = null;
    }

    //    public void testToString() {
    //        System.out.println(A.toString());
    //    }

    public void testAggregateLongLongFunctionLongFunction() {
        long expected = 0;
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                long elem = A.getQuick(r, c);
                expected += elem * elem;
            }
        }
        long result = A.aggregate(LongFunctions.plus, LongFunctions.square);
        assertEquals(expected, result);
    }

    public void testAggregateLongLongFunctionLongFunctionLongProcedure() {
        LongProcedure procedure = new LongProcedure() {
            public boolean apply(long element) {
                if (Math.abs(element) > 0.2) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        long expected = 0;
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                long elem = A.getQuick(r, c);
                if (Math.abs(elem) > 0.2) {
                    expected += elem * elem;
                }
            }
        }

        long result = A.aggregate(LongFunctions.plus, LongFunctions.square, procedure);
        assertEquals(expected, result);
    }

    public void testAggregateLongLongFunctionLongFunctionIntArrayListIntArrayList() {
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                rowList.add(r);
                columnList.add(c);
            }
        }
        long expected = 0;
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                long elem = A.getQuick(r, c);
                expected += elem * elem;
            }
        }
        long result = A.aggregate(LongFunctions.plus, LongFunctions.square, rowList, columnList);
        assertEquals(expected, result);
    }

    public void testAggregateLongMatrix2DLongLongFunctionLongLongFunction() {
        long expected = 0;
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                long elemA = A.getQuick(r, c);
                long elemB = B.getQuick(r, c);
                expected += elemA * elemB;
            }
        }
        long result = A.aggregate(B, LongFunctions.plus, LongFunctions.mult);
        assertEquals(expected, result);
    }

    public void testAssignLong() {
        long value = rand.nextLong();
        A.assign(value);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        LongArrayList valueList = new LongArrayList();
        A.getNonZeros(rowList, columnList, valueList);
        for (int i = 0; i < valueList.size(); i++) {
            assertEquals(value, valueList.getQuick(i));
        }
    }

    public void testAssignLongArrayArray() {
        long[][] expected = new long[A.rows()][A.columns()];
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                expected[r][c] = rand.nextLong();
            }
        }
        A.assign(expected);
        for (int r = 0; r < A.rows(); r++) {
            assertTrue(A.columns() == expected[r].length);
            for (int c = 0; c < A.columns(); c++)
                assertEquals(expected[r][c], A.getQuick(r, c));
        }
    }

    public void testAssignLongFunction() {
        LongMatrix2D Acopy = A.copy();
        A.assign(LongFunctions.neg);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                long expected = -Acopy.getQuick(r, c);
                assertEquals(expected, A.getQuick(r, c));
            }
        }
    }

    public void testAssignLongMatrix2D() {
        A.assign(B);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++)
                assertEquals(B.getQuick(r, c), A.getQuick(r, c));
        }
    }

    public void testAssignLongMatrix2DLongLongFunction() {
        LongMatrix2D Acopy = A.copy();
        A.assign(B, LongFunctions.plus);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(Acopy.getQuick(r, c) + B.getQuick(r, c), A.getQuick(r, c));
            }
        }
    }

    public void testAssignLongMatrix2DLongLongFunctionIntArrayListIntArrayList() {
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                rowList.add(r);
                columnList.add(c);
            }
        }
        LongMatrix2D Acopy = A.copy();
        A.assign(B, LongFunctions.plus, rowList, columnList);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(Acopy.getQuick(r, c) + B.getQuick(r, c), A.getQuick(r, c));
            }
        }
    }

    public void testAssignLongProcedureLong() {
        LongProcedure procedure = new LongProcedure() {
            public boolean apply(long element) {
                if (Math.abs(element) > 1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        LongMatrix2D Acopy = A.copy();
        A.assign(procedure, -1);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                if (Math.abs(Acopy.getQuick(r, c)) > 1) {
                    assertEquals(-1, A.getQuick(r, c));
                } else {
                    assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c));
                }
            }
        }
    }

    public void testAssignLongProcedureLongFunction() {
        LongProcedure procedure = new LongProcedure() {
            public boolean apply(long element) {
                if (Math.abs(element) > 1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        LongMatrix2D Acopy = A.copy();
        A.assign(procedure, LongFunctions.neg);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                if (Math.abs(Acopy.getQuick(r, c)) > 1) {
                    assertEquals(-Acopy.getQuick(r, c), A.getQuick(r, c));
                } else {
                    assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c));
                }
            }
        }
    }

    public void testCardinality() {
        int card = A.cardinality();
        int expected = 0;
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                if (A.getQuick(r, c) != 0)
                    expected++;
            }
        }
        assertEquals(expected, card);
    }

    public void testEqualsLong() {
        long value = 1;
        A.assign(value);
        boolean eq = A.equals(value);
        assertTrue(eq);
        eq = A.equals(2);
        assertFalse(eq);
    }

    public void testEqualsObject() {
        boolean eq = A.equals(A);
        assertTrue(eq);
        eq = A.equals(B);
        assertFalse(eq);
    }

    public void testForEachNonZero() {
        LongMatrix2D Acopy = A.copy();
        IntIntLongFunction function = new IntIntLongFunction() {
            public long apply(int first, int second, long third) {
                return -third;
            }
        };
        A.forEachNonZero(function);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(-Acopy.getQuick(r, c), A.getQuick(r, c));
            }
        }
    }

    public void testMaxLocation() {
        A.assign(0);
        A.setQuick(A.rows() / 3, A.columns() / 3, 7);
        A.setQuick(A.rows() / 2, A.columns() / 2, 1);
        long[] maxAndLoc = A.getMaxLocation();
        assertEquals(7, maxAndLoc[0]);
        assertEquals(A.rows() / 3, (int) maxAndLoc[1]);
        assertEquals(A.columns() / 3, (int) maxAndLoc[2]);
    }

    public void testMinLocation() {
        A.assign(0);
        A.setQuick(A.rows() / 3, A.columns() / 3, -7);
        A.setQuick(A.rows() / 2, A.columns() / 2, -1);
        long[] minAndLoc = A.getMinLocation();
        assertEquals(-7, minAndLoc[0]);
        assertEquals(A.rows() / 3, (int) minAndLoc[1]);
        assertEquals(A.columns() / 3, (int) minAndLoc[2]);
    }

    public void testGetNegativeValues() {
        A.assign(0);
        A.setQuick(A.rows() / 3, A.columns() / 3, -7);
        A.setQuick(A.rows() / 2, A.columns() / 2, -1);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        LongArrayList valueList = new LongArrayList();
        A.getNegativeValues(rowList, columnList, valueList);
        assertEquals(2, rowList.size());
        assertEquals(2, columnList.size());
        assertEquals(2, valueList.size());
        assertTrue(rowList.contains(A.rows() / 3));
        assertTrue(rowList.contains(A.rows() / 2));
        assertTrue(columnList.contains(A.columns() / 3));
        assertTrue(columnList.contains(A.columns() / 2));
        assertTrue(valueList.contains(-7));
        assertTrue(valueList.contains(-1));
    }

    public void testGetNonZeros() {
        A.assign(0);
        A.setQuick(A.rows() / 3, A.columns() / 3, 7);
        A.setQuick(A.rows() / 2, A.columns() / 2, 1);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        LongArrayList valueList = new LongArrayList();
        A.getNonZeros(rowList, columnList, valueList);
        assertEquals(2, rowList.size());
        assertEquals(2, columnList.size());
        assertEquals(2, valueList.size());
        assertTrue(rowList.contains(A.rows() / 3));
        assertTrue(rowList.contains(A.rows() / 2));
        assertTrue(columnList.contains(A.columns() / 3));
        assertTrue(columnList.contains(A.columns() / 2));
        assertTrue(valueList.contains(7));
        assertTrue(valueList.contains(1));
    }

    public void testGetPositiveValues() {
        A.assign(0);
        A.setQuick(A.rows() / 3, A.columns() / 3, 7);
        A.setQuick(A.rows() / 2, A.columns() / 2, 1);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        LongArrayList valueList = new LongArrayList();
        A.getPositiveValues(rowList, columnList, valueList);
        assertEquals(2, rowList.size());
        assertEquals(2, columnList.size());
        assertEquals(2, valueList.size());
        assertTrue(rowList.contains(A.rows() / 3));
        assertTrue(rowList.contains(A.rows() / 2));
        assertTrue(columnList.contains(A.columns() / 3));
        assertTrue(columnList.contains(A.columns() / 2));
        assertTrue(valueList.contains(7));
        assertTrue(valueList.contains(1));
    }

    public void testToArray() {
        long[][] array = A.toArray();
        assertTrue(A.rows() == array.length);
        for (int r = 0; r < A.rows(); r++) {
            assertTrue(A.columns() == array[r].length);
            for (int c = 0; c < A.columns(); c++)
                assertEquals(0, Math.abs(array[r][c] - A.getQuick(r, c)));
        }
    }

    public void testVectorize() {
        LongMatrix1D Avec = A.vectorize();
        int idx = 0;
        for (int c = 0; c < A.columns(); c++) {
            for (int r = 0; r < A.rows(); r++) {
                assertEquals(A.getQuick(r, c), Avec.getQuick(idx++));
            }
        }
    }

    public void testViewColumn() {
        LongMatrix1D col = A.viewColumn(A.columns() / 2);
        assertEquals(A.rows(), col.size());
        for (int r = 0; r < A.rows(); r++) {
            assertEquals(A.getQuick(r, A.columns() / 2), col.getQuick(r));
        }
    }

    public void testViewColumnFlip() {
        LongMatrix2D B = A.viewColumnFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(r, A.columns() - 1 - c), B.getQuick(r, c));
            }
        }
    }

    public void testViewDice() {
        LongMatrix2D B = A.viewDice();
        assertEquals(A.rows(), B.columns());
        assertEquals(A.columns(), B.rows());
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(c, r));
            }
        }
    }

    public void testViewPart() {
        LongMatrix2D B = A.viewPart(A.rows() / 2, A.columns() / 2, A.rows() / 3, A.columns() / 3);
        assertEquals(A.rows() / 3, B.rows());
        assertEquals(A.columns() / 3, B.columns());
        for (int r = 0; r < A.rows() / 3; r++) {
            for (int c = 0; c < A.columns() / 3; c++) {
                assertEquals(A.getQuick(A.rows() / 2 + r, A.columns() / 2 + c), B.getQuick(r, c));
            }
        }
    }

    public void testViewRow() {
        LongMatrix1D B = A.viewRow(A.rows() / 2);
        assertEquals(A.columns(), B.size());
        for (int r = 0; r < A.columns(); r++) {
            assertEquals(A.getQuick(A.rows() / 2, r), B.getQuick(r));
        }
    }

    public void testViewRowFlip() {
        LongMatrix2D B = A.viewRowFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(A.rows() - 1 - r, c), B.getQuick(r, c));
            }
        }
    }

    public void testViewSelectionLongMatrix1DProcedure() {
        final long value = 2;
        A.assign(0);
        A.setQuick(A.rows() / 4, 0, value);
        A.setQuick(A.rows() / 2, 0, value);
        LongMatrix2D B = A.viewSelection(new LongMatrix1DProcedure() {
            public boolean apply(LongMatrix1D element) {
                if (Math.abs(element.getQuick(0) - value) == 0) {
                    return true;
                } else {
                    return false;
                }
            }
        });
        assertEquals(2, B.rows());
        assertEquals(A.columns(), B.columns());
        assertEquals(A.getQuick(A.rows() / 4, 0), B.getQuick(0, 0));
        assertEquals(A.getQuick(A.rows() / 2, 0), B.getQuick(1, 0));
    }

    public void testViewSelectionIntArrayIntArray() {
        int[] rowIndexes = new int[] { A.rows() / 6, A.rows() / 5, A.rows() / 4, A.rows() / 3, A.rows() / 2 };
        int[] colIndexes = new int[] { A.columns() / 6, A.columns() / 5, A.columns() / 4, A.columns() / 3,
                A.columns() / 2, A.columns() - 1 };
        LongMatrix2D B = A.viewSelection(rowIndexes, colIndexes);
        assertEquals(rowIndexes.length, B.rows());
        assertEquals(colIndexes.length, B.columns());
        for (int r = 0; r < rowIndexes.length; r++) {
            for (int c = 0; c < colIndexes.length; c++) {
                assertEquals(A.getQuick(rowIndexes[r], colIndexes[c]), B.getQuick(r, c));
            }
        }
    }

    public void testViewSorted() {
        LongMatrix2D B = A.viewSorted(1);
        for (int r = 0; r < A.rows() - 1; r++) {
            assertTrue(B.getQuick(r + 1, 1) >= B.getQuick(r, 1));
        }
    }

    public void testViewStrides() {
        int rowStride = 3;
        int colStride = 5;
        LongMatrix2D B = A.viewStrides(rowStride, colStride);
        for (int r = 0; r < B.rows(); r++) {
            for (int c = 0; c < B.columns(); c++) {
                assertEquals(A.getQuick(r * rowStride, c * colStride), B.getQuick(r, c));
            }
        }
    }

    public void testZMultLongMatrix1DLongMatrix1DLongLongBoolean() {
        LongMatrix1D y = new DenseLongMatrix1D(A.columns());
        for (int i = 0; i < y.size(); i++) {
            y.setQuick(i, rand.nextLong() % A.rows());
        }
        long alpha = 3;
        long beta = 5;
        LongMatrix1D z = LongFactory1D.dense.random(A.rows());
        z.assign(LongFunctions.mod(A.rows()));
        long[] expected = z.toArray();
        z = A.zMult(y, z, alpha, beta, false);
        for (int r = 0; r < A.rows(); r++) {
            long s = 0;
            for (int c = 0; c < A.columns(); c++) {
                s += A.getQuick(r, c) * y.getQuick(c);
            }
            expected[r] = s * alpha + expected[r] * beta;
        }

        for (int r = 0; r < A.rows(); r++) {
            assertEquals(expected[r], z.getQuick(r));
        }
        //---
        z = null;
        z = A.zMult(y, z, alpha, beta, false);
        expected = new long[A.rows()];
        for (int r = 0; r < A.rows(); r++) {
            long s = 0;
            for (int c = 0; c < A.columns(); c++) {
                s += A.getQuick(r, c) * y.getQuick(c);
            }
            expected[r] = s * alpha;
        }
        for (int r = 0; r < A.rows(); r++) {
            assertEquals(expected[r], z.getQuick(r));
        }

        //transpose
        y = new DenseLongMatrix1D(A.rows());
        for (int i = 0; i < y.size(); i++) {
            y.setQuick(i, rand.nextLong() % A.rows());
        }
        z = LongFactory1D.dense.random(A.columns());
        z.assign(LongFunctions.mod(A.rows()));
        expected = z.toArray();
        z = A.zMult(y, z, alpha, beta, true);
        for (int r = 0; r < A.columns(); r++) {
            long s = 0;
            for (int c = 0; c < A.rows(); c++) {
                s += A.getQuick(c, r) * y.getQuick(c);
            }
            expected[r] = s * alpha + expected[r] * beta;
        }
        for (int r = 0; r < A.columns(); r++) {
            assertEquals(expected[r], z.getQuick(r));
        }
        //---
        z = null;
        z = A.zMult(y, z, alpha, beta, true);
        expected = new long[A.columns()];
        for (int r = 0; r < A.columns(); r++) {
            long s = 0;
            for (int c = 0; c < A.rows(); c++) {
                s += A.getQuick(c, r) * y.getQuick(c);
            }
            expected[r] = s * alpha;
        }
        for (int r = 0; r < A.columns(); r++) {
            assertEquals(expected[r], z.getQuick(r));
        }
    }

    public void testZMultLongMatrix2DLongMatrix2DLongLongBooleanBoolean() {
        long alpha = 3;
        long beta = 5;
        LongMatrix2D C = LongFactory2D.dense.random(A.rows(), A.rows());
        C.assign(LongFunctions.mod(A.rows()));
        long[][] expected = C.toArray();
        C = A.zMult(Bt, C, alpha, beta, false, false);
        for (int j = 0; j < A.rows(); j++) {
            for (int i = 0; i < A.rows(); i++) {
                long s = 0;
                for (int k = 0; k < A.columns(); k++) {
                    s += A.getQuick(i, k) * Bt.getQuick(k, j);
                }
                expected[i][j] = s * alpha + expected[i][j] * beta;
            }
        }
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.rows(); c++) {
                assertEquals(expected[r][c], C.getQuick(r, c));
            }
        }

        //---
        C = null;
        C = A.zMult(Bt, C, alpha, beta, false, false);
        expected = new long[A.rows()][A.rows()];
        for (int j = 0; j < A.rows(); j++) {
            for (int i = 0; i < A.rows(); i++) {
                long s = 0;
                for (int k = 0; k < A.columns(); k++) {
                    s += A.getQuick(i, k) * Bt.getQuick(k, j);
                }
                expected[i][j] = s * alpha;
            }
        }
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.rows(); c++) {
                assertEquals(expected[r][c], C.getQuick(r, c));
            }
        }

        //transposeA
        C = LongFactory2D.dense.random(A.columns(), A.columns());
        C.assign(LongFunctions.mod(A.rows()));
        expected = C.toArray();
        C = A.zMult(B, C, alpha, beta, true, false);
        for (int j = 0; j < A.columns(); j++) {
            for (int i = 0; i < A.columns(); i++) {
                long s = 0;
                for (int k = 0; k < A.rows(); k++) {
                    s += A.getQuick(k, i) * B.getQuick(k, j);
                }
                expected[i][j] = s * alpha + expected[i][j] * beta;
            }
        }
        for (int r = 0; r < A.columns(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(expected[r][c], C.getQuick(r, c));
            }
        }
        //---
        C = null;
        C = A.zMult(B, C, alpha, beta, true, false);
        expected = new long[A.columns()][A.columns()];
        for (int j = 0; j < A.columns(); j++) {
            for (int i = 0; i < A.columns(); i++) {
                long s = 0;
                for (int k = 0; k < A.rows(); k++) {
                    s += A.getQuick(k, i) * B.getQuick(k, j);
                }
                expected[i][j] = s * alpha;
            }
        }
        for (int r = 0; r < A.columns(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(expected[r][c], C.getQuick(r, c));
            }
        }

        //transposeB
        C = LongFactory2D.dense.random(A.rows(), A.rows());
        C.assign(LongFunctions.mod(A.rows()));
        expected = C.toArray();
        C = A.zMult(B, C, alpha, beta, false, true);
        for (int j = 0; j < A.rows(); j++) {
            for (int i = 0; i < A.rows(); i++) {
                long s = 0;
                for (int k = 0; k < A.columns(); k++) {
                    s += A.getQuick(i, k) * B.getQuick(j, k);
                }
                expected[i][j] = s * alpha + expected[i][j] * beta;
            }
        }
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.rows(); c++) {
                assertEquals(expected[r][c], C.getQuick(r, c));
            }
        }
        //---
        C = null;
        C = A.zMult(B, C, alpha, beta, false, true);
        expected = new long[A.rows()][A.rows()];
        for (int j = 0; j < A.rows(); j++) {
            for (int i = 0; i < A.rows(); i++) {
                long s = 0;
                for (int k = 0; k < A.columns(); k++) {
                    s += A.getQuick(i, k) * B.getQuick(j, k);
                }
                expected[i][j] = s * alpha;
            }
        }
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.rows(); c++) {
                assertEquals(expected[r][c], C.getQuick(r, c));
            }
        }
        //transposeA and transposeB
        C = LongFactory2D.dense.random(A.columns(), A.columns());
        C.assign(LongFunctions.mod(A.rows()));
        expected = C.toArray();
        C = A.zMult(Bt, C, alpha, beta, true, true);
        for (int j = 0; j < A.columns(); j++) {
            for (int i = 0; i < A.columns(); i++) {
                long s = 0;
                for (int k = 0; k < A.rows(); k++) {
                    s += A.getQuick(k, i) * Bt.getQuick(j, k);
                }
                expected[i][j] = s * alpha + expected[i][j] * beta;
            }
        }
        for (int r = 0; r < A.columns(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(expected[r][c], C.getQuick(r, c));
            }
        }
        //---
        C = null;
        C = A.zMult(Bt, C, alpha, beta, true, true);
        expected = new long[A.columns()][A.columns()];
        for (int j = 0; j < A.columns(); j++) {
            for (int i = 0; i < A.columns(); i++) {
                long s = 0;
                for (int k = 0; k < A.rows(); k++) {
                    s += A.getQuick(k, i) * Bt.getQuick(j, k);
                }
                expected[i][j] = s * alpha;
            }
        }
        for (int r = 0; r < A.columns(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(expected[r][c], C.getQuick(r, c));
            }
        }

    }

    public void testZSum() {
        long sum = A.zSum();
        long expected = 0;
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                expected += A.getQuick(r, c);
            }
        }
        assertEquals(expected, sum);
    }

}
