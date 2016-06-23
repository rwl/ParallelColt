package cern.colt.matrix.tlong;

import java.util.Random;

import junit.framework.TestCase;
import cern.colt.function.tlong.LongProcedure;
import cern.colt.list.tint.IntArrayList;
import cern.colt.list.tlong.LongArrayList;
import cern.jet.math.tlong.LongFunctions;
import edu.emory.mathcs.utils.pc.ConcurrencyUtils;

public abstract class LongMatrix3DTest extends TestCase {
    /**
     * Matrix to test
     */
    protected LongMatrix3D A;

    /**
     * Matrix of the same size as A
     */
    protected LongMatrix3D B;

    protected int NSLICES = 5;

    protected int NROWS = 13;

    protected int NCOLUMNS = 17;

    protected Random rand = new Random(0);

    /**
     * Constructor for LongMatrix2DTest
     */
    public LongMatrix3DTest(String arg0) {
        super(arg0);
    }

    protected void setUp() throws Exception {
        createMatrices();
        populateMatrices();
    }

    protected abstract void createMatrices() throws Exception;

    protected void populateMatrices() {
        ConcurrencyUtils.setThreadsBeginN_3D(1);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    A.setQuick(s, r, c, Math.max(1, rand.nextLong() % A.rows()));
                }
            }
        }

        for (int s = 0; s < B.slices(); s++) {
            for (int r = 0; r < B.rows(); r++) {
                for (int c = 0; c < B.columns(); c++) {
                    B.setQuick(s, r, c, Math.max(1, rand.nextLong() % B.rows()));
                }
            }
        }
    }

    protected void tearDown() throws Exception {
        A = B = null;
    }

    public void testAggregateLongLongFunctionLongFunction() {
        long expected = 0;
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    long elem = A.getQuick(s, r, c);
                    expected += elem * elem;
                }
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
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    long elem = A.getQuick(s, r, c);
                    if (Math.abs(elem) > 0.2) {
                        expected += elem * elem;
                    }
                }
            }
        }

        long result = A.aggregate(LongFunctions.plus, LongFunctions.square, procedure);
        assertEquals(expected, result);
    }

    public void testAggregateLongLongFunctionLongFunctionIntArrayListIntArrayListIntArrayList() {
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    sliceList.add(s);
                    rowList.add(r);
                    columnList.add(c);
                }
            }
        }
        long expected = 0;
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    long elem = A.getQuick(s, r, c);
                    expected += elem * elem;
                }
            }
        }
        long result = A.aggregate(LongFunctions.plus, LongFunctions.square, sliceList, rowList, columnList);
        assertEquals(expected, result);
    }

    public void testAggregateLongMatrix2DLongLongFunctionLongLongFunction() {
        long expected = 0;
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    long elemA = A.getQuick(s, r, c);
                    long elemB = B.getQuick(s, r, c);
                    expected += elemA * elemB;
                }
            }
        }
        long result = A.aggregate(B, LongFunctions.plus, LongFunctions.mult);
        assertEquals(expected, result);
    }

    public void testAssignLong() {
        long value = rand.nextLong();
        A.assign(value);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++)
                    assertEquals(value, A.getQuick(s, r, c));
            }
        }
    }

    public void testAssignLongArray() {
        long[] expected = new long[(int) A.size()];
        for (int i = 0; i < A.size(); i++) {
            expected[i] = rand.nextLong();
        }
        A.assign(expected);
        int idx = 0;
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(expected[idx++], A.getQuick(s, r, c));
                }
            }
        }
    }

    public void testAssignLongArrayArrayArray() {
        long[][][] expected = new long[A.slices()][A.rows()][A.columns()];
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    expected[s][r][c] = rand.nextLong();
                }
            }
        }
        A.assign(expected);
        for (int s = 0; s < A.slices(); s++) {
            assertTrue(A.rows() == expected[s].length);
            for (int r = 0; r < A.rows(); r++) {
                assertTrue(A.columns() == expected[s][r].length);
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(expected[s][r][c], A.getQuick(s, r, c));
                }
            }
        }
    }

    public void testAssignLongFunction() {
        LongMatrix3D Acopy = A.copy();
        A.assign(LongFunctions.neg);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    long expected = -Acopy.getQuick(s, r, c);
                    assertEquals(expected, A.getQuick(s, r, c));
                }
            }
        }
    }

    public void testAssignLongMatrix3D() {
        A.assign(B);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++)
                    assertEquals(B.getQuick(s, r, c), A.getQuick(s, r, c));
            }
        }
    }

    public void testAssignLongMatrix3DLongLongFunction() {
        LongMatrix3D Acopy = A.copy();
        A.assign(B, LongFunctions.plus);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(Acopy.getQuick(s, r, c) + B.getQuick(s, r, c), A.getQuick(s, r, c));
                }
            }
        }
    }

    public void testAssignLongMatrix3DLongLongFunctionIntArrayListIntArrayListIntArrayList() {
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    sliceList.add(s);
                    rowList.add(r);
                    columnList.add(c);
                }
            }
        }
        LongMatrix3D Acopy = A.copy();
        A.assign(B, LongFunctions.plus, sliceList, rowList, columnList);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(Acopy.getQuick(s, r, c) + B.getQuick(s, r, c), A.getQuick(s, r, c));
                }
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
        LongMatrix3D Acopy = A.copy();
        A.assign(procedure, -1);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    if (Math.abs(Acopy.getQuick(s, r, c)) > 1) {
                        assertEquals(-1, A.getQuick(s, r, c));
                    } else {
                        assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c));
                    }
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
        LongMatrix3D Acopy = A.copy();
        A.assign(procedure, LongFunctions.neg);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    if (Math.abs(Acopy.getQuick(s, r, c)) > 1) {
                        assertEquals(-Acopy.getQuick(s, r, c), A.getQuick(s, r, c));
                    } else {
                        assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c));
                    }
                }
            }
        }
    }

    public void testCardinality() {
        int card = A.cardinality();
        int expected = 0;
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    if (A.getQuick(s, r, c) != 0)
                        expected++;
                }
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

    public void testMaxLocation() {
        A.assign(0);
        A.setQuick(A.slices() / 3, A.rows() / 3, A.columns() / 3, 7);
        A.setQuick(A.slices() / 3, A.rows() / 2, A.columns() / 2, 1);
        long[] maxAndLoc = A.getMaxLocation();
        assertEquals(7, maxAndLoc[0]);
        assertEquals(A.slices() / 3, (int) maxAndLoc[1]);
        assertEquals(A.rows() / 3, (int) maxAndLoc[2]);
        assertEquals(A.columns() / 3, (int) maxAndLoc[3]);
    }

    public void testMinLocation() {
        A.assign(0);
        A.setQuick(A.slices() / 3, A.rows() / 3, A.columns() / 3, -7);
        A.setQuick(A.slices() / 3, A.rows() / 2, A.columns() / 2, -1);
        long[] minAndLoc = A.getMinLocation();
        assertEquals(-7, minAndLoc[0]);
        assertEquals(A.slices() / 3, (int) minAndLoc[1]);
        assertEquals(A.rows() / 3, (int) minAndLoc[2]);
        assertEquals(A.columns() / 3, (int) minAndLoc[3]);
    }

    public void testGetNegativeValues() {
        A.assign(0);
        A.setQuick(A.slices() / 3, A.rows() / 3, A.columns() / 3, -7);
        A.setQuick(A.slices() / 2, A.rows() / 2, A.columns() / 2, -1);
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        LongArrayList valueList = new LongArrayList();
        A.getNegativeValues(sliceList, rowList, columnList, valueList);
        assertEquals(2, sliceList.size());
        assertEquals(2, rowList.size());
        assertEquals(2, columnList.size());
        assertEquals(2, valueList.size());
        assertTrue(sliceList.contains(A.slices() / 3));
        assertTrue(sliceList.contains(A.slices() / 2));
        assertTrue(rowList.contains(A.rows() / 3));
        assertTrue(rowList.contains(A.rows() / 2));
        assertTrue(columnList.contains(A.columns() / 3));
        assertTrue(columnList.contains(A.columns() / 2));
        assertTrue(valueList.contains(-7));
        assertTrue(valueList.contains(-1));
    }

    public void testGetNonZeros() {
        A.assign(0);
        A.setQuick(A.slices() / 3, A.rows() / 3, A.columns() / 3, 7);
        A.setQuick(A.slices() / 2, A.rows() / 2, A.columns() / 2, 1);
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        LongArrayList valueList = new LongArrayList();
        A.getNonZeros(sliceList, rowList, columnList, valueList);
        assertEquals(2, sliceList.size());
        assertEquals(2, rowList.size());
        assertEquals(2, columnList.size());
        assertEquals(2, valueList.size());
        assertTrue(sliceList.contains(A.slices() / 3));
        assertTrue(sliceList.contains(A.slices() / 2));
        assertTrue(rowList.contains(A.rows() / 3));
        assertTrue(rowList.contains(A.rows() / 2));
        assertTrue(columnList.contains(A.columns() / 3));
        assertTrue(columnList.contains(A.columns() / 2));
        assertTrue(valueList.contains(7));
        assertTrue(valueList.contains(1));
    }

    public void testGetPositiveValues() {
        A.assign(0);
        A.setQuick(A.slices() / 3, A.rows() / 3, A.columns() / 3, 7);
        A.setQuick(A.slices() / 2, A.rows() / 2, A.columns() / 2, 1);
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        LongArrayList valueList = new LongArrayList();
        A.getPositiveValues(sliceList, rowList, columnList, valueList);
        assertEquals(2, sliceList.size());
        assertEquals(2, rowList.size());
        assertEquals(2, columnList.size());
        assertEquals(2, valueList.size());
        assertTrue(sliceList.contains(A.slices() / 3));
        assertTrue(sliceList.contains(A.slices() / 2));
        assertTrue(rowList.contains(A.rows() / 3));
        assertTrue(rowList.contains(A.rows() / 2));
        assertTrue(columnList.contains(A.columns() / 3));
        assertTrue(columnList.contains(A.columns() / 2));
        assertTrue(valueList.contains(7));
        assertTrue(valueList.contains(1));
    }

    public void testToArray() {
        long[][][] array = A.toArray();
        for (int s = 0; s < A.slices(); s++) {
            assertTrue(A.rows() == array[s].length);
            for (int r = 0; r < A.rows(); r++) {
                assertTrue(A.columns() == array[s][r].length);
                for (int c = 0; c < A.columns(); c++)
                    assertEquals(0, Math.abs(array[s][r][c] - A.getQuick(s, r, c)));
            }
        }
    }

    public void testVectorize() {
        LongMatrix1D Avec = A.vectorize();
        int idx = 0;
        for (int s = 0; s < A.slices(); s++) {
            for (int c = 0; c < A.columns(); c++) {
                for (int r = 0; r < A.rows(); r++) {
                    assertEquals(A.getQuick(s, r, c), Avec.getQuick(idx++));
                }
            }
        }
    }

    public void testViewColumn() {
        LongMatrix2D B = A.viewColumn(A.columns() / 2);
        assertEquals(A.slices(), B.rows());
        assertEquals(A.rows(), B.columns());
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                assertEquals(A.getQuick(s, r, A.columns() / 2), B.getQuick(s, r));
            }
        }
    }

    public void testViewColumnFlip() {
        LongMatrix3D B = A.viewColumnFlip();
        assertEquals(A.size(), B.size());
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(A.getQuick(s, r, A.columns() - 1 - c), B.getQuick(s, r, c));
                }
            }
        }
    }

    public void testViewDice() {
        LongMatrix3D B = A.viewDice(2, 1, 0);
        assertEquals(A.slices(), B.columns());
        assertEquals(A.rows(), B.rows());
        assertEquals(A.columns(), B.slices());
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(A.getQuick(s, r, c), B.getQuick(c, r, s));
                }
            }
        }
    }

    public void testViewPart() {
        LongMatrix3D B = A.viewPart(A.slices() / 2, A.rows() / 2, A.columns() / 2, A.slices() / 3, A.rows() / 3, A
                .columns() / 3);
        for (int s = 0; s < A.slices() / 3; s++) {
            for (int r = 0; r < A.rows() / 3; r++) {
                for (int c = 0; c < A.columns() / 3; c++) {
                    assertEquals(A.getQuick(A.slices() / 2 + s, A.rows() / 2 + r, A.columns() / 2 + c), B.getQuick(s,
                            r, c));
                }
            }
        }
    }

    public void testViewRow() {
        LongMatrix2D B = A.viewRow(A.rows() / 2);
        assertEquals(A.slices(), B.rows());
        assertEquals(A.columns(), B.columns());
        for (int s = 0; s < A.slices(); s++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(s, A.rows() / 2, c), B.getQuick(s, c));
            }
        }
    }

    public void testViewRowFlip() {
        LongMatrix3D B = A.viewRowFlip();
        assertEquals(A.size(), B.size());
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(A.getQuick(s, A.rows() - 1 - r, c), B.getQuick(s, r, c));
                }
            }
        }
    }

    public void testViewSelectionLongMatrix2DProcedure() {
        A.assign(0);
        final long value = 2;
        A.setQuick(A.slices() / 2, A.rows() / 4, 0, value);
        LongMatrix3D B = A.viewSelection(new LongMatrix2DProcedure() {
            public boolean apply(LongMatrix2D element) {
                if (Math.abs(element.getQuick(A.rows() / 4, 0) - value) == 0) {
                    return true;
                } else {
                    return false;
                }

            }
        });
        assertEquals(1, B.slices());
        assertEquals(A.rows(), B.rows());
        assertEquals(A.columns(), B.columns());
        assertEquals(A.getQuick(A.slices() / 2, A.rows() / 4, 0), B.getQuick(0, A.rows() / 4, 0));
    }

    public void testViewSelectionIntArrayIntArrayIntArray() {
        int[] sliceIndexes = new int[] { A.slices() / 2, A.slices() / 3 };
        int[] rowIndexes = new int[] { A.rows() / 6, A.rows() / 5, A.rows() / 4, A.rows() / 3, A.rows() / 2 };
        int[] colIndexes = new int[] { A.columns() / 6, A.columns() / 5, A.columns() / 4, A.columns() / 3,
                A.columns() / 2, A.columns() - 1 };
        LongMatrix3D B = A.viewSelection(sliceIndexes, rowIndexes, colIndexes);
        assertEquals(sliceIndexes.length, B.slices());
        assertEquals(rowIndexes.length, B.rows());
        assertEquals(colIndexes.length, B.columns());
        for (int s = 0; s < sliceIndexes.length; s++) {
            for (int r = 0; r < rowIndexes.length; r++) {
                for (int c = 0; c < colIndexes.length; c++) {
                    assertEquals(A.getQuick(sliceIndexes[s], rowIndexes[r], colIndexes[c]), B.getQuick(s, r, c));
                }
            }
        }
    }

    public void testViewSlice() {
        LongMatrix2D B = A.viewSlice(A.slices() / 2);
        assertEquals(A.rows(), B.rows());
        assertEquals(A.columns(), B.columns());
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(A.slices() / 2, r, c), B.getQuick(r, c));
            }
        }
    }

    public void testViewSliceFlip() {
        LongMatrix3D B = A.viewSliceFlip();
        assertEquals(A.size(), B.size());
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(A.getQuick(A.slices() - 1 - s, r, c), B.getQuick(s, r, c));
                }
            }
        }
    }

    public void testViewSorted() {
        LongMatrix3D B = A.viewSorted(1, 1);
        for (int s = 0; s < A.slices() - 1; s++) {
            assertTrue(B.getQuick(s + 1, 1, 1) >= B.getQuick(s, 1, 1));
        }
    }

    public void testViewStrides() {
        int sliceStride = 2;
        int rowStride = 2;
        int colStride = 2;
        LongMatrix3D B = A.viewStrides(sliceStride, rowStride, colStride);
        for (int s = 0; s < B.slices(); s++) {
            for (int r = 0; r < B.rows(); r++) {
                for (int c = 0; c < B.columns(); c++) {
                    assertEquals(A.getQuick(s * sliceStride, r * rowStride, c * colStride), B.getQuick(s, r, c));
                }
            }
        }
    }

    public void testZSum() {
        long sum = A.zSum();
        long expected = 0;
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    expected += A.getQuick(s, r, c);
                }
            }
        }
        assertEquals(expected, sum);
    }

}
