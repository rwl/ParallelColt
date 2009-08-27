package cern.colt.matrix.tint;

import java.util.Random;

import junit.framework.TestCase;
import cern.colt.function.tint.IntProcedure;
import cern.colt.list.tint.IntArrayList;
import cern.jet.math.tint.IntFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public abstract class IntMatrix3DTest extends TestCase {
    /**
     * Matrix to test
     */
    protected IntMatrix3D A;

    /**
     * Matrix of the same size as A
     */
    protected IntMatrix3D B;

    protected int NSLICES = 5;

    protected int NROWS = 13;

    protected int NCOLUMNS = 17;

    protected Random rand = new Random(0);

    /**
     * Constructor for IntMatrix2DTest
     */
    public IntMatrix3DTest(String arg0) {
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
                    A.setQuick(s, r, c, Math.max(1, rand.nextInt() % A.rows()));
                }
            }
        }

        for (int s = 0; s < B.slices(); s++) {
            for (int r = 0; r < B.rows(); r++) {
                for (int c = 0; c < B.columns(); c++) {
                    B.setQuick(s, r, c, Math.max(1, rand.nextInt() % B.rows()));
                }
            }
        }
    }

    protected void tearDown() throws Exception {
        A = B = null;
    }

    public void testAggregateIntIntFunctionIntFunction() {
        int expected = 0;
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    int elem = A.getQuick(s, r, c);
                    expected += elem * elem;
                }
            }
        }
        int result = A.aggregate(IntFunctions.plus, IntFunctions.square);
        assertEquals(expected, result);
    }

    public void testAggregateIntIntFunctionIntFunctionIntProcedure() {
        IntProcedure procedure = new IntProcedure() {
            public boolean apply(int element) {
                if (Math.abs(element) > 0.2) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        int expected = 0;
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    int elem = A.getQuick(s, r, c);
                    if (Math.abs(elem) > 0.2) {
                        expected += elem * elem;
                    }
                }
            }
        }

        int result = A.aggregate(IntFunctions.plus, IntFunctions.square, procedure);
        assertEquals(expected, result);
    }

    public void testAggregateIntIntFunctionIntFunctionIntArrayListIntArrayListIntArrayList() {
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
        int expected = 0;
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    int elem = A.getQuick(s, r, c);
                    expected += elem * elem;
                }
            }
        }
        int result = A.aggregate(IntFunctions.plus, IntFunctions.square, sliceList, rowList, columnList);
        assertEquals(expected, result);
    }

    public void testAggregateIntMatrix2DIntIntFunctionIntIntFunction() {
        int expected = 0;
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    int elemA = A.getQuick(s, r, c);
                    int elemB = B.getQuick(s, r, c);
                    expected += elemA * elemB;
                }
            }
        }
        int result = A.aggregate(B, IntFunctions.plus, IntFunctions.mult);
        assertEquals(expected, result);
    }

    public void testAssignInt() {
        int value = rand.nextInt();
        A.assign(value);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++)
                    assertEquals(value, A.getQuick(s, r, c));
            }
        }
    }

    public void testAssignIntArray() {
        int[] expected = new int[(int) A.size()];
        for (int i = 0; i < A.size(); i++) {
            expected[i] = rand.nextInt();
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

    public void testAssignIntArrayArrayArray() {
        int[][][] expected = new int[A.slices()][A.rows()][A.columns()];
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    expected[s][r][c] = rand.nextInt();
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

    public void testAssignIntFunction() {
        IntMatrix3D Acopy = A.copy();
        A.assign(IntFunctions.neg);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    int expected = -Acopy.getQuick(s, r, c);
                    assertEquals(expected, A.getQuick(s, r, c));
                }
            }
        }
    }

    public void testAssignIntMatrix3D() {
        A.assign(B);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++)
                    assertEquals(B.getQuick(s, r, c), A.getQuick(s, r, c));
            }
        }
    }

    public void testAssignIntMatrix3DIntIntFunction() {
        IntMatrix3D Acopy = A.copy();
        A.assign(B, IntFunctions.plus);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(Acopy.getQuick(s, r, c) + B.getQuick(s, r, c), A.getQuick(s, r, c));
                }
            }
        }
    }

    public void testAssignIntMatrix3DIntIntFunctionIntArrayListIntArrayListIntArrayList() {
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
        IntMatrix3D Acopy = A.copy();
        A.assign(B, IntFunctions.plus, sliceList, rowList, columnList);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(Acopy.getQuick(s, r, c) + B.getQuick(s, r, c), A.getQuick(s, r, c));
                }
            }
        }
    }

    public void testAssignIntProcedureInt() {
        IntProcedure procedure = new IntProcedure() {
            public boolean apply(int element) {
                if (Math.abs(element) > 1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        IntMatrix3D Acopy = A.copy();
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

    public void testAssignIntProcedureIntFunction() {
        IntProcedure procedure = new IntProcedure() {
            public boolean apply(int element) {
                if (Math.abs(element) > 1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        IntMatrix3D Acopy = A.copy();
        A.assign(procedure, IntFunctions.neg);
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

    public void testEqualsInt() {
        int value = 1;
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
        int[] maxAndLoc = A.getMaxLocation();
        assertEquals(7, maxAndLoc[0]);
        assertEquals(A.slices() / 3, (int) maxAndLoc[1]);
        assertEquals(A.rows() / 3, (int) maxAndLoc[2]);
        assertEquals(A.columns() / 3, (int) maxAndLoc[3]);
    }

    public void testMinLocation() {
        A.assign(0);
        A.setQuick(A.slices() / 3, A.rows() / 3, A.columns() / 3, -7);
        A.setQuick(A.slices() / 3, A.rows() / 2, A.columns() / 2, -1);
        int[] minAndLoc = A.getMinLocation();
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
        IntArrayList valueList = new IntArrayList();
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
        IntArrayList valueList = new IntArrayList();
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
        IntArrayList valueList = new IntArrayList();
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
        int[][][] array = A.toArray();
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
        IntMatrix1D Avec = A.vectorize();
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
        IntMatrix2D B = A.viewColumn(A.columns() / 2);
        assertEquals(A.slices(), B.rows());
        assertEquals(A.rows(), B.columns());
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                assertEquals(A.getQuick(s, r, A.columns() / 2), B.getQuick(s, r));
            }
        }
    }

    public void testViewColumnFlip() {
        IntMatrix3D B = A.viewColumnFlip();
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
        IntMatrix3D B = A.viewDice(2, 1, 0);
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
        IntMatrix3D B = A.viewPart(A.slices() / 2, A.rows() / 2, A.columns() / 2, A.slices() / 3, A.rows() / 3, A
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
        IntMatrix2D B = A.viewRow(A.rows() / 2);
        assertEquals(A.slices(), B.rows());
        assertEquals(A.columns(), B.columns());
        for (int s = 0; s < A.slices(); s++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(s, A.rows() / 2, c), B.getQuick(s, c));
            }
        }
    }

    public void testViewRowFlip() {
        IntMatrix3D B = A.viewRowFlip();
        assertEquals(A.size(), B.size());
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(A.getQuick(s, A.rows() - 1 - r, c), B.getQuick(s, r, c));
                }
            }
        }
    }

    public void testViewSelectionIntMatrix2DProcedure() {
        A.assign(0);
        final int value = 2;
        A.setQuick(A.slices() / 2, A.rows() / 4, 0, value);
        IntMatrix3D B = A.viewSelection(new IntMatrix2DProcedure() {
            public boolean apply(IntMatrix2D element) {
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
        IntMatrix3D B = A.viewSelection(sliceIndexes, rowIndexes, colIndexes);
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
        IntMatrix2D B = A.viewSlice(A.slices() / 2);
        assertEquals(A.rows(), B.rows());
        assertEquals(A.columns(), B.columns());
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(A.slices() / 2, r, c), B.getQuick(r, c));
            }
        }
    }

    public void testViewSliceFlip() {
        IntMatrix3D B = A.viewSliceFlip();
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
        IntMatrix3D B = A.viewSorted(1, 1);
        for (int s = 0; s < A.slices() - 1; s++) {
            assertTrue(B.getQuick(s + 1, 1, 1) >= B.getQuick(s, 1, 1));
        }
    }

    public void testViewStrides() {
        int sliceStride = 2;
        int rowStride = 2;
        int colStride = 2;
        IntMatrix3D B = A.viewStrides(sliceStride, rowStride, colStride);
        for (int s = 0; s < B.slices(); s++) {
            for (int r = 0; r < B.rows(); r++) {
                for (int c = 0; c < B.columns(); c++) {
                    assertEquals(A.getQuick(s * sliceStride, r * rowStride, c * colStride), B.getQuick(s, r, c));
                }
            }
        }
    }

    public void testZSum() {
        int sum = A.zSum();
        int expected = 0;
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
