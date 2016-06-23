package cern.colt.matrix.tlong.impl;

import cern.colt.list.tint.IntArrayList;
import cern.colt.list.tlong.LongArrayList;
import cern.colt.matrix.tlong.LongMatrix1D;
import cern.colt.matrix.tlong.LongMatrix1DProcedure;
import cern.colt.matrix.tlong.LongMatrix2D;
import cern.colt.matrix.tlong.LongMatrix2DTest;
import cern.jet.math.tlong.LongFunctions;
import edu.emory.mathcs.utils.pc.ConcurrencyUtils;

public class DiagonalLongMatrix2DTest extends LongMatrix2DTest {

    protected int DLENGTH;

    protected int DINDEX;

    public DiagonalLongMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        DINDEX = 3;
        A = new DiagonalLongMatrix2D(NROWS, NCOLUMNS, DINDEX);
        B = new DiagonalLongMatrix2D(NROWS, NCOLUMNS, DINDEX);
        Bt = new DiagonalLongMatrix2D(NCOLUMNS, NROWS, -DINDEX);
        DLENGTH = ((DiagonalLongMatrix2D) A).diagonalLength();

    }

    protected void populateMatrices() {
        ConcurrencyUtils.setThreadsBeginN_2D(1);
        if (DINDEX >= 0) {
            for (int r = 0; r < DLENGTH; r++) {
                A.setQuick(r, r + DINDEX, Math.max(1, rand.nextLong() % A.rows()));
            }

            for (int r = 0; r < DLENGTH; r++) {
                B.setQuick(r, r + DINDEX, Math.max(1, rand.nextLong() % A.rows()));
            }

            for (int r = 0; r < DLENGTH; r++) {
                Bt.setQuick(r - DINDEX, r, Math.max(1, rand.nextLong() % A.rows()));
            }

        } else {
            for (int r = 0; r < DLENGTH; r++) {
                A.setQuick(r - DINDEX, r, Math.max(1, rand.nextLong() % A.rows()));
            }

            for (int r = 0; r < DLENGTH; r++) {
                B.setQuick(r - DINDEX, r, Math.max(1, rand.nextLong() % A.rows()));
            }
            for (int r = 0; r < DLENGTH; r++) {
                Bt.setQuick(r, r + DINDEX, Math.max(1, rand.nextLong() % A.rows()));
            }

        }
    }

    public void testAssignLong() {
        long value = Math.max(1, rand.nextLong() % A.rows());
        A.assign(value);
        if (DINDEX >= 0) {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(value, A.getQuick(r, r + DINDEX));
            }
        } else {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(value, A.getQuick(r - DINDEX, r));
            }
        }
    }

    public void testAssignLongArrayArray() {
        long[][] expected = new long[NROWS][NCOLUMNS];
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                expected[r][c] = Math.max(1, rand.nextLong() % A.rows());
            }
        }
        A.assign(expected);
        if (DINDEX >= 0) {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(expected[r][r + DINDEX], A.getQuick(r, r + DINDEX));
            }
        } else {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(expected[r - DINDEX][r], A.getQuick(r - DINDEX, r));
            }
        }
    }

    public void testAssignLongFunction() {
        LongMatrix2D Acopy = A.copy();
        A.assign(LongFunctions.neg);
        if (DINDEX >= 0) {
            for (int r = 0; r < DLENGTH; r++) {
                long expected = -Acopy.getQuick(r, r + DINDEX);
                assertEquals(expected, A.getQuick(r, r + DINDEX));
            }
        } else {
            for (int r = 0; r < DLENGTH; r++) {
                long expected = -Acopy.getQuick(r - DINDEX, r);
                assertEquals(expected, A.getQuick(r - DINDEX, r));
            }
        }
    }

    public void testAssignLongMatrix2DLongLongFunction() {
        LongMatrix2D Acopy = A.copy();
        A.assign(B, LongFunctions.div);
        if (DINDEX >= 0) {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(Acopy.getQuick(r, r + DINDEX) / B.getQuick(r, r + DINDEX), A.getQuick(r, r + DINDEX));
            }
        } else {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(Acopy.getQuick(r - DINDEX, r) / B.getQuick(r - DINDEX, r), A.getQuick(r - DINDEX, r));
            }
        }
    }

    public void testAssignLongMatrix2DLongLongFunctionIntArrayListIntArrayList() {
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        if (DINDEX >= 0) {
            for (int r = 0; r < DLENGTH; r++) {
                rowList.add(r);
                columnList.add(r + DINDEX);
            }
            LongMatrix2D Acopy = A.copy();
            A.assign(B, LongFunctions.div, rowList, columnList);
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(Acopy.getQuick(r, r + DINDEX) / B.getQuick(r, r + DINDEX), A.getQuick(r, r + DINDEX));
            }
        } else {
            for (int r = 0; r < DLENGTH; r++) {
                rowList.add(r - DINDEX);
                columnList.add(r);
            }
            LongMatrix2D Acopy = A.copy();
            A.assign(B, LongFunctions.div, rowList, columnList);
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(Acopy.getQuick(r - DINDEX, r) / B.getQuick(r - DINDEX, r), A.getQuick(r - DINDEX, r));
            }
        }
    }

    public void testCardinality() {
        int card = A.cardinality();
        assertEquals(DLENGTH, card);
    }

    public void testMaxLocation() {
        A.assign(0);
        if (DINDEX >= 0) {
            A.setQuick(NROWS / 3, NROWS / 3 + DINDEX, 7);
            A.setQuick(NROWS / 2, NROWS / 2 + DINDEX, 1);
            long[] maxAndLoc = A.getMaxLocation();
            assertEquals(7, maxAndLoc[0]);
            assertEquals(NROWS / 3, (int) maxAndLoc[1]);
            assertEquals(NROWS / 3 + DINDEX, (int) maxAndLoc[2]);
        } else {
            A.setQuick(NROWS / 3 - DINDEX, NROWS / 3, 7);
            A.setQuick(NROWS / 2 - DINDEX, NROWS / 2, 1);
            long[] maxAndLoc = A.getMaxLocation();
            assertEquals(7, maxAndLoc[0]);
            assertEquals(NROWS / 3 - DINDEX, (int) maxAndLoc[1]);
            assertEquals(NROWS / 3, (int) maxAndLoc[2]);
        }
    }

    public void testMinLocation() {
        A.assign(0);
        if (DINDEX >= 0) {
            A.setQuick(NROWS / 3, NROWS / 3 + DINDEX, -7);
            A.setQuick(NROWS / 2, NROWS / 2 + DINDEX, -1);
            long[] minAndLoc = A.getMinLocation();
            assertEquals(-7, minAndLoc[0]);
            assertEquals(NROWS / 3, (int) minAndLoc[1]);
            assertEquals(NROWS / 3 + DINDEX, (int) minAndLoc[2]);
        } else {
            A.setQuick(NROWS / 3 - DINDEX, NROWS / 3, -7);
            A.setQuick(NROWS / 2 - DINDEX, NROWS / 2, -1);
            long[] minAndLoc = A.getMinLocation();
            assertEquals(-7, minAndLoc[0]);
            assertEquals(NROWS / 3 - DINDEX, (int) minAndLoc[1]);
            assertEquals(NROWS / 3, (int) minAndLoc[2]);
        }
    }

    public void testGetNegativeValues() {
        A.assign(0);
        if (DINDEX >= 0) {
            A.setQuick(NROWS / 3, NROWS / 3 + DINDEX, -7);
            A.setQuick(NROWS / 2, NROWS / 2 + DINDEX, -1);
            IntArrayList rowList = new IntArrayList();
            IntArrayList columnList = new IntArrayList();
            LongArrayList valueList = new LongArrayList();
            A.getNegativeValues(rowList, columnList, valueList);
            assertEquals(2, rowList.size());
            assertEquals(2, columnList.size());
            assertEquals(2, valueList.size());
            assertTrue(rowList.contains(NROWS / 3));
            assertTrue(rowList.contains(NROWS / 2));
            assertTrue(columnList.contains(NROWS / 3 + DINDEX));
            assertTrue(columnList.contains(NROWS / 2 + DINDEX));
            assertTrue(valueList.contains(-7));
            assertTrue(valueList.contains(-1));
        } else {
            A.setQuick(NROWS / 3 - DINDEX, NROWS / 3, -7);
            A.setQuick(NROWS / 2 - DINDEX, NROWS / 2, -1);
            IntArrayList rowList = new IntArrayList();
            IntArrayList columnList = new IntArrayList();
            LongArrayList valueList = new LongArrayList();
            A.getNegativeValues(rowList, columnList, valueList);
            assertEquals(2, rowList.size());
            assertEquals(2, columnList.size());
            assertEquals(2, valueList.size());
            assertTrue(rowList.contains(NROWS / 3 - DINDEX));
            assertTrue(rowList.contains(NROWS / 2 - DINDEX));
            assertTrue(columnList.contains(NROWS / 3));
            assertTrue(columnList.contains(NROWS / 2));
            assertTrue(valueList.contains(-7));
            assertTrue(valueList.contains(-1));
        }
    }

    public void testGetNonZeros() {
        A.assign(0);
        if (DINDEX >= 0) {
            A.setQuick(NROWS / 3, NROWS / 3 + DINDEX, 7);
            A.setQuick(NROWS / 2, NROWS / 2 + DINDEX, 1);
            IntArrayList rowList = new IntArrayList();
            IntArrayList columnList = new IntArrayList();
            LongArrayList valueList = new LongArrayList();
            A.getNonZeros(rowList, columnList, valueList);
            assertEquals(2, rowList.size());
            assertEquals(2, columnList.size());
            assertEquals(2, valueList.size());
            assertTrue(rowList.contains(NROWS / 3));
            assertTrue(rowList.contains(NROWS / 2));
            assertTrue(columnList.contains(NROWS / 3 + DINDEX));
            assertTrue(columnList.contains(NROWS / 2 + DINDEX));
            assertTrue(valueList.contains(7));
            assertTrue(valueList.contains(1));
        } else {
            A.setQuick(NROWS / 3 - DINDEX, NROWS / 3, 7);
            A.setQuick(NROWS / 2 - DINDEX, NROWS / 2, 1);
            IntArrayList rowList = new IntArrayList();
            IntArrayList columnList = new IntArrayList();
            LongArrayList valueList = new LongArrayList();
            A.getNonZeros(rowList, columnList, valueList);
            assertEquals(2, rowList.size());
            assertEquals(2, columnList.size());
            assertEquals(2, valueList.size());
            assertTrue(rowList.contains(NROWS / 3 - DINDEX));
            assertTrue(rowList.contains(NROWS / 2 - DINDEX));
            assertTrue(columnList.contains(NROWS / 3));
            assertTrue(columnList.contains(NROWS / 2));
            assertTrue(valueList.contains(7));
            assertTrue(valueList.contains(1));
        }
    }

    public void testGetPositiveValues() {
        A.assign(0);
        if (DINDEX >= 0) {
            A.setQuick(NROWS / 3, NROWS / 3 + DINDEX, 7);
            A.setQuick(NROWS / 2, NROWS / 2 + DINDEX, 1);
            IntArrayList rowList = new IntArrayList();
            IntArrayList columnList = new IntArrayList();
            LongArrayList valueList = new LongArrayList();
            A.getPositiveValues(rowList, columnList, valueList);
            assertEquals(2, rowList.size());
            assertEquals(2, columnList.size());
            assertEquals(2, valueList.size());
            assertTrue(rowList.contains(NROWS / 3));
            assertTrue(rowList.contains(NROWS / 2));
            assertTrue(columnList.contains(NROWS / 3 + DINDEX));
            assertTrue(columnList.contains(NROWS / 2 + DINDEX));
            assertTrue(valueList.contains(7));
            assertTrue(valueList.contains(1));
        } else {
            A.setQuick(NROWS / 3 - DINDEX, NROWS / 3, 7);
            A.setQuick(NROWS / 2 - DINDEX, NROWS / 2, 1);
            IntArrayList rowList = new IntArrayList();
            IntArrayList columnList = new IntArrayList();
            LongArrayList valueList = new LongArrayList();
            A.getPositiveValues(rowList, columnList, valueList);
            assertEquals(2, rowList.size());
            assertEquals(2, columnList.size());
            assertEquals(2, valueList.size());
            assertTrue(rowList.contains(NROWS / 3 - DINDEX));
            assertTrue(rowList.contains(NROWS / 2 - DINDEX));
            assertTrue(columnList.contains(NROWS / 3));
            assertTrue(columnList.contains(NROWS / 2));
            assertTrue(valueList.contains(7));
            assertTrue(valueList.contains(1));
        }
    }

    public void testToArray() {
        long[][] array = A.toArray();
        assertTrue(NROWS == array.length);
        for (int r = 0; r < NROWS; r++) {
            assertTrue(NCOLUMNS == array[r].length);
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(array[r][c], A.getQuick(r, c));
            }
        }
    }

    public void testVectorize() {
        LongMatrix1D Avec = A.vectorize();
        int idx = 0;
        for (int c = 0; c < NCOLUMNS; c++) {
            for (int r = 0; r < NROWS; r++) {
                assertEquals(A.getQuick(r, c), Avec.getQuick(idx++));
            }
        }
    }

    public void testViewColumn() {
        LongMatrix1D col = A.viewColumn(NCOLUMNS / 2);
        assertEquals(NROWS, col.size());
        for (int r = 0; r < NROWS; r++) {
            assertEquals(A.getQuick(r, NCOLUMNS / 2), col.getQuick(r));
        }
    }

    public void testViewColumnFlip() {
        LongMatrix2D B = A.viewColumnFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, NCOLUMNS - 1 - c), B.getQuick(r, c));
            }
        }
    }

    public void testViewDice() {
        LongMatrix2D B = A.viewDice();
        assertEquals(NROWS, B.columns());
        assertEquals(NCOLUMNS, B.rows());
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(c, r));
            }
        }
    }

    public void testViewPart() {
        LongMatrix2D B = A.viewPart(NROWS / 2, NCOLUMNS / 2, NROWS / 3, NCOLUMNS / 3);
        assertEquals(NROWS / 3, B.rows());
        assertEquals(NCOLUMNS / 3, B.columns());
        for (int r = 0; r < NROWS / 3; r++) {
            for (int c = 0; c < NCOLUMNS / 3; c++) {
                assertEquals(A.getQuick(NROWS / 2 + r, NCOLUMNS / 2 + c), B.getQuick(r, c));
            }
        }
    }

    public void testViewRow() {
        LongMatrix1D B = A.viewRow(NROWS / 2);
        assertEquals(NCOLUMNS, B.size());
        for (int r = 0; r < NCOLUMNS; r++) {
            assertEquals(A.getQuick(NROWS / 2, r), B.getQuick(r));
        }
    }

    public void testViewRowFlip() {
        LongMatrix2D B = A.viewRowFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(NROWS - 1 - r, c), B.getQuick(r, c));
            }
        }
    }

    public void testViewSelectionLongMatrix1DProcedure() {
        final long value = 2;
        A.assign(0);
        if (DINDEX >= 0) {
            A.setQuick(NROWS / 4, NROWS / 4 + DINDEX, value);
            A.setQuick(NROWS / 2, NROWS / 2 + DINDEX, value);
            LongMatrix2D B = A.viewSelection(new LongMatrix1DProcedure() {
                public boolean apply(LongMatrix1D element) {
                    if (element.getQuick(NROWS / 4 + DINDEX) == value) {
                        return true;
                    } else {
                        return false;
                    }
                }
            });
            assertEquals(1, B.rows());
            assertEquals(NCOLUMNS, B.columns());
            assertEquals(A.getQuick(NROWS / 4, NROWS / 4 + DINDEX), B.getQuick(0, NROWS / 4 + DINDEX));
        } else {
            A.setQuick(NROWS / 4 - DINDEX, NROWS / 4, value);
            A.setQuick(NROWS / 2 - DINDEX, NROWS / 2, value);
            LongMatrix2D B = A.viewSelection(new LongMatrix1DProcedure() {
                public boolean apply(LongMatrix1D element) {
                    if (element.getQuick(NROWS / 4) == value) {
                        return true;
                    } else {
                        return false;
                    }
                }
            });
            assertEquals(1, B.rows());
            assertEquals(NCOLUMNS, B.columns());
            assertEquals(A.getQuick(NROWS / 4 - DINDEX, NROWS / 4), B.getQuick(0, NROWS / 4));
        }
    }

    public void testViewSelectionIntArrayIntArray() {
        int[] rowIndexes = new int[] { NROWS / 6, NROWS / 5, NROWS / 4, NROWS / 3, NROWS / 2 };
        int[] colIndexes = new int[] { NROWS / 6, NROWS / 5, NROWS / 4, NROWS / 3, NROWS / 2, NROWS - 1 };
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
        for (int r = 0; r < NROWS - 1; r++) {
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

    public void testZMultLongMatrix2DLongMatrix2DLongLongBooleanBoolean() {
        long alpha = 3;
        long beta = 5;
        LongMatrix2D C = new DiagonalLongMatrix2D(NROWS, NROWS, 0);
        for (int i = 0; i < DLENGTH; i++) {
            C.setQuick(i, i, Math.max(1, rand.nextLong() % A.rows()));
        }
        long[][] expected = C.toArray();
        C = A.zMult(Bt, C, alpha, beta, false, false);
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                long s = 0;
                for (int k = 0; k < NCOLUMNS; k++) {
                    s += A.getQuick(i, k) * Bt.getQuick(k, j);
                }
                expected[i][j] = s * alpha + expected[i][j] * beta;
            }
        }
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NROWS; c++) {
                assertEquals(expected[r][c], C.getQuick(r, c));
            }
        }

        //---
        C = null;
        C = A.zMult(Bt, C, alpha, beta, false, false);
        expected = new long[NROWS][NROWS];
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                long s = 0;
                for (int k = 0; k < NCOLUMNS; k++) {
                    s += A.getQuick(i, k) * Bt.getQuick(k, j);
                }
                expected[i][j] = s * alpha;
            }
        }
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NROWS; c++) {
                assertEquals(expected[r][c], C.getQuick(r, c));
            }
        }

        //transposeA
        C = new DiagonalLongMatrix2D(NCOLUMNS, NCOLUMNS, 0);
        for (int i = 0; i < DLENGTH; i++) {
            C.setQuick(i, i, Math.max(1, rand.nextLong() % A.rows()));
        }
        expected = C.toArray();
        C = A.zMult(B, C, alpha, beta, true, false);
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                long s = 0;
                for (int k = 0; k < NROWS; k++) {
                    s += A.getQuick(k, i) * B.getQuick(k, j);
                }
                expected[i][j] = s * alpha + expected[i][j] * beta;
            }
        }
        for (int r = 0; r < NCOLUMNS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(expected[r][c], C.getQuick(r, c));
            }
        }
        //---
        C = null;
        C = A.zMult(B, C, alpha, beta, true, false);
        expected = new long[NCOLUMNS][NCOLUMNS];
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                long s = 0;
                for (int k = 0; k < NROWS; k++) {
                    s += A.getQuick(k, i) * B.getQuick(k, j);
                }
                expected[i][j] = s * alpha;
            }
        }
        for (int r = 0; r < NCOLUMNS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(expected[r][c], C.getQuick(r, c));
            }
        }

        //transposeB
        C = new DiagonalLongMatrix2D(NROWS, NROWS, 0);
        for (int i = 0; i < DLENGTH; i++) {
            C.setQuick(i, i, Math.max(1, rand.nextLong() % A.rows()));
        }
        expected = C.toArray();
        C = A.zMult(B, C, alpha, beta, false, true);
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                long s = 0;
                for (int k = 0; k < NCOLUMNS; k++) {
                    s += A.getQuick(i, k) * B.getQuick(j, k);
                }
                expected[i][j] = s * alpha + expected[i][j] * beta;
            }
        }
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NROWS; c++) {
                assertEquals(expected[r][c], C.getQuick(r, c));
            }
        }
        //---
        C = null;
        C = A.zMult(B, C, alpha, beta, false, true);
        expected = new long[NROWS][NROWS];
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                long s = 0;
                for (int k = 0; k < NCOLUMNS; k++) {
                    s += A.getQuick(i, k) * B.getQuick(j, k);
                }
                expected[i][j] = s * alpha;
            }
        }
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NROWS; c++) {
                assertEquals(expected[r][c], C.getQuick(r, c));
            }
        }
        //transposeA and transposeB
        C = new DiagonalLongMatrix2D(NCOLUMNS, NCOLUMNS, 0);
        for (int i = 0; i < DLENGTH; i++) {
            C.setQuick(i, i, Math.max(1, rand.nextLong() % A.rows()));
        }
        expected = C.toArray();
        C = A.zMult(Bt, C, alpha, beta, true, true);
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                long s = 0;
                for (int k = 0; k < NROWS; k++) {
                    s += A.getQuick(k, i) * Bt.getQuick(j, k);
                }
                expected[i][j] = s * alpha + expected[i][j] * beta;
            }
        }
        for (int r = 0; r < NCOLUMNS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(expected[r][c], C.getQuick(r, c));
            }
        }
        //---
        C = null;
        C = A.zMult(Bt, C, alpha, beta, true, true);
        expected = new long[NCOLUMNS][NCOLUMNS];
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                long s = 0;
                for (int k = 0; k < NROWS; k++) {
                    s += A.getQuick(k, i) * Bt.getQuick(j, k);
                }
                expected[i][j] = s * alpha;
            }
        }
        for (int r = 0; r < NCOLUMNS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(expected[r][c], C.getQuick(r, c));
            }
        }

    }
}
