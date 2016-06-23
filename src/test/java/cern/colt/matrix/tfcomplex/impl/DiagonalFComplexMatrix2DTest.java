package cern.colt.matrix.tfcomplex.impl;

import java.util.ArrayList;

import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.FComplexMatrix1DProcedure;
import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfcomplex.FComplexMatrix2DTest;
import cern.colt.matrix.tfloat.FloatFactory2D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.jet.math.tfcomplex.FComplex;
import cern.jet.math.tfcomplex.FComplexFunctions;
import edu.emory.mathcs.utils.pc.ConcurrencyUtils;

public class DiagonalFComplexMatrix2DTest extends FComplexMatrix2DTest {

    protected int DLENGTH;

    protected int DINDEX;

    public DiagonalFComplexMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        DINDEX = 3;
        A = new DiagonalFComplexMatrix2D(NROWS, NCOLUMNS, DINDEX);
        B = new DiagonalFComplexMatrix2D(NROWS, NCOLUMNS, DINDEX);
        Bt = new DiagonalFComplexMatrix2D(NCOLUMNS, NROWS, -DINDEX);
        DLENGTH = ((DiagonalFComplexMatrix2D) A).diagonalLength();

    }

    protected void populateMatrices() {
        ConcurrencyUtils.setThreadsBeginN_2D(1);
        if (DINDEX >= 0) {
            for (int r = 0; r < DLENGTH; r++) {
                A.setQuick(r, r + DINDEX, (float) Math.random(), (float) Math.random());
            }

            for (int r = 0; r < DLENGTH; r++) {
                B.setQuick(r, r + DINDEX, (float) Math.random(), (float) Math.random());
            }

            for (int r = 0; r < DLENGTH; r++) {
                Bt.setQuick(r - DINDEX, r, (float) Math.random(), (float) Math.random());
            }

        } else {
            for (int r = 0; r < DLENGTH; r++) {
                A.setQuick(r - DINDEX, r, (float) Math.random(), (float) Math.random());
            }

            for (int r = 0; r < DLENGTH; r++) {
                B.setQuick(r - DINDEX, r, (float) Math.random(), (float) Math.random());
            }
            for (int r = 0; r < DLENGTH; r++) {
                Bt.setQuick(r, r + DINDEX, (float) Math.random(), (float) Math.random());
            }

        }
    }

    public void testAssignFloatFloat() {
        float[] value = new float[] { (float) Math.random(), (float) Math.random() };
        A.assign(value[0], value[1]);
        if (DINDEX >= 0) {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(value, A.getQuick(r, r + DINDEX), TOL);
            }
        } else {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(value, A.getQuick(r - DINDEX, r), TOL);
            }
        }
    }

    public void testAssignFloatArray() {
        float[] expected = new float[2 * DLENGTH];
        for (int i = 0; i < 2 * DLENGTH; i++) {
            expected[i] = (float) Math.random();
        }
        A.assign(expected);
        if (DINDEX >= 0) {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(expected[2 * r], A.getQuick(r, r + DINDEX)[0], TOL);
                assertEquals(expected[2 * r + 1], A.getQuick(r, r + DINDEX)[1], TOL);
            }
        } else {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(expected[2 * r], A.getQuick(r - DINDEX, r)[0], TOL);
                assertEquals(expected[2 * r + 1], A.getQuick(r - DINDEX, r)[1], TOL);
            }
        }
    }

    public void testAssignImaginary() {
        FloatMatrix2D Im = FloatFactory2D.dense.random(A.rows(), A.columns());
        FComplexMatrix2D Acopy = A.copy();
        A.assignImaginary(Im);
        if (DINDEX >= 0) {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(Acopy.getQuick(r, r + DINDEX)[0], A.getQuick(r, r + DINDEX)[0], TOL);
                assertEquals(Im.getQuick(r, r + DINDEX), A.getQuick(r, r + DINDEX)[1], TOL);
            }
        } else {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(Acopy.getQuick(r - DINDEX, r)[0], A.getQuick(r - DINDEX, r)[0], TOL);
                assertEquals(Im.getQuick(r - DINDEX, r), A.getQuick(r - DINDEX, r)[1], TOL);
            }
        }
    }

    public void testAssignReal() {
        FloatMatrix2D Re = FloatFactory2D.dense.random(A.rows(), A.columns());
        FComplexMatrix2D Acopy = A.copy();
        A.assignReal(Re);
        if (DINDEX >= 0) {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(Acopy.getQuick(r, r + DINDEX)[1], A.getQuick(r, r + DINDEX)[1], TOL);
                assertEquals(Re.getQuick(r, r + DINDEX), A.getQuick(r, r + DINDEX)[0], TOL);
            }
        } else {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(Acopy.getQuick(r - DINDEX, r)[1], A.getQuick(r - DINDEX, r)[1], TOL);
                assertEquals(Re.getQuick(r - DINDEX, r), A.getQuick(r - DINDEX, r)[0], TOL);
            }
        }
    }

    public void testAssignFloatArrayArray() {
        float[][] expected = new float[NROWS][2 * NCOLUMNS];
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                expected[r][2 * c] = (float) Math.random();
                expected[r][2 * c + 1] = (float) Math.random();
            }
        }
        A.assign(expected);
        if (DINDEX >= 0) {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(expected[r][2 * (r + DINDEX)], A.getQuick(r, r + DINDEX)[0], TOL);
                assertEquals(expected[r][2 * (r + DINDEX) + 1], A.getQuick(r, r + DINDEX)[1], TOL);
            }
        } else {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(expected[r - DINDEX][2 * r], A.getQuick(r - DINDEX, r)[0], TOL);
                assertEquals(expected[r - DINDEX][2 * r + 1], A.getQuick(r - DINDEX, r)[1], TOL);
            }
        }
    }

    public void testAssignComplexComplexFunction() {
        FComplexMatrix2D Acopy = A.copy();
        A.assign(FComplexFunctions.acos);
        if (DINDEX >= 0) {
            for (int r = 0; r < DLENGTH; r++) {
                float[] expected = FComplex.acos(Acopy.getQuick(r, r + DINDEX));
                assertEquals(expected[0], A.getQuick(r, r + DINDEX)[0], TOL);
                assertEquals(expected[1], A.getQuick(r, r + DINDEX)[1], TOL);
            }
        } else {
            for (int r = 0; r < DLENGTH; r++) {
                float[] expected = FComplex.acos(Acopy.getQuick(r - DINDEX, r));
                assertEquals(expected[0], A.getQuick(r - DINDEX, r)[0], TOL);
                assertEquals(expected[1], A.getQuick(r - DINDEX, r)[1], TOL);
            }
        }
    }

    public void testAssignComplexMatrix2FComplexComplexComplexFunction() {
        FComplexMatrix2D Acopy = A.copy();
        A.assign(B, FComplexFunctions.div);
        if (DINDEX >= 0) {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(FComplex.div(Acopy.getQuick(r, r + DINDEX), B.getQuick(r, r + DINDEX))[0], A.getQuick(r, r
                        + DINDEX)[0], TOL);
                assertEquals(FComplex.div(Acopy.getQuick(r, r + DINDEX), B.getQuick(r, r + DINDEX))[1], A.getQuick(r, r
                        + DINDEX)[1], TOL);
            }
        } else {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(FComplex.div(Acopy.getQuick(r - DINDEX, r), B.getQuick(r - DINDEX, r))[0], A.getQuick(r
                        - DINDEX, r)[0], TOL);
                assertEquals(FComplex.div(Acopy.getQuick(r - DINDEX, r), B.getQuick(r - DINDEX, r))[1], A.getQuick(r
                        - DINDEX, r)[1], TOL);
            }
        }
    }

    public void testAssignFComplexMatrix2DFComplexFComplexFunctionIntArrayListIntArrayList() {
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        if (DINDEX >= 0) {
            for (int r = 0; r < DLENGTH; r++) {
                rowList.add(r);
                columnList.add(r + DINDEX);
            }
            FComplexMatrix2D Acopy = A.copy();
            A.assign(B, FComplexFunctions.div, rowList, columnList);
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(FComplex.div(Acopy.getQuick(r, r + DINDEX), B.getQuick(r, r + DINDEX))[0], A.getQuick(r, r
                        + DINDEX)[0], TOL);
                assertEquals(FComplex.div(Acopy.getQuick(r, r + DINDEX), B.getQuick(r, r + DINDEX))[1], A.getQuick(r, r
                        + DINDEX)[1], TOL);
            }
        } else {
            for (int r = 0; r < DLENGTH; r++) {
                rowList.add(r - DINDEX);
                columnList.add(r);
            }
            FComplexMatrix2D Acopy = A.copy();
            A.assign(B, FComplexFunctions.div, rowList, columnList);
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(FComplex.div(Acopy.getQuick(r - DINDEX, r), B.getQuick(r - DINDEX, r))[0], A.getQuick(r
                        - DINDEX, r)[0], TOL);
                assertEquals(FComplex.div(Acopy.getQuick(r - DINDEX, r), B.getQuick(r - DINDEX, r))[1], A.getQuick(r
                        - DINDEX, r)[1], TOL);
            }
        }
    }

    public void testCardinality() {
        int card = A.cardinality();
        assertEquals(DLENGTH, card);
    }

    public void testGetNonZeros() {
        A.assign(0, 0);
        float[] elem1 = new float[] { 0.7f, 0.8f };
        float[] elem2 = new float[] { 0.1f, 0.2f };
        if (DINDEX >= 0) {
            A.setQuick(NROWS / 3, NROWS / 3 + DINDEX, elem1);
            A.setQuick(NROWS / 2, NROWS / 2 + DINDEX, elem2);
            IntArrayList rowList = new IntArrayList();
            IntArrayList columnList = new IntArrayList();
            ArrayList<float[]> valueList = new ArrayList<float[]>();
            A.getNonZeros(rowList, columnList, valueList);
            assertEquals(2, rowList.size());
            assertEquals(2, columnList.size());
            assertEquals(2, valueList.size());
            assertTrue(rowList.contains(NROWS / 3));
            assertTrue(rowList.contains(NROWS / 2));
            assertTrue(columnList.contains(NROWS / 3 + DINDEX));
            assertTrue(columnList.contains(NROWS / 2 + DINDEX));
            assertEquals(A.getQuick(rowList.get(0), columnList.get(0)), valueList.get(0), TOL);
            assertEquals(A.getQuick(rowList.get(1), columnList.get(1)), valueList.get(1), TOL);
        } else {
            A.setQuick(NROWS / 3 - DINDEX, NROWS / 3, elem1);
            A.setQuick(NROWS / 2 - DINDEX, NROWS / 2, elem2);
            IntArrayList rowList = new IntArrayList();
            IntArrayList columnList = new IntArrayList();
            ArrayList<float[]> valueList = new ArrayList<float[]>();
            A.getNonZeros(rowList, columnList, valueList);
            assertEquals(2, rowList.size());
            assertEquals(2, columnList.size());
            assertEquals(2, valueList.size());
            assertTrue(rowList.contains(NROWS / 3 - DINDEX));
            assertTrue(rowList.contains(NROWS / 2 - DINDEX));
            assertTrue(columnList.contains(NROWS / 3));
            assertTrue(columnList.contains(NROWS / 2));
            assertEquals(A.getQuick(rowList.get(0), columnList.get(0)), valueList.get(0), TOL);
            assertEquals(A.getQuick(rowList.get(1), columnList.get(1)), valueList.get(1), TOL);
        }
    }

    public void testToArray() {
        float[][] array = A.toArray();
        assertTrue(NROWS == array.length);
        for (int r = 0; r < NROWS; r++) {
            assertTrue(2 * NCOLUMNS == array[r].length);
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(array[r][2 * c], A.getQuick(r, c)[0], TOL);
                assertEquals(array[r][2 * c + 1], A.getQuick(r, c)[1], TOL);
            }
        }
    }

    public void testVectorize() {
        FComplexMatrix1D Avec = A.vectorize();
        int idx = 0;
        for (int c = 0; c < NCOLUMNS; c++) {
            for (int r = 0; r < NROWS; r++) {
                assertEquals(A.getQuick(r, c), Avec.getQuick(idx++), TOL);
            }
        }
    }

    public void testViewColumn() {
        FComplexMatrix1D col = A.viewColumn(NCOLUMNS / 2);
        assertEquals(NROWS, col.size());
        for (int r = 0; r < NROWS; r++) {
            assertEquals(A.getQuick(r, NCOLUMNS / 2), col.getQuick(r), TOL);
        }
    }

    public void testViewColumnFlip() {
        FComplexMatrix2D B = A.viewColumnFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, NCOLUMNS - 1 - c), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewDice() {
        FComplexMatrix2D B = A.viewDice();
        assertEquals(NROWS, B.columns());
        assertEquals(NCOLUMNS, B.rows());
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(c, r), TOL);
            }
        }
    }

    public void testViewPart() {
        FComplexMatrix2D B = A.viewPart(NROWS / 2, NCOLUMNS / 2, NROWS / 3, NCOLUMNS / 3);
        assertEquals(NROWS / 3, B.rows());
        assertEquals(NCOLUMNS / 3, B.columns());
        for (int r = 0; r < NROWS / 3; r++) {
            for (int c = 0; c < NCOLUMNS / 3; c++) {
                assertEquals(A.getQuick(NROWS / 2 + r, NCOLUMNS / 2 + c), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewRow() {
        FComplexMatrix1D B = A.viewRow(NROWS / 2);
        assertEquals(NCOLUMNS, B.size());
        for (int r = 0; r < NCOLUMNS; r++) {
            assertEquals(A.getQuick(NROWS / 2, r), B.getQuick(r), TOL);
        }
    }

    public void testViewRowFlip() {
        FComplexMatrix2D B = A.viewRowFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(NROWS - 1 - r, c), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewSelectionComplexMatrix1DProcedure() {
        final float[] value = new float[] { 2, 3 };
        A.assign(0, 0);
        if (DINDEX >= 0) {
            A.setQuick(NROWS / 4, NROWS / 4 + DINDEX, value);
            A.setQuick(NROWS / 2, NROWS / 2 + DINDEX, value);
            FComplexMatrix2D B = A.viewSelection(new FComplexMatrix1DProcedure() {
                public boolean apply(FComplexMatrix1D element) {
                    if (FComplex.abs(FComplex.minus(element.getQuick(NROWS / 4 + DINDEX), value)) < TOL) {
                        return true;
                    } else {
                        return false;
                    }
                }
            });
            assertEquals(1, B.rows());
            assertEquals(NCOLUMNS, B.columns());
            assertEquals(A.getQuick(NROWS / 4, NROWS / 4 + DINDEX), B.getQuick(0, NROWS / 4 + DINDEX), TOL);
        } else {
            A.setQuick(NROWS / 4 - DINDEX, NROWS / 4, value);
            A.setQuick(NROWS / 2 - DINDEX, NROWS / 2, value);
            FComplexMatrix2D B = A.viewSelection(new FComplexMatrix1DProcedure() {
                public boolean apply(FComplexMatrix1D element) {
                    if (FComplex.abs(FComplex.minus(element.getQuick(NROWS / 4), value)) < TOL) {
                        return true;
                    } else {
                        return false;
                    }
                }
            });
            assertEquals(1, B.rows());
            assertEquals(NCOLUMNS, B.columns());
            assertEquals(A.getQuick(NROWS / 4 - DINDEX, NROWS / 4), B.getQuick(0, NROWS / 4), TOL);
        }
    }

    public void testViewSelectionIntArrayIntArray() {
        int[] rowIndexes = new int[] { NROWS / 6, NROWS / 5, NROWS / 4, NROWS / 3, NROWS / 2 };
        int[] colIndexes = new int[] { NROWS / 6, NROWS / 5, NROWS / 4, NROWS / 3, NROWS / 2, NROWS - 1 };
        FComplexMatrix2D B = A.viewSelection(rowIndexes, colIndexes);
        assertEquals(rowIndexes.length, B.rows());
        assertEquals(colIndexes.length, B.columns());
        for (int r = 0; r < rowIndexes.length; r++) {
            for (int c = 0; c < colIndexes.length; c++) {
                assertEquals(A.getQuick(rowIndexes[r], colIndexes[c]), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewStrides() {
        int rowStride = 3;
        int colStride = 5;
        FComplexMatrix2D B = A.viewStrides(rowStride, colStride);
        for (int r = 0; r < B.rows(); r++) {
            for (int c = 0; c < B.columns(); c++) {
                assertEquals(A.getQuick(r * rowStride, c * colStride), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testZMultFComplexMatrix2DFComplexMatrix2DFComplexFComplexBooleanBoolean() {
        float[] alpha = new float[] { 3, 4 };
        float[] beta = new float[] { 5, 6 };
        FComplexMatrix2D C = new DiagonalFComplexMatrix2D(NROWS, NROWS, 0);
        for (int i = 0; i < DLENGTH; i++) {
            C.setQuick(i, i, (float) Math.random(), (float) Math.random());
        }
        float[][] expected = C.toArray();
        C = A.zMult(Bt, C, alpha, beta, false, false);
        float[] elem = new float[2];
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                float[] s = new float[2];
                for (int k = 0; k < NCOLUMNS; k++) {
                    s = FComplex.plus(s, FComplex.mult(A.getQuick(i, k), Bt.getQuick(k, j)));
                }
                elem[0] = expected[i][2 * j];
                elem[1] = expected[i][2 * j + 1];
                elem = FComplex.mult(beta, elem);
                s = FComplex.mult(alpha, s);
                expected[i][2 * j] = s[0] + elem[0];
                expected[i][2 * j + 1] = s[1] + elem[1];
            }
        }
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NROWS; c++) {
                assertEquals(expected[r][2 * c], C.getQuick(r, c)[0], TOL);
                assertEquals(expected[r][2 * c + 1], C.getQuick(r, c)[1], TOL);
            }
        }

        //---
        C = null;
        C = A.zMult(Bt, C, alpha, beta, false, false);
        expected = new float[NROWS][2 * NROWS];
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                float[] s = new float[2];
                for (int k = 0; k < NCOLUMNS; k++) {
                    s = FComplex.plus(s, FComplex.mult(A.getQuick(i, k), Bt.getQuick(k, j)));
                }
                s = FComplex.mult(alpha, s);
                expected[i][2 * j] = s[0];
                expected[i][2 * j + 1] = s[1];
            }
        }
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NROWS; c++) {
                assertEquals(expected[r][2 * c], C.getQuick(r, c)[0], TOL);
                assertEquals(expected[r][2 * c + 1], C.getQuick(r, c)[1], TOL);
            }
        }

        //transposeA
        C = new DiagonalFComplexMatrix2D(NCOLUMNS, NCOLUMNS, 0);
        for (int i = 0; i < DLENGTH; i++) {
            C.setQuick(i, i, (float) Math.random(), (float) Math.random());
        }
        expected = C.toArray();
        C = A.zMult(B, C, alpha, beta, true, false);
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                float[] s = new float[2];
                for (int k = 0; k < NROWS; k++) {
                    s = FComplex.plus(s, FComplex.mult(FComplex.conj(A.getQuick(k, i)), B.getQuick(k, j)));
                }
                elem[0] = expected[i][2 * j];
                elem[1] = expected[i][2 * j + 1];
                elem = FComplex.mult(beta, elem);
                s = FComplex.mult(alpha, s);
                expected[i][2 * j] = s[0] + elem[0];
                expected[i][2 * j + 1] = s[1] + elem[1];
            }
        }
        for (int r = 0; r < NCOLUMNS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(expected[r][2 * c], C.getQuick(r, c)[0], TOL);
                assertEquals(expected[r][2 * c + 1], C.getQuick(r, c)[1], TOL);
            }
        }
        //---
        C = null;
        C = A.zMult(B, C, alpha, beta, true, false);
        expected = new float[NCOLUMNS][2 * NCOLUMNS];
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                float[] s = new float[2];
                for (int k = 0; k < NROWS; k++) {
                    s = FComplex.plus(s, FComplex.mult(FComplex.conj(A.getQuick(k, i)), B.getQuick(k, j)));
                }
                s = FComplex.mult(alpha, s);
                expected[i][2 * j] = s[0];
                expected[i][2 * j + 1] = s[1];
            }
        }
        for (int r = 0; r < NCOLUMNS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(expected[r][2 * c], C.getQuick(r, c)[0], TOL);
                assertEquals(expected[r][2 * c + 1], C.getQuick(r, c)[1], TOL);
            }
        }

        //transposeB
        C = new DiagonalFComplexMatrix2D(NROWS, NROWS, 0);
        for (int i = 0; i < DLENGTH; i++) {
            C.setQuick(i, i, (float) Math.random(), (float) Math.random());
        }
        expected = C.toArray();
        C = A.zMult(B, C, alpha, beta, false, true);
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                float[] s = new float[2];
                for (int k = 0; k < NCOLUMNS; k++) {
                    s = FComplex.plus(s, FComplex.mult(A.getQuick(i, k), FComplex.conj(B.getQuick(j, k))));
                }
                elem[0] = expected[i][2 * j];
                elem[1] = expected[i][2 * j + 1];
                elem = FComplex.mult(beta, elem);
                s = FComplex.mult(alpha, s);
                expected[i][2 * j] = s[0] + elem[0];
                expected[i][2 * j + 1] = s[1] + elem[1];
            }
        }
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NROWS; c++) {
                assertEquals(expected[r][2 * c], C.getQuick(r, c)[0], TOL);
                assertEquals(expected[r][2 * c + 1], C.getQuick(r, c)[1], TOL);
            }
        }
        //---
        C = null;
        C = A.zMult(B, C, alpha, beta, false, true);
        expected = new float[NROWS][2 * NROWS];
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                float[] s = new float[2];
                for (int k = 0; k < NCOLUMNS; k++) {
                    s = FComplex.plus(s, FComplex.mult(A.getQuick(i, k), FComplex.conj(B.getQuick(j, k))));
                }
                s = FComplex.mult(alpha, s);
                expected[i][2 * j] = s[0];
                expected[i][2 * j + 1] = s[1];
            }
        }
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NROWS; c++) {
                assertEquals(expected[r][2 * c], C.getQuick(r, c)[0], TOL);
                assertEquals(expected[r][2 * c + 1], C.getQuick(r, c)[1], TOL);
            }
        }
        //transposeA and transposeB
        C = new DiagonalFComplexMatrix2D(NCOLUMNS, NCOLUMNS, 0);
        for (int i = 0; i < DLENGTH; i++) {
            C.setQuick(i, i, (float) Math.random(), (float) Math.random());
        }
        expected = C.toArray();
        C = A.zMult(Bt, C, alpha, beta, true, true);
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                float[] s = new float[2];
                for (int k = 0; k < NROWS; k++) {
                    s = FComplex.plus(s, FComplex.mult(FComplex.conj(A.getQuick(k, i)), FComplex
                            .conj(Bt.getQuick(j, k))));
                }
                elem[0] = expected[i][2 * j];
                elem[1] = expected[i][2 * j + 1];
                elem = FComplex.mult(beta, elem);
                s = FComplex.mult(alpha, s);
                expected[i][2 * j] = s[0] + elem[0];
                expected[i][2 * j + 1] = s[1] + elem[1];
            }
        }
        for (int r = 0; r < NCOLUMNS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(expected[r][2 * c], C.getQuick(r, c)[0], TOL);
                assertEquals(expected[r][2 * c + 1], C.getQuick(r, c)[1], TOL);
            }
        }
        //---
        C = null;
        C = A.zMult(Bt, C, alpha, beta, true, true);
        expected = new float[NCOLUMNS][2 * NCOLUMNS];
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                float[] s = new float[2];
                for (int k = 0; k < NROWS; k++) {
                    s = FComplex.plus(s, FComplex.mult(A.getQuick(k, i), Bt.getQuick(j, k)));
                }
                s = FComplex.mult(alpha, s);
                expected[i][2 * j] = s[0];
                expected[i][2 * j + 1] = s[1];
            }
        }
        for (int r = 0; r < NCOLUMNS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(expected[r][2 * c], C.getQuick(r, c)[0], TOL);
                assertEquals(expected[r][2 * c + 1], C.getQuick(r, c)[1], TOL);
            }
        }

    }
}
