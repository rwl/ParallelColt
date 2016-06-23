package cern.colt.matrix.tdcomplex.impl;

import java.util.ArrayList;

import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix1DProcedure;
import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2DTest;
import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.jet.math.tdcomplex.DComplex;
import cern.jet.math.tdcomplex.DComplexFunctions;
import edu.emory.mathcs.utils.pc.ConcurrencyUtils;

public class DiagonalDComplexMatrix2DTest extends DComplexMatrix2DTest {

    protected int DLENGTH;

    protected int DINDEX;

    public DiagonalDComplexMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        DINDEX = 3;
        A = new DiagonalDComplexMatrix2D(NROWS, NCOLUMNS, DINDEX);
        B = new DiagonalDComplexMatrix2D(NROWS, NCOLUMNS, DINDEX);
        Bt = new DiagonalDComplexMatrix2D(NCOLUMNS, NROWS, -DINDEX);
        DLENGTH = ((DiagonalDComplexMatrix2D) A).diagonalLength();

    }

    protected void populateMatrices() {
        ConcurrencyUtils.setThreadsBeginN_2D(1);
        if (DINDEX >= 0) {
            for (int r = 0; r < DLENGTH; r++) {
                A.setQuick(r, r + DINDEX, Math.random(), Math.random());
            }

            for (int r = 0; r < DLENGTH; r++) {
                B.setQuick(r, r + DINDEX, Math.random(), Math.random());
            }

            for (int r = 0; r < DLENGTH; r++) {
                Bt.setQuick(r - DINDEX, r, Math.random(), Math.random());
            }

        } else {
            for (int r = 0; r < DLENGTH; r++) {
                A.setQuick(r - DINDEX, r, Math.random(), Math.random());
            }

            for (int r = 0; r < DLENGTH; r++) {
                B.setQuick(r - DINDEX, r, Math.random(), Math.random());
            }
            for (int r = 0; r < DLENGTH; r++) {
                Bt.setQuick(r, r + DINDEX, Math.random(), Math.random());
            }

        }
    }

    public void testAssignDoubleDouble() {
        double[] value = new double[] { Math.random(), Math.random() };
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

    public void testAssignDoubleArray() {
        double[] expected = new double[2 * DLENGTH];
        for (int i = 0; i < 2 * DLENGTH; i++) {
            expected[i] = Math.random();
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
        DoubleMatrix2D Im = DoubleFactory2D.dense.random(A.rows(), A.columns());
        DComplexMatrix2D Acopy = A.copy();
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
        DoubleMatrix2D Re = DoubleFactory2D.dense.random(A.rows(), A.columns());
        DComplexMatrix2D Acopy = A.copy();
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

    public void testAssignDoubleArrayArray() {
        double[][] expected = new double[NROWS][2 * NCOLUMNS];
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                expected[r][2 * c] = Math.random();
                expected[r][2 * c + 1] = Math.random();
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
        DComplexMatrix2D Acopy = A.copy();
        A.assign(DComplexFunctions.acos);
        if (DINDEX >= 0) {
            for (int r = 0; r < DLENGTH; r++) {
                double[] expected = DComplex.acos(Acopy.getQuick(r, r + DINDEX));
                assertEquals(expected[0], A.getQuick(r, r + DINDEX)[0], TOL);
                assertEquals(expected[1], A.getQuick(r, r + DINDEX)[1], TOL);
            }
        } else {
            for (int r = 0; r < DLENGTH; r++) {
                double[] expected = DComplex.acos(Acopy.getQuick(r - DINDEX, r));
                assertEquals(expected[0], A.getQuick(r - DINDEX, r)[0], TOL);
                assertEquals(expected[1], A.getQuick(r - DINDEX, r)[1], TOL);
            }
        }
    }

    public void testAssignComplexMatrix2DComplexComplexComplexFunction() {
        DComplexMatrix2D Acopy = A.copy();
        A.assign(B, DComplexFunctions.div);
        if (DINDEX >= 0) {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(DComplex.div(Acopy.getQuick(r, r + DINDEX), B.getQuick(r, r + DINDEX))[0], A.getQuick(r, r
                        + DINDEX)[0], TOL);
                assertEquals(DComplex.div(Acopy.getQuick(r, r + DINDEX), B.getQuick(r, r + DINDEX))[1], A.getQuick(r, r
                        + DINDEX)[1], TOL);
            }
        } else {
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(DComplex.div(Acopy.getQuick(r - DINDEX, r), B.getQuick(r - DINDEX, r))[0], A.getQuick(r
                        - DINDEX, r)[0], TOL);
                assertEquals(DComplex.div(Acopy.getQuick(r - DINDEX, r), B.getQuick(r - DINDEX, r))[1], A.getQuick(r
                        - DINDEX, r)[1], TOL);
            }
        }
    }

    public void testAssignDComplexMatrix2DDComplexDComplexFunctionIntArrayListIntArrayList() {
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        if (DINDEX >= 0) {
            for (int r = 0; r < DLENGTH; r++) {
                rowList.add(r);
                columnList.add(r + DINDEX);
            }
            DComplexMatrix2D Acopy = A.copy();
            A.assign(B, DComplexFunctions.div, rowList, columnList);
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(DComplex.div(Acopy.getQuick(r, r + DINDEX), B.getQuick(r, r + DINDEX))[0], A.getQuick(r, r
                        + DINDEX)[0], TOL);
                assertEquals(DComplex.div(Acopy.getQuick(r, r + DINDEX), B.getQuick(r, r + DINDEX))[1], A.getQuick(r, r
                        + DINDEX)[1], TOL);
            }
        } else {
            for (int r = 0; r < DLENGTH; r++) {
                rowList.add(r - DINDEX);
                columnList.add(r);
            }
            DComplexMatrix2D Acopy = A.copy();
            A.assign(B, DComplexFunctions.div, rowList, columnList);
            for (int r = 0; r < DLENGTH; r++) {
                assertEquals(DComplex.div(Acopy.getQuick(r - DINDEX, r), B.getQuick(r - DINDEX, r))[0], A.getQuick(r
                        - DINDEX, r)[0], TOL);
                assertEquals(DComplex.div(Acopy.getQuick(r - DINDEX, r), B.getQuick(r - DINDEX, r))[1], A.getQuick(r
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
        double[] elem1 = new double[] { 0.7, 0.8 };
        double[] elem2 = new double[] { 0.1, 0.2 };
        if (DINDEX >= 0) {
            A.setQuick(NROWS / 3, NROWS / 3 + DINDEX, elem1);
            A.setQuick(NROWS / 2, NROWS / 2 + DINDEX, elem2);
            IntArrayList rowList = new IntArrayList();
            IntArrayList columnList = new IntArrayList();
            ArrayList<double[]> valueList = new ArrayList<double[]>();
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
            ArrayList<double[]> valueList = new ArrayList<double[]>();
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
        double[][] array = A.toArray();
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
        DComplexMatrix1D Avec = A.vectorize();
        int idx = 0;
        for (int c = 0; c < NCOLUMNS; c++) {
            for (int r = 0; r < NROWS; r++) {
                assertEquals(A.getQuick(r, c), Avec.getQuick(idx++), TOL);
            }
        }
    }

    public void testViewColumn() {
        DComplexMatrix1D col = A.viewColumn(NCOLUMNS / 2);
        assertEquals(NROWS, col.size());
        for (int r = 0; r < NROWS; r++) {
            assertEquals(A.getQuick(r, NCOLUMNS / 2), col.getQuick(r), TOL);
        }
    }

    public void testViewColumnFlip() {
        DComplexMatrix2D B = A.viewColumnFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, NCOLUMNS - 1 - c), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewDice() {
        DComplexMatrix2D B = A.viewDice();
        assertEquals(NROWS, B.columns());
        assertEquals(NCOLUMNS, B.rows());
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(c, r), TOL);
            }
        }
    }

    public void testViewPart() {
        DComplexMatrix2D B = A.viewPart(NROWS / 2, NCOLUMNS / 2, NROWS / 3, NCOLUMNS / 3);
        assertEquals(NROWS / 3, B.rows());
        assertEquals(NCOLUMNS / 3, B.columns());
        for (int r = 0; r < NROWS / 3; r++) {
            for (int c = 0; c < NCOLUMNS / 3; c++) {
                assertEquals(A.getQuick(NROWS / 2 + r, NCOLUMNS / 2 + c), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewRow() {
        DComplexMatrix1D B = A.viewRow(NROWS / 2);
        assertEquals(NCOLUMNS, B.size());
        for (int r = 0; r < NCOLUMNS; r++) {
            assertEquals(A.getQuick(NROWS / 2, r), B.getQuick(r), TOL);
        }
    }

    public void testViewRowFlip() {
        DComplexMatrix2D B = A.viewRowFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(NROWS - 1 - r, c), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewSelectionComplexMatrix1DProcedure() {
        final double[] value = new double[] { 2, 3 };
        A.assign(0, 0);
        if (DINDEX >= 0) {
            A.setQuick(NROWS / 4, NROWS / 4 + DINDEX, value);
            A.setQuick(NROWS / 2, NROWS / 2 + DINDEX, value);
            DComplexMatrix2D B = A.viewSelection(new DComplexMatrix1DProcedure() {
                public boolean apply(DComplexMatrix1D element) {
                    if (DComplex.abs(DComplex.minus(element.getQuick(NROWS / 4 + DINDEX), value)) < TOL) {
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
            DComplexMatrix2D B = A.viewSelection(new DComplexMatrix1DProcedure() {
                public boolean apply(DComplexMatrix1D element) {
                    if (DComplex.abs(DComplex.minus(element.getQuick(NROWS / 4), value)) < TOL) {
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
        DComplexMatrix2D B = A.viewSelection(rowIndexes, colIndexes);
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
        DComplexMatrix2D B = A.viewStrides(rowStride, colStride);
        for (int r = 0; r < B.rows(); r++) {
            for (int c = 0; c < B.columns(); c++) {
                assertEquals(A.getQuick(r * rowStride, c * colStride), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testZMultDComplexMatrix2DDComplexMatrix2DDComplexDComplexBooleanBoolean() {
        double[] alpha = new double[] { 3, 4 };
        double[] beta = new double[] { 5, 6 };
        DComplexMatrix2D C = new DiagonalDComplexMatrix2D(NROWS, NROWS, 0);
        for (int i = 0; i < DLENGTH; i++) {
            C.setQuick(i, i, Math.random(), Math.random());
        }
        double[][] expected = C.toArray();
        C = A.zMult(Bt, C, alpha, beta, false, false);
        double[] elem = new double[2];
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                double[] s = new double[2];
                for (int k = 0; k < NCOLUMNS; k++) {
                    s = DComplex.plus(s, DComplex.mult(A.getQuick(i, k), Bt.getQuick(k, j)));
                }
                elem[0] = expected[i][2 * j];
                elem[1] = expected[i][2 * j + 1];
                elem = DComplex.mult(beta, elem);
                s = DComplex.mult(alpha, s);
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
        expected = new double[NROWS][2 * NROWS];
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                double[] s = new double[2];
                for (int k = 0; k < NCOLUMNS; k++) {
                    s = DComplex.plus(s, DComplex.mult(A.getQuick(i, k), Bt.getQuick(k, j)));
                }
                s = DComplex.mult(alpha, s);
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
        C = new DiagonalDComplexMatrix2D(NCOLUMNS, NCOLUMNS, 0);
        for (int i = 0; i < DLENGTH; i++) {
            C.setQuick(i, i, Math.random(), Math.random());
        }
        expected = C.toArray();
        C = A.zMult(B, C, alpha, beta, true, false);
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                double[] s = new double[2];
                for (int k = 0; k < NROWS; k++) {
                    s = DComplex.plus(s, DComplex.mult(DComplex.conj(A.getQuick(k, i)), B.getQuick(k, j)));
                }
                elem[0] = expected[i][2 * j];
                elem[1] = expected[i][2 * j + 1];
                elem = DComplex.mult(beta, elem);
                s = DComplex.mult(alpha, s);
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
        expected = new double[NCOLUMNS][2 * NCOLUMNS];
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                double[] s = new double[2];
                for (int k = 0; k < NROWS; k++) {
                    s = DComplex.plus(s, DComplex.mult(DComplex.conj(A.getQuick(k, i)), B.getQuick(k, j)));
                }
                s = DComplex.mult(alpha, s);
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
        C = new DiagonalDComplexMatrix2D(NROWS, NROWS, 0);
        for (int i = 0; i < DLENGTH; i++) {
            C.setQuick(i, i, Math.random(), Math.random());
        }
        expected = C.toArray();
        C = A.zMult(B, C, alpha, beta, false, true);
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                double[] s = new double[2];
                for (int k = 0; k < NCOLUMNS; k++) {
                    s = DComplex.plus(s, DComplex.mult(A.getQuick(i, k), DComplex.conj(B.getQuick(j, k))));
                }
                elem[0] = expected[i][2 * j];
                elem[1] = expected[i][2 * j + 1];
                elem = DComplex.mult(beta, elem);
                s = DComplex.mult(alpha, s);
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
        expected = new double[NROWS][2 * NROWS];
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                double[] s = new double[2];
                for (int k = 0; k < NCOLUMNS; k++) {
                    s = DComplex.plus(s, DComplex.mult(A.getQuick(i, k), DComplex.conj(B.getQuick(j, k))));
                }
                s = DComplex.mult(alpha, s);
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
        C = new DiagonalDComplexMatrix2D(NCOLUMNS, NCOLUMNS, 0);
        for (int i = 0; i < DLENGTH; i++) {
            C.setQuick(i, i, Math.random(), Math.random());
        }
        expected = C.toArray();
        C = A.zMult(Bt, C, alpha, beta, true, true);
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                double[] s = new double[2];
                for (int k = 0; k < NROWS; k++) {
                    s = DComplex.plus(s, DComplex.mult(DComplex.conj(A.getQuick(k, i)), DComplex
                            .conj(Bt.getQuick(j, k))));
                }
                elem[0] = expected[i][2 * j];
                elem[1] = expected[i][2 * j + 1];
                elem = DComplex.mult(beta, elem);
                s = DComplex.mult(alpha, s);
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
        expected = new double[NCOLUMNS][2 * NCOLUMNS];
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                double[] s = new double[2];
                for (int k = 0; k < NROWS; k++) {
                    s = DComplex.plus(s, DComplex.mult(A.getQuick(k, i), Bt.getQuick(j, k)));
                }
                s = DComplex.mult(alpha, s);
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
