package cern.colt.matrix.tdouble;

import junit.framework.TestCase;
import cern.colt.function.tdouble.DoubleProcedure;
import cern.colt.function.tdouble.IntIntDoubleFunction;
import cern.colt.list.tdouble.DoubleArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.jet.math.tdouble.DoubleFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public abstract class DoubleMatrix2DTest extends TestCase {
    /**
     * Matrix to test
     */
    protected DoubleMatrix2D A;

    /**
     * Matrix of the same size as A
     */
    protected DoubleMatrix2D B;

    /**
     * Matrix of the size A.columns() x A.rows()
     */
    protected DoubleMatrix2D Bt;

    protected int NROWS = 13;

    protected int NCOLUMNS = 17;

    protected double TOL = 1e-10;

    /**
     * Constructor for DoubleMatrix2DTest
     */
    public DoubleMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void setUp() throws Exception {
        createMatrices();
        populateMatrices();
    }

    protected abstract void createMatrices() throws Exception;

    protected void populateMatrices() {
        ConcurrencyUtils.setThreadsBeginN_2D(1);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                A.setQuick(r, c, Math.random());
            }
        }

        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                B.setQuick(r, c, Math.random());
            }
        }

        for (int r = 0; r < NCOLUMNS; r++) {
            for (int c = 0; c < NROWS; c++) {
                Bt.setQuick(r, c, Math.random());
            }
        }
    }

    @Override
    protected void tearDown() throws Exception {
        A = B = Bt = null;
    }

    public void testAggregateDoubleDoubleFunctionDoubleFunction() {
        double expected = 0;
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                double elem = A.getQuick(r, c);
                expected += elem * elem;
            }
        }
        double result = A.aggregate(DoubleFunctions.plus, DoubleFunctions.square);
        assertEquals(expected, result, TOL);
    }

    public void testAggregateDoubleDoubleFunctionDoubleFunctionDoubleProcedure() {
        DoubleProcedure procedure = new DoubleProcedure() {
            public boolean apply(double element) {
                if (Math.abs(element) > 0.2) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        double expected = 0;
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                double elem = A.getQuick(r, c);
                if (Math.abs(elem) > 0.2) {
                    expected += elem * elem;
                }
            }
        }

        double result = A.aggregate(DoubleFunctions.plus, DoubleFunctions.square, procedure);
        assertEquals(expected, result, TOL);
    }

    public void testAggregateDoubleDoubleFunctionDoubleFunctionIntArrayListIntArrayList() {
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                rowList.add(r);
                columnList.add(c);
            }
        }
        double expected = 0;
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                double elem = A.getQuick(r, c);
                expected += elem * elem;
            }
        }
        double result = A.aggregate(DoubleFunctions.plus, DoubleFunctions.square, rowList, columnList);
        assertEquals(expected, result, TOL);
    }

    public void testAggregateDoubleMatrix2DDoubleDoubleFunctionDoubleDoubleFunction() {
        double expected = 0;
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                double elemA = A.getQuick(r, c);
                double elemB = B.getQuick(r, c);
                expected += elemA * elemB;
            }
        }
        double result = A.aggregate(B, DoubleFunctions.plus, DoubleFunctions.mult);
        assertEquals(expected, result, TOL);
    }

    public void testAssignDouble() {
        double value = Math.random();
        A.assign(value);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(value, A.getQuick(r, c), TOL);
        }
    }

    public void testAssignDoubleArrayArray() {
        double[][] expected = new double[NROWS][NCOLUMNS];
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                expected[r][c] = Math.random();
            }
        }
        A.assign(expected);
        for (int r = 0; r < NROWS; r++) {
            assertTrue(NCOLUMNS == expected[r].length);
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(expected[r][c], A.getQuick(r, c), TOL);
        }
    }

    public void testAssignDoubleFunction() {
        DoubleMatrix2D Acopy = A.copy();
        A.assign(DoubleFunctions.acos);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                double expected = Math.acos(Acopy.getQuick(r, c));
                assertEquals(expected, A.getQuick(r, c), TOL);
            }
        }
    }

    public void testAssignDoubleMatrix2D() {
        A.assign(B);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(B.getQuick(r, c), A.getQuick(r, c), TOL);
        }
    }

    public void testAssignDoubleMatrix2DDoubleDoubleFunction() {
        DoubleMatrix2D Acopy = A.copy();
        A.assign(B, DoubleFunctions.div);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(Acopy.getQuick(r, c) / B.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testAssignDoubleMatrix2DDoubleDoubleFunctionIntArrayListIntArrayList() {
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                rowList.add(r);
                columnList.add(c);
            }
        }
        DoubleMatrix2D Acopy = A.copy();
        A.assign(B, DoubleFunctions.div, rowList, columnList);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(Acopy.getQuick(r, c) / B.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testAssignDoubleProcedureDouble() {
        DoubleProcedure procedure = new DoubleProcedure() {
            public boolean apply(double element) {
                if (Math.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        DoubleMatrix2D Acopy = A.copy();
        A.assign(procedure, -1.0);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                if (Math.abs(Acopy.getQuick(r, c)) > 0.1) {
                    assertEquals(-1.0, A.getQuick(r, c), TOL);
                } else {
                    assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
                }
            }
        }
    }

    public void testAssignDoubleProcedureDoubleFunction() {
        DoubleProcedure procedure = new DoubleProcedure() {
            public boolean apply(double element) {
                if (Math.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        DoubleMatrix2D Acopy = A.copy();
        A.assign(procedure, DoubleFunctions.tan);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                if (Math.abs(Acopy.getQuick(r, c)) > 0.1) {
                    assertEquals(Math.tan(Acopy.getQuick(r, c)), A.getQuick(r, c), TOL);
                } else {
                    assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
                }
            }
        }
    }

    public void testCardinality() {
        int card = A.cardinality();
        assertEquals(NROWS * NCOLUMNS, card);
    }

    public void testEqualsDouble() {
        double value = 1;
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
        DoubleMatrix2D Acopy = A.copy();
        IntIntDoubleFunction function = new IntIntDoubleFunction() {
            public double apply(int first, int second, double third) {
                return Math.sqrt(third);
            }
        };
        A.forEachNonZero(function);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(Math.sqrt(Acopy.getQuick(r, c)), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testMaxLocation() {
        A.assign(0);
        A.setQuick(NROWS / 3, NCOLUMNS / 3, 0.7);
        A.setQuick(NROWS / 2, NCOLUMNS / 2, 0.1);
        double[] maxAndLoc = A.getMaxLocation();
        assertEquals(0.7, maxAndLoc[0], TOL);
        assertEquals(NROWS / 3, (int) maxAndLoc[1]);
        assertEquals(NCOLUMNS / 3, (int) maxAndLoc[2]);
    }

    public void testMinLocation() {
        A.assign(0);
        A.setQuick(NROWS / 3, NCOLUMNS / 3, -0.7);
        A.setQuick(NROWS / 2, NCOLUMNS / 2, -0.1);
        double[] minAndLoc = A.getMinLocation();
        assertEquals(-0.7, minAndLoc[0], TOL);
        assertEquals(NROWS / 3, (int) minAndLoc[1]);
        assertEquals(NCOLUMNS / 3, (int) minAndLoc[2]);
    }

    public void testGetNegativeValues() {
        A.assign(0);
        A.setQuick(NROWS / 3, NCOLUMNS / 3, -0.7);
        A.setQuick(NROWS / 2, NCOLUMNS / 2, -0.1);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        A.getNegativeValues(rowList, columnList, valueList);
        assertEquals(2, rowList.size());
        assertEquals(2, columnList.size());
        assertEquals(2, valueList.size());
        assertTrue(rowList.contains(NROWS / 3));
        assertTrue(rowList.contains(NROWS / 2));
        assertTrue(columnList.contains(NCOLUMNS / 3));
        assertTrue(columnList.contains(NCOLUMNS / 2));
        assertTrue(valueList.contains(-0.7));
        assertTrue(valueList.contains(-0.1));
    }

    public void testGetNonZeros() {
        A.assign(0);
        A.setQuick(NROWS / 3, NCOLUMNS / 3, 0.7);
        A.setQuick(NROWS / 2, NCOLUMNS / 2, 0.1);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        A.getNonZeros(rowList, columnList, valueList);
        assertEquals(2, rowList.size());
        assertEquals(2, columnList.size());
        assertEquals(2, valueList.size());
        assertTrue(rowList.contains(NROWS / 3));
        assertTrue(rowList.contains(NROWS / 2));
        assertTrue(columnList.contains(NCOLUMNS / 3));
        assertTrue(columnList.contains(NCOLUMNS / 2));
        assertTrue(valueList.contains(0.7));
        assertTrue(valueList.contains(0.1));
    }

    public void testGetPositiveValues() {
        A.assign(0);
        A.setQuick(NROWS / 3, NCOLUMNS / 3, 0.7);
        A.setQuick(NROWS / 2, NCOLUMNS / 2, 0.1);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        A.getPositiveValues(rowList, columnList, valueList);
        assertEquals(2, rowList.size());
        assertEquals(2, columnList.size());
        assertEquals(2, valueList.size());
        assertTrue(rowList.contains(NROWS / 3));
        assertTrue(rowList.contains(NROWS / 2));
        assertTrue(columnList.contains(NCOLUMNS / 3));
        assertTrue(columnList.contains(NCOLUMNS / 2));
        assertTrue(valueList.contains(0.7));
        assertTrue(valueList.contains(0.1));
    }

    public void testToArray() {
        double[][] array = A.toArray();
        assertTrue(NROWS == array.length);
        for (int r = 0; r < NROWS; r++) {
            assertTrue(NCOLUMNS == array[r].length);
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(array[r][c] - A.getQuick(r, c)), TOL);
        }
    }

    public void testVectorize() {
        DoubleMatrix1D Avec = A.vectorize();
        int idx = 0;
        for (int c = 0; c < NCOLUMNS; c++) {
            for (int r = 0; r < NROWS; r++) {
                assertEquals(A.getQuick(r, c), Avec.getQuick(idx++), TOL);
            }
        }
    }

    public void testViewColumn() {
        DoubleMatrix1D col = A.viewColumn(NCOLUMNS / 2);
        assertEquals(NROWS, col.size());
        for (int r = 0; r < NROWS; r++) {
            assertEquals(A.getQuick(r, NCOLUMNS / 2), col.getQuick(r), TOL);
        }
    }

    public void testViewColumnFlip() {
        DoubleMatrix2D B = A.viewColumnFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, NCOLUMNS - 1 - c), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewDice() {
        DoubleMatrix2D B = A.viewDice();
        assertEquals(NROWS, B.columns());
        assertEquals(NCOLUMNS, B.rows());
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(c, r), TOL);
            }
        }
    }

    public void testViewPart() {
        DoubleMatrix2D B = A.viewPart(NROWS / 2, NCOLUMNS / 2, NROWS / 3, NCOLUMNS / 3);
        assertEquals(NROWS / 3, B.rows());
        assertEquals(NCOLUMNS / 3, B.columns());
        for (int r = 0; r < NROWS / 3; r++) {
            for (int c = 0; c < NCOLUMNS / 3; c++) {
                assertEquals(A.getQuick(NROWS / 2 + r, NCOLUMNS / 2 + c), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewRow() {
        DoubleMatrix1D B = A.viewRow(NROWS / 2);
        assertEquals(NCOLUMNS, B.size());
        for (int r = 0; r < NCOLUMNS; r++) {
            assertEquals(A.getQuick(NROWS / 2, r), B.getQuick(r), TOL);
        }
    }

    public void testViewRowFlip() {
        DoubleMatrix2D B = A.viewRowFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(NROWS - 1 - r, c), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewSelectionDoubleMatrix1DProcedure() {
        final double value = 2;
        A.assign(0);
        A.setQuick(NROWS / 4, 0, value);
        A.setQuick(NROWS / 2, 0, value);
        DoubleMatrix2D B = A.viewSelection(new DoubleMatrix1DProcedure() {
            public boolean apply(DoubleMatrix1D element) {
                if (Math.abs(element.getQuick(0) - value) < TOL) {
                    return true;
                } else {
                    return false;
                }
            }
        });
        assertEquals(2, B.rows());
        assertEquals(NCOLUMNS, B.columns());
        assertEquals(A.getQuick(NROWS / 4, 0), B.getQuick(0, 0), TOL);
        assertEquals(A.getQuick(NROWS / 2, 0), B.getQuick(1, 0), TOL);
    }

    public void testViewSelectionIntArrayIntArray() {
        int[] rowIndexes = new int[] { NROWS / 6, NROWS / 5, NROWS / 4, NROWS / 3, NROWS / 2 };
        int[] colIndexes = new int[] { NCOLUMNS / 6, NCOLUMNS / 5, NCOLUMNS / 4, NCOLUMNS / 3, NCOLUMNS / 2, NCOLUMNS - 1 };
        DoubleMatrix2D B = A.viewSelection(rowIndexes, colIndexes);
        assertEquals(rowIndexes.length, B.rows());
        assertEquals(colIndexes.length, B.columns());
        for (int r = 0; r < rowIndexes.length; r++) {
            for (int c = 0; c < colIndexes.length; c++) {
                assertEquals(A.getQuick(rowIndexes[r], colIndexes[c]), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewSorted() {
        DoubleMatrix2D B = A.viewSorted(1);
        for (int r = 0; r < NROWS - 1; r++) {
            assertTrue(B.getQuick(r + 1, 1) >= B.getQuick(r, 1));
        }
    }

    public void testViewStrides() {
        int rowStride = 3;
        int colStride = 5;
        DoubleMatrix2D B = A.viewStrides(rowStride, colStride);
        for (int r = 0; r < B.rows(); r++) {
            for (int c = 0; c < B.columns(); c++) {
                assertEquals(A.getQuick(r * rowStride, c * colStride), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testZMultDoubleMatrix1DDoubleMatrix1DDoubleDoubleBoolean() {
        DoubleMatrix1D y = new DenseDoubleMatrix1D(NCOLUMNS);
        for (int i = 0; i < y.size(); i++) {
            y.setQuick(i, Math.random());
        }
        double alpha = 3;
        double beta = 5;
        DoubleMatrix1D z = DoubleFactory1D.dense.random(NROWS);
        double[] expected = z.toArray();
        z = A.zMult(y, z, alpha, beta, false);
        for (int r = 0; r < NROWS; r++) {
            double s = 0;
            for (int c = 0; c < NCOLUMNS; c++) {
                s += A.getQuick(r, c) * y.getQuick(c);
            }
            expected[r] = s * alpha + expected[r] * beta;
        }

        for (int r = 0; r < NROWS; r++) {
            assertEquals(expected[r], z.getQuick(r), TOL);
        }
        //---
        z = null;
        z = A.zMult(y, z, alpha, beta, false);
        expected = new double[NROWS];
        for (int r = 0; r < NROWS; r++) {
            double s = 0;
            for (int c = 0; c < NCOLUMNS; c++) {
                s += A.getQuick(r, c) * y.getQuick(c);
            }
            expected[r] = s * alpha;
        }
        for (int r = 0; r < NROWS; r++) {
            assertEquals(expected[r], z.getQuick(r), TOL);
        }

        //transpose
        y = new DenseDoubleMatrix1D(NROWS);
        for (int i = 0; i < y.size(); i++) {
            y.setQuick(i, Math.random());
        }
        z = DoubleFactory1D.dense.random(NCOLUMNS);
        expected = z.toArray();
        z = A.zMult(y, z, alpha, beta, true);
        for (int r = 0; r < NCOLUMNS; r++) {
            double s = 0;
            for (int c = 0; c < NROWS; c++) {
                s += A.getQuick(c, r) * y.getQuick(c);
            }
            expected[r] = s * alpha + expected[r] * beta;
        }
        for (int r = 0; r < NCOLUMNS; r++) {
            assertEquals(expected[r], z.getQuick(r), TOL);
        }
        //---
        z = null;
        z = A.zMult(y, z, alpha, beta, true);
        expected = new double[NCOLUMNS];
        for (int r = 0; r < NCOLUMNS; r++) {
            double s = 0;
            for (int c = 0; c < NROWS; c++) {
                s += A.getQuick(c, r) * y.getQuick(c);
            }
            expected[r] = s * alpha;
        }
        for (int r = 0; r < NCOLUMNS; r++) {
            assertEquals(expected[r], z.getQuick(r), TOL);
        }
    }

    public void testZMultDoubleMatrix2DDoubleMatrix2DDoubleDoubleBooleanBoolean() {
        double alpha = 3;
        double beta = 5;
        DoubleMatrix2D C = DoubleFactory2D.dense.random(NROWS, NROWS);
        double[][] expected = C.toArray();
        C = A.zMult(Bt, C, alpha, beta, false, false);
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                double s = 0;
                for (int k = 0; k < NCOLUMNS; k++) {
                    s += A.getQuick(i, k) * Bt.getQuick(k, j);
                }
                expected[i][j] = s * alpha + expected[i][j] * beta;
            }
        }
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NROWS; c++) {
                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
            }
        }

        //---
        C = null;
        C = A.zMult(Bt, C, alpha, beta, false, false);
        expected = new double[NROWS][NROWS];
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                double s = 0;
                for (int k = 0; k < NCOLUMNS; k++) {
                    s += A.getQuick(i, k) * Bt.getQuick(k, j);
                }
                expected[i][j] = s * alpha;
            }
        }
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NROWS; c++) {
                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
            }
        }

        //transposeA
        C = DoubleFactory2D.dense.random(NCOLUMNS, NCOLUMNS);
        expected = C.toArray();
        C = A.zMult(B, C, alpha, beta, true, false);
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                double s = 0;
                for (int k = 0; k < NROWS; k++) {
                    s += A.getQuick(k, i) * B.getQuick(k, j);
                }
                expected[i][j] = s * alpha + expected[i][j] * beta;
            }
        }
        for (int r = 0; r < NCOLUMNS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
            }
        }
        //---
        C = null;
        C = A.zMult(B, C, alpha, beta, true, false);
        expected = new double[NCOLUMNS][NCOLUMNS];
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                double s = 0;
                for (int k = 0; k < NROWS; k++) {
                    s += A.getQuick(k, i) * B.getQuick(k, j);
                }
                expected[i][j] = s * alpha;
            }
        }
        for (int r = 0; r < NCOLUMNS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
            }
        }

        //transposeB
        C = DoubleFactory2D.dense.random(NROWS, NROWS);
        expected = C.toArray();
        C = A.zMult(B, C, alpha, beta, false, true);
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                double s = 0;
                for (int k = 0; k < NCOLUMNS; k++) {
                    s += A.getQuick(i, k) * B.getQuick(j, k);
                }
                expected[i][j] = s * alpha + expected[i][j] * beta;
            }
        }
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NROWS; c++) {
                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
            }
        }
        //---
        C = null;
        C = A.zMult(B, C, alpha, beta, false, true);
        expected = new double[NROWS][NROWS];
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                double s = 0;
                for (int k = 0; k < NCOLUMNS; k++) {
                    s += A.getQuick(i, k) * B.getQuick(j, k);
                }
                expected[i][j] = s * alpha;
            }
        }
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NROWS; c++) {
                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
            }
        }
        //transposeA and transposeB
        C = DoubleFactory2D.dense.random(NCOLUMNS, NCOLUMNS);
        expected = C.toArray();
        C = A.zMult(Bt, C, alpha, beta, true, true);
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                double s = 0;
                for (int k = 0; k < NROWS; k++) {
                    s += A.getQuick(k, i) * Bt.getQuick(j, k);
                }
                expected[i][j] = s * alpha + expected[i][j] * beta;
            }
        }
        for (int r = 0; r < NCOLUMNS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
            }
        }
        //---
        C = null;
        C = A.zMult(Bt, C, alpha, beta, true, true);
        expected = new double[NCOLUMNS][NCOLUMNS];
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                double s = 0;
                for (int k = 0; k < NROWS; k++) {
                    s += A.getQuick(k, i) * Bt.getQuick(j, k);
                }
                expected[i][j] = s * alpha;
            }
        }
        for (int r = 0; r < NCOLUMNS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
            }
        }

    }

    public void testZSum() {
        double sum = A.zSum();
        double expected = 0;
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                expected += A.getQuick(r, c);
            }
        }
        assertEquals(expected, sum, TOL);
    }

}
