package cern.colt.matrix.tdouble;

import java.util.Random;

import junit.framework.TestCase;
import cern.colt.function.tdouble.DoubleProcedure;
import cern.colt.function.tdouble.IntIntDoubleFunction;
import cern.colt.list.tdouble.DoubleArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.jet.math.tdouble.DoubleFunctions;
import edu.emory.mathcs.utils.pc.ConcurrencyUtils;

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

    protected static final Random random = new Random(0);

    /**
     * Constructor for DoubleMatrix2DTest
     */
    public DoubleMatrix2DTest(String arg0) {
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
                A.setQuick(r, c, random.nextDouble());
            }
        }

        for (int r = 0; r < B.rows(); r++) {
            for (int c = 0; c < B.columns(); c++) {
                B.setQuick(r, c, random.nextDouble());
            }
        }

        for (int r = 0; r < Bt.rows(); r++) {
            for (int c = 0; c < Bt.columns(); c++) {
                Bt.setQuick(r, c, random.nextDouble());
            }
        }
    }

    protected void tearDown() throws Exception {
        A = B = Bt = null;
    }

    //    public void testToString() {
    //        System.out.println(A.toString());
    //    }

    public void testAggregateDoubleDoubleFunctionDoubleFunction() {
        double expected = 0;
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
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
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
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
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                rowList.add(r);
                columnList.add(c);
            }
        }
        double expected = 0;
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                double elem = A.getQuick(r, c);
                expected += elem * elem;
            }
        }
        double result = A.aggregate(DoubleFunctions.plus, DoubleFunctions.square, rowList, columnList);
        assertEquals(expected, result, TOL);
    }

    public void testAggregateDoubleMatrix2DDoubleDoubleFunctionDoubleDoubleFunction() {
        double expected = 0;
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
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
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++)
                assertEquals(value, A.getQuick(r, c), TOL);
        }
    }

    public void testAssignDoubleArrayArray() {
        double[][] expected = new double[A.rows()][A.columns()];
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                expected[r][c] = Math.random();
            }
        }
        A.assign(expected);
        for (int r = 0; r < A.rows(); r++) {
            assertTrue(A.columns() == expected[r].length);
            for (int c = 0; c < A.columns(); c++)
                assertEquals(expected[r][c], A.getQuick(r, c), TOL);
        }
    }

    public void testAssignDoubleFunction() {
        DoubleMatrix2D Acopy = A.copy();
        A.assign(DoubleFunctions.acos);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                double expected = Math.acos(Acopy.getQuick(r, c));
                assertEquals(expected, A.getQuick(r, c), TOL);
            }
        }
    }

    public void testAssignDoubleMatrix2D() {
        A.assign(B);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++)
                assertEquals(B.getQuick(r, c), A.getQuick(r, c), TOL);
        }
    }

    public void testAssignDoubleMatrix2DDoubleDoubleFunction() {
        DoubleMatrix2D Acopy = A.copy();
        A.assign(B, DoubleFunctions.plus);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(Acopy.getQuick(r, c) + B.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testAssignDoubleMatrix2DDoubleDoubleFunctionIntArrayListIntArrayList() {
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                rowList.add(r);
                columnList.add(c);
            }
        }
        DoubleMatrix2D Acopy = A.copy();
        A.assign(B, DoubleFunctions.div, rowList, columnList);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
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
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
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
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
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
        assertEquals(A.rows() * A.columns(), card);
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
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(Math.sqrt(Acopy.getQuick(r, c)), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testMaxLocation() {
        A.assign(0);
        A.setQuick(A.rows() / 3, A.columns() / 3, 0.7);
        A.setQuick(A.rows() / 2, A.columns() / 2, 0.1);
        double[] maxAndLoc = A.getMaxLocation();
        assertEquals(0.7, maxAndLoc[0], TOL);
        assertEquals(A.rows() / 3, (int) maxAndLoc[1]);
        assertEquals(A.columns() / 3, (int) maxAndLoc[2]);
    }

    public void testMinLocation() {
        A.assign(0);
        A.setQuick(A.rows() / 3, A.columns() / 3, -0.7);
        A.setQuick(A.rows() / 2, A.columns() / 2, -0.1);
        double[] minAndLoc = A.getMinLocation();
        assertEquals(-0.7, minAndLoc[0], TOL);
        assertEquals(A.rows() / 3, (int) minAndLoc[1]);
        assertEquals(A.columns() / 3, (int) minAndLoc[2]);
    }

    public void testGetNegativeValues() {
        A.assign(0);
        A.setQuick(A.rows() / 3, A.columns() / 3, -0.7);
        A.setQuick(A.rows() / 2, A.columns() / 2, -0.1);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        A.getNegativeValues(rowList, columnList, valueList);
        assertEquals(2, rowList.size());
        assertEquals(2, columnList.size());
        assertEquals(2, valueList.size());
        assertTrue(rowList.contains(A.rows() / 3));
        assertTrue(rowList.contains(A.rows() / 2));
        assertTrue(columnList.contains(A.columns() / 3));
        assertTrue(columnList.contains(A.columns() / 2));
        assertTrue(valueList.contains(-0.7));
        assertTrue(valueList.contains(-0.1));
    }

    public void testGetNonZeros() {
        A.assign(0);
        A.setQuick(A.rows() / 3, A.columns() / 3, 0.7);
        A.setQuick(A.rows() / 2, A.columns() / 2, 0.1);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        A.getNonZeros(rowList, columnList, valueList);
        assertEquals(2, rowList.size());
        assertEquals(2, columnList.size());
        assertEquals(2, valueList.size());
        assertTrue(rowList.contains(A.rows() / 3));
        assertTrue(rowList.contains(A.rows() / 2));
        assertTrue(columnList.contains(A.columns() / 3));
        assertTrue(columnList.contains(A.columns() / 2));
        assertTrue(valueList.contains(0.7));
        assertTrue(valueList.contains(0.1));
    }

    public void testGetPositiveValues() {
        A.assign(0);
        A.setQuick(A.rows() / 3, A.columns() / 3, 0.7);
        A.setQuick(A.rows() / 2, A.columns() / 2, 0.1);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        A.getPositiveValues(rowList, columnList, valueList);
        assertEquals(2, rowList.size());
        assertEquals(2, columnList.size());
        assertEquals(2, valueList.size());
        assertTrue(rowList.contains(A.rows() / 3));
        assertTrue(rowList.contains(A.rows() / 2));
        assertTrue(columnList.contains(A.columns() / 3));
        assertTrue(columnList.contains(A.columns() / 2));
        assertTrue(valueList.contains(0.7));
        assertTrue(valueList.contains(0.1));
    }

    public void testToArray() {
        double[][] array = A.toArray();
        assertTrue(A.rows() == array.length);
        for (int r = 0; r < A.rows(); r++) {
            assertTrue(A.columns() == array[r].length);
            for (int c = 0; c < A.columns(); c++)
                assertEquals(0, Math.abs(array[r][c] - A.getQuick(r, c)), TOL);
        }
    }

    public void testVectorize() {
        DoubleMatrix1D Avec = A.vectorize();
        int idx = 0;
        for (int c = 0; c < A.columns(); c++) {
            for (int r = 0; r < A.rows(); r++) {
                assertEquals(A.getQuick(r, c), Avec.getQuick(idx++), TOL);
            }
        }
    }

    public void testViewColumn() {
        DoubleMatrix1D col = A.viewColumn(A.columns() / 2);
        assertEquals(A.rows(), col.size());
        for (int r = 0; r < A.rows(); r++) {
            assertEquals(A.getQuick(r, A.columns() / 2), col.getQuick(r), TOL);
        }
    }

    public void testViewColumnFlip() {
        DoubleMatrix2D B = A.viewColumnFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(r, A.columns() - 1 - c), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewDice() {
        DoubleMatrix2D B = A.viewDice();
        assertEquals(A.rows(), B.columns());
        assertEquals(A.columns(), B.rows());
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(c, r), TOL);
            }
        }
    }

    public void testViewPart() {
        DoubleMatrix2D B = A.viewPart(A.rows() / 2, A.columns() / 2, A.rows() / 3, A.columns() / 3);
        assertEquals(A.rows() / 3, B.rows());
        assertEquals(A.columns() / 3, B.columns());
        for (int r = 0; r < A.rows() / 3; r++) {
            for (int c = 0; c < A.columns() / 3; c++) {
                assertEquals(A.getQuick(A.rows() / 2 + r, A.columns() / 2 + c), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewRow() {
        DoubleMatrix1D B = A.viewRow(A.rows() / 2);
        assertEquals(A.columns(), B.size());
        for (int r = 0; r < A.columns(); r++) {
            assertEquals(A.getQuick(A.rows() / 2, r), B.getQuick(r), TOL);
        }
    }

    public void testViewRowFlip() {
        DoubleMatrix2D B = A.viewRowFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(A.rows() - 1 - r, c), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewSelectionDoubleMatrix1DProcedure() {
        final double value = 2;
        A.assign(0);
        A.setQuick(A.rows() / 4, 0, value);
        A.setQuick(A.rows() / 2, 0, value);
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
        assertEquals(A.columns(), B.columns());
        assertEquals(A.getQuick(A.rows() / 4, 0), B.getQuick(0, 0), TOL);
        assertEquals(A.getQuick(A.rows() / 2, 0), B.getQuick(1, 0), TOL);
    }

    public void testViewSelectionIntArrayIntArray() {
        int[] rowIndexes = new int[] { A.rows() / 6, A.rows() / 5, A.rows() / 4, A.rows() / 3, A.rows() / 2 };
        int[] colIndexes = new int[] { A.columns() / 6, A.columns() / 5, A.columns() / 4, A.columns() / 3,
                A.columns() / 2, A.columns() - 1 };
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
        for (int r = 0; r < A.rows() - 1; r++) {
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
        DoubleMatrix1D y = new DenseDoubleMatrix1D(A.columns());
        for (int i = 0; i < y.size(); i++) {
            y.setQuick(i, Math.random());
        }
        double alpha = 3;
        double beta = 5;
        DoubleMatrix1D z = DoubleFactory1D.dense.random(A.rows());
        double[] expected = z.toArray();
        z = A.zMult(y, z, alpha, beta, false);
        for (int r = 0; r < A.rows(); r++) {
            double s = 0;
            for (int c = 0; c < A.columns(); c++) {
                s += A.getQuick(r, c) * y.getQuick(c);
            }
            expected[r] = s * alpha + expected[r] * beta;
        }

        for (int r = 0; r < A.rows(); r++) {
            assertEquals(expected[r], z.getQuick(r), TOL);
        }
        //---
        z = null;
        z = A.zMult(y, z, alpha, beta, false);
        expected = new double[A.rows()];
        for (int r = 0; r < A.rows(); r++) {
            double s = 0;
            for (int c = 0; c < A.columns(); c++) {
                s += A.getQuick(r, c) * y.getQuick(c);
            }
            expected[r] = s * alpha;
        }
        for (int r = 0; r < A.rows(); r++) {
            assertEquals(expected[r], z.getQuick(r), TOL);
        }

        //transpose
        y = new DenseDoubleMatrix1D(A.rows());
        for (int i = 0; i < y.size(); i++) {
            y.setQuick(i, Math.random());
        }
        z = DoubleFactory1D.dense.random(A.columns());
        expected = z.toArray();
        z = A.zMult(y, z, alpha, beta, true);
        for (int r = 0; r < A.columns(); r++) {
            double s = 0;
            for (int c = 0; c < A.rows(); c++) {
                s += A.getQuick(c, r) * y.getQuick(c);
            }
            expected[r] = s * alpha + expected[r] * beta;
        }
        for (int r = 0; r < A.columns(); r++) {
            assertEquals(expected[r], z.getQuick(r), TOL);
        }
        //---
        z = null;
        z = A.zMult(y, z, alpha, beta, true);
        expected = new double[A.columns()];
        for (int r = 0; r < A.columns(); r++) {
            double s = 0;
            for (int c = 0; c < A.rows(); c++) {
                s += A.getQuick(c, r) * y.getQuick(c);
            }
            expected[r] = s * alpha;
        }
        for (int r = 0; r < A.columns(); r++) {
            assertEquals(expected[r], z.getQuick(r), TOL);
        }
    }

    public void testZMultDoubleMatrix2DDoubleMatrix2DDoubleDoubleBooleanBoolean() {
        double alpha = 3;
        double beta = 5;
        DoubleMatrix2D C = DoubleFactory2D.dense.random(A.rows(), A.rows());
        double[][] expected = C.toArray();
        C = A.zMult(Bt, C, alpha, beta, false, false);
        for (int j = 0; j < A.rows(); j++) {
            for (int i = 0; i < A.rows(); i++) {
                double s = 0;
                for (int k = 0; k < A.columns(); k++) {
                    s += A.getQuick(i, k) * Bt.getQuick(k, j);
                }
                expected[i][j] = s * alpha + expected[i][j] * beta;
            }
        }
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.rows(); c++) {
                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
            }
        }

        //---
        C = null;
        C = A.zMult(Bt, C, alpha, beta, false, false);
        expected = new double[A.rows()][A.rows()];
        for (int j = 0; j < A.rows(); j++) {
            for (int i = 0; i < A.rows(); i++) {
                double s = 0;
                for (int k = 0; k < A.columns(); k++) {
                    s += A.getQuick(i, k) * Bt.getQuick(k, j);
                }
                expected[i][j] = s * alpha;
            }
        }
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.rows(); c++) {
                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
            }
        }

        //transposeA
        C = DoubleFactory2D.dense.random(A.columns(), A.columns());
        expected = C.toArray();
        C = A.zMult(B, C, alpha, beta, true, false);
        for (int j = 0; j < A.columns(); j++) {
            for (int i = 0; i < A.columns(); i++) {
                double s = 0;
                for (int k = 0; k < A.rows(); k++) {
                    s += A.getQuick(k, i) * B.getQuick(k, j);
                }
                expected[i][j] = s * alpha + expected[i][j] * beta;
            }
        }
        for (int r = 0; r < A.columns(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
            }
        }
        //---
        C = null;
        C = A.zMult(B, C, alpha, beta, true, false);
        expected = new double[A.columns()][A.columns()];
        for (int j = 0; j < A.columns(); j++) {
            for (int i = 0; i < A.columns(); i++) {
                double s = 0;
                for (int k = 0; k < A.rows(); k++) {
                    s += A.getQuick(k, i) * B.getQuick(k, j);
                }
                expected[i][j] = s * alpha;
            }
        }
        for (int r = 0; r < A.columns(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
            }
        }

        //transposeB
        C = DoubleFactory2D.dense.random(A.rows(), A.rows());
        expected = C.toArray();
        C = A.zMult(B, C, alpha, beta, false, true);
        for (int j = 0; j < A.rows(); j++) {
            for (int i = 0; i < A.rows(); i++) {
                double s = 0;
                for (int k = 0; k < A.columns(); k++) {
                    s += A.getQuick(i, k) * B.getQuick(j, k);
                }
                expected[i][j] = s * alpha + expected[i][j] * beta;
            }
        }
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.rows(); c++) {
                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
            }
        }
        //---
        C = null;
        C = A.zMult(B, C, alpha, beta, false, true);
        expected = new double[A.rows()][A.rows()];
        for (int j = 0; j < A.rows(); j++) {
            for (int i = 0; i < A.rows(); i++) {
                double s = 0;
                for (int k = 0; k < A.columns(); k++) {
                    s += A.getQuick(i, k) * B.getQuick(j, k);
                }
                expected[i][j] = s * alpha;
            }
        }
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.rows(); c++) {
                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
            }
        }
        //transposeA and transposeB
        C = DoubleFactory2D.dense.random(A.columns(), A.columns());
        expected = C.toArray();
        C = A.zMult(Bt, C, alpha, beta, true, true);
        for (int j = 0; j < A.columns(); j++) {
            for (int i = 0; i < A.columns(); i++) {
                double s = 0;
                for (int k = 0; k < A.rows(); k++) {
                    s += A.getQuick(k, i) * Bt.getQuick(j, k);
                }
                expected[i][j] = s * alpha + expected[i][j] * beta;
            }
        }
        for (int r = 0; r < A.columns(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
            }
        }
        //---
        C = null;
        C = A.zMult(Bt, C, alpha, beta, true, true);
        expected = new double[A.columns()][A.columns()];
        for (int j = 0; j < A.columns(); j++) {
            for (int i = 0; i < A.columns(); i++) {
                double s = 0;
                for (int k = 0; k < A.rows(); k++) {
                    s += A.getQuick(k, i) * Bt.getQuick(j, k);
                }
                expected[i][j] = s * alpha;
            }
        }
        for (int r = 0; r < A.columns(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
            }
        }

    }

    public void testZSum() {
        double sum = A.zSum();
        double expected = 0;
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                expected += A.getQuick(r, c);
            }
        }
        assertEquals(expected, sum, TOL);
    }

}
