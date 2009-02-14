package cern.colt.matrix.tdcomplex;

import java.util.ArrayList;

import junit.framework.TestCase;
import cern.colt.function.tdcomplex.DComplexProcedure;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tdouble.DoubleFactory3D;
import cern.colt.matrix.tdouble.DoubleMatrix3D;
import cern.jet.math.tdcomplex.DComplex;
import cern.jet.math.tdcomplex.DComplexFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public abstract class DComplexMatrix3DTest extends TestCase {

    /**
     * Matrix to test
     */
    protected DComplexMatrix3D A;

    /**
     * Matrix of the same size as A
     */
    protected DComplexMatrix3D B;

    protected int NSLICES = 5;

    protected int NROWS = 13;

    protected int NCOLUMNS = 17;

    protected double TOL = 1e-10;

    /**
     * Constructor for DoubleMatrix2DTest
     */
    public DComplexMatrix3DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void setUp() throws Exception {
        createMatrices();
        populateMatrices();
    }

    protected abstract void createMatrices() throws Exception;

    protected void populateMatrices() {
        ConcurrencyUtils.setThreadsBeginN_3D(1);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    A.setQuick(s, r, c, new double[] { Math.random(), Math.random() });
                }
            }
        }

        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    B.setQuick(s, r, c, new double[] { Math.random(), Math.random() });
                }
            }
        }
    }

    @Override
    protected void tearDown() throws Exception {
        A = B = null;
    }

    protected void assertEquals(double[] expected, double[] actual, double tol) {
        for (int i = 0; i < actual.length; i++) {
            assertEquals(expected[i], actual[i], tol);
        }
    }

    public void testAggregateComplexComplexComplexFunctionComplexComplexFunction() {
        double[] actual = A.aggregate(DComplexFunctions.plus, DComplexFunctions.sqrt);
        double[] expected = new double[2];
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    expected = DComplex.plus(expected, DComplex.sqrt(A.getQuick(s, r, c)));
                }
            }
        }
        assertEquals(expected, actual, TOL);
    }

    public void testAggregateComplexMatrix3DComplexComplexComplexFunctionComplexComplexComplexFunction() {
        double[] actual = A.aggregate(B, DComplexFunctions.plus, DComplexFunctions.mult);
        double[] expected = new double[2];
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    expected = DComplex.plus(expected, DComplex.mult(A.getQuick(s, r, c), B.getQuick(s, r, c)));
                }
            }
        }
        assertEquals(expected, actual, TOL);
    }

    public void testAssignComplexComplexFunction() {
        DComplexMatrix3D Acopy = A.copy();
        A.assign(DComplexFunctions.acos);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(DComplex.acos(Acopy.getQuick(s, r, c)), A.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testAssignComplexMatrix3D() {
        A.assign(B);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(B.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testAssignComplexMatrix3DComplexComplexComplexFunction() {
        DComplexMatrix3D Acopy = A.copy();
        A.assign(B, DComplexFunctions.div);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(DComplex.div(Acopy.getQuick(s, r, c), B.getQuick(s, r, c)), A.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testAssignComplexProcedureComplexComplexFunction() {
        DComplexMatrix3D Acopy = A.copy();
        A.assign(new DComplexProcedure() {
            public boolean apply(double[] element) {
                if (DComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, DComplexFunctions.tan);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    if (DComplex.abs(Acopy.getQuick(s, r, c)) > 3) {
                        assertEquals(DComplex.abs(Acopy.getQuick(s, r, c)), A.getQuick(s, r, c)[0], TOL);
                        assertEquals(0, A.getQuick(s, r, c)[1], TOL);
                    } else {
                        assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                    }
                }
            }
        }
    }

    public void testAssignComplexProcedureDoubleArray() {
        DComplexMatrix3D Acopy = A.copy();
        double[] value = new double[] { Math.random(), Math.random() };
        A.assign(new DComplexProcedure() {
            public boolean apply(double[] element) {
                if (DComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, value);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    if (DComplex.abs(Acopy.getQuick(s, r, c)) > 3) {
                        assertEquals(value, A.getQuick(s, r, c), TOL);
                    } else {
                        assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                    }
                }
            }
        }
    }

    public void testAssignComplexRealFunction() {
        DComplexMatrix3D Acopy = A.copy();
        A.assign(DComplexFunctions.abs);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(DComplex.abs(A.getQuick(s, r, c)), A.getQuick(s, r, c)[0], TOL);
                    assertEquals(0, A.getQuick(s, r, c)[1], TOL);
                }
            }
        }
    }

    public void testAssignDoubleArray() {
        double[] expected = new double[2 * A.size()];
        for (int i = 0; i < 2 * A.size(); i++) {
            expected[i] = Math.random();
        }
        A.assign(expected);
        int idx = 0;
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(expected[idx], A.getQuick(s, r, c)[0], TOL);
                    assertEquals(expected[idx + 1], A.getQuick(s, r, c)[1], TOL);
                    idx += 2;
                }
            }
        }
    }

    public void testAssignDoubleArrayArrayArray() {
        double[][][] expected = new double[NSLICES][NROWS][2 * NCOLUMNS];
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < 2 * NCOLUMNS; c++) {
                    expected[s][r][c] = Math.random();
                }
            }
        }
        A.assign(expected);
        for (int s = 0; s < NSLICES; s++) {
            assertTrue(NROWS == expected[s].length);
            for (int r = 0; r < NROWS; r++) {
                assertTrue(2 * NCOLUMNS == expected[s][r].length);
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(expected[s][r][2 * c], A.getQuick(s, r, c)[0], TOL);
                    assertEquals(expected[s][r][2 * c + 1], A.getQuick(s, r, c)[1], TOL);
                }
            }
        }
    }

    public void testAssignDoubleDouble() {
        double[] value = new double[] { Math.random(), Math.random() };
        A.assign(value[0], value[1]);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(value, A.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testAssignImaginary() {
        DComplexMatrix3D Acopy = A.copy();
        DoubleMatrix3D Im = DoubleFactory3D.dense.random(NSLICES, NROWS, NCOLUMNS);
        A.assignImaginary(Im);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(Im.getQuick(s, r, c), A.getQuick(s, r, c)[1], TOL);
                    assertEquals(Acopy.getQuick(s, r, c)[0], A.getQuick(s, r, c)[0], TOL);
                }
            }
        }
    }

    public void testAssignReal() {
        DComplexMatrix3D Acopy = A.copy();
        DoubleMatrix3D Re = DoubleFactory3D.dense.random(NSLICES, NROWS, NCOLUMNS);
        A.assignReal(Re);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(Re.getQuick(s, r, c), A.getQuick(s, r, c)[0], TOL);
                    assertEquals(Acopy.getQuick(s, r, c)[1], A.getQuick(s, r, c)[1], TOL);
                }
            }
        }
    }

    public void testCardinality() {
        int card = A.cardinality();
        assertEquals(A.size(), card);
    }

    public void testEqualsDoubleArray() {
        double[] value = new double[] { Math.random(), Math.random() };
        A.assign(value[0], value[1]);
        boolean eq = A.equals(value);
        assertTrue(eq);
        eq = A.equals(new double[] { value[0] + 1, value[1] + 1 });
        assertFalse(eq);
    }

    public void testEqualsObject() {
        boolean eq = A.equals(A);
        assertTrue(eq);
        eq = A.equals(B);
        assertFalse(eq);
    }

    public void testGetImaginaryPart() {
        DoubleMatrix3D Im = A.getImaginaryPart();
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(A.getQuick(s, r, c)[1], Im.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testGetRealPart() {
        DoubleMatrix3D Re = A.getRealPart();
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(A.getQuick(s, r, c)[0], Re.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testGetNonZeros() {
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList colList = new IntArrayList();
        ArrayList<double[]> valueList = new ArrayList<double[]>();
        A.getNonZeros(sliceList, rowList, colList, valueList);
        assertEquals(A.size(), sliceList.size());
        assertEquals(A.size(), rowList.size());
        assertEquals(A.size(), colList.size());
        assertEquals(A.size(), valueList.size());
        int idx = 0;
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(A.getQuick(sliceList.get(idx), rowList.get(idx), colList.get(idx)), valueList.get(idx), TOL);
                    idx++;
                }
            }
        }
    }

    public void testToArray() {
        double[][][] array = A.toArray();
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(A.getQuick(s, r, c)[0], array[s][r][2 * c], TOL);
                    assertEquals(A.getQuick(s, r, c)[1], array[s][r][2 * c + 1], TOL);
                }
            }
        }
    }

    public void testVectorize() {
        DComplexMatrix1D B = A.vectorize();
        int idx = 0;
        for (int s = 0; s < NSLICES; s++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                for (int r = 0; r < NROWS; r++) {
                    assertEquals(A.getQuick(s, r, c), B.getQuick(idx++), TOL);
                }
            }
        }
    }

    public void testViewColumn() {
        DComplexMatrix2D B = A.viewColumn(NCOLUMNS / 2);
        assertEquals(NSLICES, B.rows());
        assertEquals(NROWS, B.columns());
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                assertEquals(A.getQuick(s, r, NCOLUMNS / 2), B.getQuick(s, r), TOL);
            }
        }
    }

    public void testViewColumnFlip() {
        DComplexMatrix3D B = A.viewColumnFlip();
        assertEquals(A.size(), B.size());
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(A.getQuick(s, r, NCOLUMNS - 1 - c), B.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testViewDice() {
        DComplexMatrix3D B = A.viewDice(2, 1, 0);
        assertEquals(A.slices(), B.columns());
        assertEquals(A.rows(), B.rows());
        assertEquals(A.columns(), B.slices());
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(A.getQuick(s, r, c), B.getQuick(c, r, s), TOL);
                }
            }
        }
    }

    public void testViewPart() {
        DComplexMatrix3D B = A.viewPart(NSLICES / 2, NROWS / 2, NCOLUMNS / 2, NSLICES / 3, NROWS / 3, NCOLUMNS / 3);
        for (int s = 0; s < NSLICES / 3; s++) {
            for (int r = 0; r < NROWS / 3; r++) {
                for (int c = 0; c < NCOLUMNS / 3; c++) {
                    assertEquals(A.getQuick(NSLICES / 2 + s, NROWS / 2 + r, NCOLUMNS / 2 + c), B.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testViewRow() {
        DComplexMatrix2D B = A.viewRow(NROWS / 2);
        assertEquals(NSLICES, B.rows());
        assertEquals(NCOLUMNS, B.columns());
        for (int s = 0; s < NSLICES; s++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(s, NROWS / 2, c), B.getQuick(s, c), TOL);
            }
        }
    }

    public void testViewRowFlip() {
        DComplexMatrix3D B = A.viewRowFlip();
        assertEquals(A.size(), B.size());
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(A.getQuick(s, NROWS - 1 - r, c), B.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testViewSelectionComplexMatrix2DProcedure() {
        final double[] value = new double[] { 2, 3 };
        A.setQuick(NSLICES / 2, NROWS / 2, 0, value);
        DComplexMatrix3D B = A.viewSelection(new DComplexMatrix2DProcedure() {
            public boolean apply(DComplexMatrix2D element) {
                return DComplex.isEqual(element.getQuick(NROWS / 2, 0), value, TOL);

            }
        });
        assertEquals(1, B.slices());
        assertEquals(A.rows(), B.rows());
        assertEquals(A.columns(), B.columns());
        assertEquals(A.getQuick(NSLICES / 2, NROWS / 2, 0), B.getQuick(0, NROWS / 2, 0), TOL);
    }

    public void testViewSelectionIntArrayIntArrayIntArray() {
        int[] sliceIndexes = new int[] { NSLICES / 2, NSLICES / 3 };
        int[] rowIndexes = new int[] { NROWS / 6, NROWS / 5, NROWS / 4, NROWS / 3, NROWS / 2 };
        int[] colIndexes = new int[] { NCOLUMNS / 6, NCOLUMNS / 5, NCOLUMNS / 4, NCOLUMNS / 3, NCOLUMNS / 2, NCOLUMNS - 1 };
        DComplexMatrix3D B = A.viewSelection(sliceIndexes, rowIndexes, colIndexes);
        assertEquals(sliceIndexes.length, B.slices());
        assertEquals(rowIndexes.length, B.rows());
        assertEquals(colIndexes.length, B.columns());
        for (int s = 0; s < sliceIndexes.length; s++) {
            for (int r = 0; r < rowIndexes.length; r++) {
                for (int c = 0; c < colIndexes.length; c++) {
                    assertEquals(A.getQuick(sliceIndexes[s], rowIndexes[r], colIndexes[c]), B.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testViewSlice() {
        DComplexMatrix2D B = A.viewSlice(NSLICES / 2);
        assertEquals(NROWS, B.rows());
        assertEquals(NCOLUMNS, B.columns());
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(NSLICES / 2, r, c), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewSliceFlip() {
        DComplexMatrix3D B = A.viewSliceFlip();
        assertEquals(A.size(), B.size());
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(A.getQuick(NSLICES - 1 - s, r, c), B.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testViewStrides() {
        int sliceStride = 2;
        int rowStride = 2;
        int colStride = 2;
        DComplexMatrix3D B = A.viewStrides(sliceStride, rowStride, colStride);
        for (int s = 0; s < B.slices(); s++) {
            for (int r = 0; r < B.rows(); r++) {
                for (int c = 0; c < B.columns(); c++) {
                    assertEquals(A.getQuick(s * sliceStride, r * rowStride, c * colStride), B.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testZSum() {
        double[] sum = A.zSum();
        double[] expected = new double[2];
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    expected = DComplex.plus(expected, A.getQuick(s, r, c));
                }
            }
        }
        assertEquals(expected, sum, TOL);
    }

}
