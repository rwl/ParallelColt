package cern.colt.matrix.tfcomplex;

import java.util.ArrayList;

import junit.framework.TestCase;
import cern.colt.function.tfcomplex.FComplexProcedure;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tfloat.FloatFactory3D;
import cern.colt.matrix.tfloat.FloatMatrix3D;
import cern.jet.math.tfcomplex.FComplex;
import cern.jet.math.tfcomplex.FComplexFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public abstract class FComplexMatrix3DTest extends TestCase {

    /**
     * Matrix to test
     */
    protected FComplexMatrix3D A;

    /**
     * Matrix of the same size as A
     */
    protected FComplexMatrix3D B;

    protected int NSLICES = 5;

    protected int NROWS = 13;

    protected int NCOLUMNS = 17;

    protected float TOL = 1e-3f;

    /**
     * Constructor for FloatMatrix2DTest
     */
    public FComplexMatrix3DTest(String arg0) {
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
                    A.setQuick(s, r, c, new float[] { (float) Math.random(), (float) Math.random() });
                }
            }
        }

        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    B.setQuick(s, r, c, new float[] { (float) Math.random(), (float) Math.random() });
                }
            }
        }
    }

    @Override
    protected void tearDown() throws Exception {
        A = B = null;
    }

    protected void assertEquals(float[] expected, float[] actual, float tol) {
        for (int i = 0; i < actual.length; i++) {
            assertEquals(expected[i], actual[i], tol);
        }
    }

    public void testAggregateComplexComplexComplexFunctionComplexComplexFunction() {
        float[] actual = A.aggregate(FComplexFunctions.plus, FComplexFunctions.sqrt);
        float[] expected = new float[2];
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    expected = FComplex.plus(expected, FComplex.sqrt(A.getQuick(s, r, c)));
                }
            }
        }
        assertEquals(expected, actual, TOL);
    }

    public void testAggregateComplexMatrix3FComplexComplexComplexFunctionComplexComplexComplexFunction() {
        float[] actual = A.aggregate(B, FComplexFunctions.plus, FComplexFunctions.mult);
        float[] expected = new float[2];
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    expected = FComplex.plus(expected, FComplex.mult(A.getQuick(s, r, c), B.getQuick(s, r, c)));
                }
            }
        }
        assertEquals(expected, actual, TOL);
    }

    public void testAssignComplexComplexFunction() {
        FComplexMatrix3D Acopy = A.copy();
        A.assign(FComplexFunctions.acos);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(FComplex.acos(Acopy.getQuick(s, r, c)), A.getQuick(s, r, c), TOL);
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

    public void testAssignComplexMatrix3FComplexComplexComplexFunction() {
        FComplexMatrix3D Acopy = A.copy();
        A.assign(B, FComplexFunctions.div);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(FComplex.div(Acopy.getQuick(s, r, c), B.getQuick(s, r, c)), A.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testAssignComplexProcedureComplexComplexFunction() {
        FComplexMatrix3D Acopy = A.copy();
        A.assign(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, FComplexFunctions.tan);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    if (FComplex.abs(Acopy.getQuick(s, r, c)) > 3) {
                        assertEquals(FComplex.abs(Acopy.getQuick(s, r, c)), A.getQuick(s, r, c)[0], TOL);
                        assertEquals(0, A.getQuick(s, r, c)[1], TOL);
                    } else {
                        assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                    }
                }
            }
        }
    }

    public void testAssignComplexProcedureFloatArray() {
        FComplexMatrix3D Acopy = A.copy();
        float[] value = new float[] { (float) Math.random(), (float) Math.random() };
        A.assign(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, value);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    if (FComplex.abs(Acopy.getQuick(s, r, c)) > 3) {
                        assertEquals(value, A.getQuick(s, r, c), TOL);
                    } else {
                        assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                    }
                }
            }
        }
    }

    public void testAssignComplexRealFunction() {
        FComplexMatrix3D Acopy = A.copy();
        A.assign(FComplexFunctions.abs);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(FComplex.abs(A.getQuick(s, r, c)), A.getQuick(s, r, c)[0], TOL);
                    assertEquals(0, A.getQuick(s, r, c)[1], TOL);
                }
            }
        }
    }

    public void testAssignFloatArray() {
        float[] expected = new float[2 * (int) A.size()];
        for (int i = 0; i < 2 * A.size(); i++) {
            expected[i] = (float) Math.random();
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

    public void testAssignFloatArrayArrayArray() {
        float[][][] expected = new float[NSLICES][NROWS][2 * NCOLUMNS];
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < 2 * NCOLUMNS; c++) {
                    expected[s][r][c] = (float) Math.random();
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

    public void testAssignFloatFloat() {
        float[] value = new float[] { (float) Math.random(), (float) Math.random() };
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
        FComplexMatrix3D Acopy = A.copy();
        FloatMatrix3D Im = FloatFactory3D.dense.random(NSLICES, NROWS, NCOLUMNS);
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
        FComplexMatrix3D Acopy = A.copy();
        FloatMatrix3D Re = FloatFactory3D.dense.random(NSLICES, NROWS, NCOLUMNS);
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

    public void testEqualsFloatArray() {
        float[] value = new float[] { (float) Math.random(), (float) Math.random() };
        A.assign(value[0], value[1]);
        boolean eq = A.equals(value);
        assertTrue(eq);
        eq = A.equals(new float[] { value[0] + 1, value[1] + 1 });
        assertFalse(eq);
    }

    public void testEqualsObject() {
        boolean eq = A.equals(A);
        assertTrue(eq);
        eq = A.equals(B);
        assertFalse(eq);
    }

    public void testGetImaginaryPart() {
        FloatMatrix3D Im = A.getImaginaryPart();
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(A.getQuick(s, r, c)[1], Im.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testGetRealPart() {
        FloatMatrix3D Re = A.getRealPart();
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
        ArrayList<float[]> valueList = new ArrayList<float[]>();
        A.getNonZeros(sliceList, rowList, colList, valueList);
        assertEquals(A.size(), sliceList.size());
        assertEquals(A.size(), rowList.size());
        assertEquals(A.size(), colList.size());
        assertEquals(A.size(), valueList.size());
        int idx = 0;
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(A.getQuick(sliceList.get(idx), rowList.get(idx), colList.get(idx)),
                            valueList.get(idx), TOL);
                    idx++;
                }
            }
        }
    }

    public void testToArray() {
        float[][][] array = A.toArray();
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
        FComplexMatrix1D B = A.vectorize();
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
        FComplexMatrix2D B = A.viewColumn(NCOLUMNS / 2);
        assertEquals(NSLICES, B.rows());
        assertEquals(NROWS, B.columns());
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                assertEquals(A.getQuick(s, r, NCOLUMNS / 2), B.getQuick(s, r), TOL);
            }
        }
    }

    public void testViewColumnFlip() {
        FComplexMatrix3D B = A.viewColumnFlip();
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
        FComplexMatrix3D B = A.viewDice(2, 1, 0);
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
        FComplexMatrix3D B = A.viewPart(NSLICES / 2, NROWS / 2, NCOLUMNS / 2, NSLICES / 3, NROWS / 3, NCOLUMNS / 3);
        for (int s = 0; s < NSLICES / 3; s++) {
            for (int r = 0; r < NROWS / 3; r++) {
                for (int c = 0; c < NCOLUMNS / 3; c++) {
                    assertEquals(A.getQuick(NSLICES / 2 + s, NROWS / 2 + r, NCOLUMNS / 2 + c), B.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testViewRow() {
        FComplexMatrix2D B = A.viewRow(NROWS / 2);
        assertEquals(NSLICES, B.rows());
        assertEquals(NCOLUMNS, B.columns());
        for (int s = 0; s < NSLICES; s++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(s, NROWS / 2, c), B.getQuick(s, c), TOL);
            }
        }
    }

    public void testViewRowFlip() {
        FComplexMatrix3D B = A.viewRowFlip();
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
        final float[] value = new float[] { 2, 3 };
        A.setQuick(NSLICES / 2, NROWS / 2, 0, value);
        FComplexMatrix3D B = A.viewSelection(new FComplexMatrix2DProcedure() {
            public boolean apply(FComplexMatrix2D element) {
                return FComplex.isEqual(element.getQuick(NROWS / 2, 0), value, TOL);

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
        int[] colIndexes = new int[] { NCOLUMNS / 6, NCOLUMNS / 5, NCOLUMNS / 4, NCOLUMNS / 3, NCOLUMNS / 2,
                NCOLUMNS - 1 };
        FComplexMatrix3D B = A.viewSelection(sliceIndexes, rowIndexes, colIndexes);
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
        FComplexMatrix2D B = A.viewSlice(NSLICES / 2);
        assertEquals(NROWS, B.rows());
        assertEquals(NCOLUMNS, B.columns());
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(NSLICES / 2, r, c), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewSliceFlip() {
        FComplexMatrix3D B = A.viewSliceFlip();
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
        FComplexMatrix3D B = A.viewStrides(sliceStride, rowStride, colStride);
        for (int s = 0; s < B.slices(); s++) {
            for (int r = 0; r < B.rows(); r++) {
                for (int c = 0; c < B.columns(); c++) {
                    assertEquals(A.getQuick(s * sliceStride, r * rowStride, c * colStride), B.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testZSum() {
        float[] sum = A.zSum();
        float[] expected = new float[2];
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    expected = FComplex.plus(expected, A.getQuick(s, r, c));
                }
            }
        }
        assertEquals(expected, sum, TOL);
    }

}
