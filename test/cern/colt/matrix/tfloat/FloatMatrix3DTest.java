package cern.colt.matrix.tfloat;

import junit.framework.TestCase;
import cern.colt.function.tfloat.FloatProcedure;
import cern.colt.list.tfloat.FloatArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.jet.math.tfloat.FloatFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public abstract class FloatMatrix3DTest extends TestCase {
    /**
     * Matrix to test
     */
    protected FloatMatrix3D A;

    /**
     * Matrix of the same size as A
     */
    protected FloatMatrix3D B;

    protected int NSLICES = 5;

    protected int NROWS = 13;

    protected int NCOLUMNS = 17;

    protected float TOL = 1e-3f;

    /**
     * Constructor for FloatMatrix2DTest
     */
    public FloatMatrix3DTest(String arg0) {
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
                    A.setQuick(s, r, c, (float) Math.random());
                }
            }
        }

        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    B.setQuick(s, r, c, (float) Math.random());
                }
            }
        }
    }

    @Override
    protected void tearDown() throws Exception {
        A = B = null;
    }

    public void testAggregateFloatFloatFunctionFloatFunction() {
        float expected = 0;
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    float elem = A.getQuick(s, r, c);
                    expected += elem * elem;
                }
            }
        }
        float result = A.aggregate(FloatFunctions.plus, FloatFunctions.square);
        assertEquals(expected, result, TOL);
    }

    public void testAggregateFloatFloatFunctionFloatFunctionFloatProcedure() {
        FloatProcedure procedure = new FloatProcedure() {
            public boolean apply(float element) {
                if (Math.abs(element) > 0.2) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        float expected = 0;
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    float elem = A.getQuick(s, r, c);
                    if (Math.abs(elem) > 0.2) {
                        expected += elem * elem;
                    }
                }
            }
        }

        float result = A.aggregate(FloatFunctions.plus, FloatFunctions.square, procedure);
        assertEquals(expected, result, TOL);
    }

    public void testAggregateFloatFloatFunctionFloatFunctionIntArrayListIntArrayListIntArrayList() {
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    sliceList.add(s);
                    rowList.add(r);
                    columnList.add(c);
                }
            }
        }
        float expected = 0;
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    float elem = A.getQuick(s, r, c);
                    expected += elem * elem;
                }
            }
        }
        float result = A.aggregate(FloatFunctions.plus, FloatFunctions.square, sliceList, rowList, columnList);
        assertEquals(expected, result, TOL);
    }

    public void testAggregateFloatMatrix2DFloatFloatFunctionFloatFloatFunction() {
        float expected = 0;
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    float elemA = A.getQuick(s, r, c);
                    float elemB = B.getQuick(s, r, c);
                    expected += elemA * elemB;
                }
            }
        }
        float result = A.aggregate(B, FloatFunctions.plus, FloatFunctions.mult);
        assertEquals(expected, result, TOL);
    }

    public void testAssignFloat() {
        float value = (float) Math.random();
        A.assign(value);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++)
                    assertEquals(value, A.getQuick(s, r, c), TOL);
            }
        }
    }

    public void testAssignFloatArray() {
        float[] expected = new float[A.size()];
        for (int i = 0; i < A.size(); i++) {
            expected[i] = (float) Math.random();
        }
        A.assign(expected);
        int idx = 0;
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(expected[idx++], A.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testAssignFloatArrayArrayArray() {
        float[][][] expected = new float[NSLICES][NROWS][NCOLUMNS];
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    expected[s][r][c] = (float) Math.random();
                }
            }
        }
        A.assign(expected);
        for (int s = 0; s < NSLICES; s++) {
            assertTrue(NROWS == expected[s].length);
            for (int r = 0; r < NROWS; r++) {
                assertTrue(NCOLUMNS == expected[s][r].length);
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(expected[s][r][c], A.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testAssignFloatFunction() {
        FloatMatrix3D Acopy = A.copy();
        A.assign(FloatFunctions.acos);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    float expected = (float) Math.acos(Acopy.getQuick(s, r, c));
                    assertEquals(expected, A.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testAssignFloatMatrix3D() {
        A.assign(B);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++)
                    assertEquals(B.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
            }
        }
    }

    public void testAssignFloatMatrix3DFloatFloatFunction() {
        FloatMatrix3D Acopy = A.copy();
        A.assign(B, FloatFunctions.div);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(Acopy.getQuick(s, r, c) / B.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testAssignFloatMatrix3DFloatFloatFunctionIntArrayListIntArrayListIntArrayList() {
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    sliceList.add(s);
                    rowList.add(r);
                    columnList.add(c);
                }
            }
        }
        FloatMatrix3D Acopy = A.copy();
        A.assign(B, FloatFunctions.div, sliceList, rowList, columnList);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(Acopy.getQuick(s, r, c) / B.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testAssignFloatProcedureFloat() {
        FloatProcedure procedure = new FloatProcedure() {
            public boolean apply(float element) {
                if (Math.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        FloatMatrix3D Acopy = A.copy();
        A.assign(procedure, -1.0f);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    if (Math.abs(Acopy.getQuick(s, r, c)) > 0.1) {
                        assertEquals(-1.0, A.getQuick(s, r, c), TOL);
                    } else {
                        assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                    }
                }
            }
        }
    }

    public void testAssignFloatProcedureFloatFunction() {
        FloatProcedure procedure = new FloatProcedure() {
            public boolean apply(float element) {
                if (Math.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        FloatMatrix3D Acopy = A.copy();
        A.assign(procedure, FloatFunctions.tan);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    if (Math.abs(Acopy.getQuick(s, r, c)) > 0.1) {
                        assertEquals(Math.tan(Acopy.getQuick(s, r, c)), A.getQuick(s, r, c), TOL);
                    } else {
                        assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                    }
                }
            }
        }
    }

    public void testCardinality() {
        int card = A.cardinality();
        assertEquals(NSLICES * NROWS * NCOLUMNS, card);
    }

    public void testEqualsFloat() {
        float value = 1;
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
        A.setQuick(NSLICES / 3, NROWS / 3, NCOLUMNS / 3, 0.7f);
        A.setQuick(NSLICES / 3, NROWS / 2, NCOLUMNS / 2, 0.1f);
        float[] maxAndLoc = A.getMaxLocation();
        assertEquals(0.7f, maxAndLoc[0], TOL);
        assertEquals(NSLICES / 3, (int) maxAndLoc[1]);
        assertEquals(NROWS / 3, (int) maxAndLoc[2]);
        assertEquals(NCOLUMNS / 3, (int) maxAndLoc[3]);
    }

    public void testMinLocation() {
        A.assign(0);
        A.setQuick(NSLICES / 3, NROWS / 3, NCOLUMNS / 3, -0.7f);
        A.setQuick(NSLICES / 3, NROWS / 2, NCOLUMNS / 2, -0.1f);
        float[] minAndLoc = A.getMinLocation();
        assertEquals(-0.7f, minAndLoc[0], TOL);
        assertEquals(NSLICES / 3, (int) minAndLoc[1]);
        assertEquals(NROWS / 3, (int) minAndLoc[2]);
        assertEquals(NCOLUMNS / 3, (int) minAndLoc[3]);
    }

    public void testGetNegativeValues() {
        A.assign(0);
        A.setQuick(NSLICES / 3, NROWS / 3, NCOLUMNS / 3, -0.7f);
        A.setQuick(NSLICES / 2, NROWS / 2, NCOLUMNS / 2, -0.1f);
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        A.getNegativeValues(sliceList, rowList, columnList, valueList);
        assertEquals(2, sliceList.size());
        assertEquals(2, rowList.size());
        assertEquals(2, columnList.size());
        assertEquals(2, valueList.size());
        assertTrue(sliceList.contains(NSLICES / 3));
        assertTrue(sliceList.contains(NSLICES / 2));
        assertTrue(rowList.contains(NROWS / 3));
        assertTrue(rowList.contains(NROWS / 2));
        assertTrue(columnList.contains(NCOLUMNS / 3));
        assertTrue(columnList.contains(NCOLUMNS / 2));
        assertTrue(valueList.contains(-0.7f));
        assertTrue(valueList.contains(-0.1f));
    }

    public void testGetNonZeros() {
        A.assign(0);
        A.setQuick(NSLICES / 3, NROWS / 3, NCOLUMNS / 3, 0.7f);
        A.setQuick(NSLICES / 2, NROWS / 2, NCOLUMNS / 2, 0.1f);
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        A.getNonZeros(sliceList, rowList, columnList, valueList);
        assertEquals(2, sliceList.size());
        assertEquals(2, rowList.size());
        assertEquals(2, columnList.size());
        assertEquals(2, valueList.size());
        assertTrue(sliceList.contains(NSLICES / 3));
        assertTrue(sliceList.contains(NSLICES / 2));
        assertTrue(rowList.contains(NROWS / 3));
        assertTrue(rowList.contains(NROWS / 2));
        assertTrue(columnList.contains(NCOLUMNS / 3));
        assertTrue(columnList.contains(NCOLUMNS / 2));
        assertTrue(valueList.contains(0.7f));
        assertTrue(valueList.contains(0.1f));
    }

    public void testGetPositiveValues() {
        A.assign(0);
        A.setQuick(NSLICES / 3, NROWS / 3, NCOLUMNS / 3, 0.7f);
        A.setQuick(NSLICES / 2, NROWS / 2, NCOLUMNS / 2, 0.1f);
        IntArrayList sliceList = new IntArrayList();
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        A.getPositiveValues(sliceList, rowList, columnList, valueList);
        assertEquals(2, sliceList.size());
        assertEquals(2, rowList.size());
        assertEquals(2, columnList.size());
        assertEquals(2, valueList.size());
        assertTrue(sliceList.contains(NSLICES / 3));
        assertTrue(sliceList.contains(NSLICES / 2));
        assertTrue(rowList.contains(NROWS / 3));
        assertTrue(rowList.contains(NROWS / 2));
        assertTrue(columnList.contains(NCOLUMNS / 3));
        assertTrue(columnList.contains(NCOLUMNS / 2));
        assertTrue(valueList.contains(0.7f));
        assertTrue(valueList.contains(0.1f));
    }

    public void testToArray() {
        float[][][] array = A.toArray();
        for (int s = 0; s < NSLICES; s++) {
            assertTrue(NROWS == array[s].length);
            for (int r = 0; r < NROWS; r++) {
                assertTrue(NCOLUMNS == array[s][r].length);
                for (int c = 0; c < NCOLUMNS; c++)
                    assertEquals(0, Math.abs(array[s][r][c] - A.getQuick(s, r, c)), TOL);
            }
        }
    }

    public void testVectorize() {
        FloatMatrix1D Avec = A.vectorize();
        int idx = 0;
        for (int s = 0; s < NSLICES; s++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                for (int r = 0; r < NROWS; r++) {
                    assertEquals(A.getQuick(s, r, c), Avec.getQuick(idx++), TOL);
                }
            }
        }
    }

    public void testViewColumn() {
        FloatMatrix2D B = A.viewColumn(NCOLUMNS / 2);
        assertEquals(NSLICES, B.rows());
        assertEquals(NROWS, B.columns());
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                assertEquals(A.getQuick(s, r, NCOLUMNS / 2), B.getQuick(s, r), TOL);
            }
        }
    }

    public void testViewColumnFlip() {
        FloatMatrix3D B = A.viewColumnFlip();
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
        FloatMatrix3D B = A.viewDice(2, 1, 0);
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
        FloatMatrix3D B = A.viewPart(NSLICES / 2, NROWS / 2, NCOLUMNS / 2, NSLICES / 3, NROWS / 3, NCOLUMNS / 3);
        for (int s = 0; s < NSLICES / 3; s++) {
            for (int r = 0; r < NROWS / 3; r++) {
                for (int c = 0; c < NCOLUMNS / 3; c++) {
                    assertEquals(A.getQuick(NSLICES / 2 + s, NROWS / 2 + r, NCOLUMNS / 2 + c), B.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testViewRow() {
        FloatMatrix2D B = A.viewRow(NROWS / 2);
        assertEquals(NSLICES, B.rows());
        assertEquals(NCOLUMNS, B.columns());
        for (int s = 0; s < NSLICES; s++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(s, NROWS / 2, c), B.getQuick(s, c), TOL);
            }
        }
    }

    public void testViewRowFlip() {
        FloatMatrix3D B = A.viewRowFlip();
        assertEquals(A.size(), B.size());
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(A.getQuick(s, NROWS - 1 - r, c), B.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testViewSelectionFloatMatrix2DProcedure() {
        A.assign(0);
        final float value = 2;
        A.setQuick(NSLICES / 2, NROWS / 4, 0, value);
        FloatMatrix3D B = A.viewSelection(new FloatMatrix2DProcedure() {
            public boolean apply(FloatMatrix2D element) {
                if (Math.abs(element.getQuick(NROWS / 4, 0) - value) <= TOL) {
                    return true;
                } else {
                    return false;
                }

            }
        });
        assertEquals(1, B.slices());
        assertEquals(A.rows(), B.rows());
        assertEquals(A.columns(), B.columns());
        assertEquals(A.getQuick(NSLICES / 2, NROWS / 4, 0), B.getQuick(0, NROWS / 4, 0), TOL);
    }

    public void testViewSelectionIntArrayIntArrayIntArray() {
        int[] sliceIndexes = new int[] { NSLICES / 2, NSLICES / 3 };
        int[] rowIndexes = new int[] { NROWS / 6, NROWS / 5, NROWS / 4, NROWS / 3, NROWS / 2 };
        int[] colIndexes = new int[] { NCOLUMNS / 6, NCOLUMNS / 5, NCOLUMNS / 4, NCOLUMNS / 3, NCOLUMNS / 2, NCOLUMNS - 1 };
        FloatMatrix3D B = A.viewSelection(sliceIndexes, rowIndexes, colIndexes);
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
        FloatMatrix2D B = A.viewSlice(NSLICES / 2);
        assertEquals(NROWS, B.rows());
        assertEquals(NCOLUMNS, B.columns());
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(NSLICES / 2, r, c), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewSliceFlip() {
        FloatMatrix3D B = A.viewSliceFlip();
        assertEquals(A.size(), B.size());
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(A.getQuick(NSLICES - 1 - s, r, c), B.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testViewSorted() {
        FloatMatrix3D B = A.viewSorted(1, 1);
        for (int s = 0; s < NSLICES - 1; s++) {
            assertTrue(B.getQuick(s + 1, 1, 1) >= B.getQuick(s, 1, 1));
        }
    }

    public void testViewStrides() {
        int sliceStride = 2;
        int rowStride = 2;
        int colStride = 2;
        FloatMatrix3D B = A.viewStrides(sliceStride, rowStride, colStride);
        for (int s = 0; s < B.slices(); s++) {
            for (int r = 0; r < B.rows(); r++) {
                for (int c = 0; c < B.columns(); c++) {
                    assertEquals(A.getQuick(s * sliceStride, r * rowStride, c * colStride), B.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testZSum() {
        float sum = A.zSum();
        float expected = 0;
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    expected += A.getQuick(s, r, c);
                }
            }
        }
        assertEquals(expected, sum, TOL);
    }

}
