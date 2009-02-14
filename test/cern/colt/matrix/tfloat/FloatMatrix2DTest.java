package cern.colt.matrix.tfloat;

import junit.framework.TestCase;
import cern.colt.function.tfloat.FloatProcedure;
import cern.colt.function.tfloat.IntIntFloatFunction;
import cern.colt.list.tfloat.FloatArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix1D;
import cern.jet.math.tfloat.FloatFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public abstract class FloatMatrix2DTest extends TestCase {
    /**
     * Matrix to test
     */
    protected FloatMatrix2D A;

    /**
     * Matrix of the same size as A
     */
    protected FloatMatrix2D B;

    /**
     * Matrix of the size A.columns() x A.rows()
     */
    protected FloatMatrix2D Bt;

    protected int NROWS = 13;

    protected int NCOLUMNS = 17;

    protected float TOL = 1e-3f;

    /**
     * Constructor for FloatMatrix2DTest
     */
    public FloatMatrix2DTest(String arg0) {
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
                A.setQuick(r, c, (float)Math.random());
            }
        }

        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                B.setQuick(r, c, (float)Math.random());
            }
        }

        for (int r = 0; r < NCOLUMNS; r++) {
            for (int c = 0; c < NROWS; c++) {
                Bt.setQuick(r, c, (float)Math.random());
            }
        }
    }

    @Override
    protected void tearDown() throws Exception {
        A = B = Bt = null;
    }

    public void testAggregateFloatFloatFunctionFloatFunction() {
        float expected = 0;
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                float elem = A.getQuick(r, c);
                expected += elem * elem;
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
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                float elem = A.getQuick(r, c);
                if (Math.abs(elem) > 0.2) {
                    expected += elem * elem;
                }
            }
        }

        float result = A.aggregate(FloatFunctions.plus, FloatFunctions.square, procedure);
        assertEquals(expected, result, TOL);
    }

    public void testAggregateFloatFloatFunctionFloatFunctionIntArrayListIntArrayList() {
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                rowList.add(r);
                columnList.add(c);
            }
        }
        float expected = 0;
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                float elem = A.getQuick(r, c);
                expected += elem * elem;
            }
        }
        float result = A.aggregate(FloatFunctions.plus, FloatFunctions.square, rowList, columnList);
        assertEquals(expected, result, TOL);
    }

    public void testAggregateFloatMatrix2DFloatFloatFunctionFloatFloatFunction() {
        float expected = 0;
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                float elemA = A.getQuick(r, c);
                float elemB = B.getQuick(r, c);
                expected += elemA * elemB;
            }
        }
        float result = A.aggregate(B, FloatFunctions.plus, FloatFunctions.mult);
        assertEquals(expected, result, TOL);
    }

    public void testAssignFloat() {
        float value = (float)Math.random();
        A.assign(value);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(value, A.getQuick(r, c), TOL);
        }
    }


    public void testAssignFloatArrayArray() {
        float[][] expected = new float[NROWS][NCOLUMNS];
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                expected[r][c] = (float)Math.random();
            }
        }
        A.assign(expected);
        for (int r = 0; r < NROWS; r++) {
            assertTrue(NCOLUMNS == expected[r].length);
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(expected[r][c], A.getQuick(r, c), TOL);
        }
    }

    public void testAssignFloatFunction() {
        FloatMatrix2D Acopy = A.copy();
        A.assign(FloatFunctions.acos);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                float expected = (float)Math.acos(Acopy.getQuick(r, c));
                assertEquals(expected, A.getQuick(r, c), TOL);
            }
        }
    }

    public void testAssignFloatMatrix2D() {
        A.assign(B);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(B.getQuick(r, c), A.getQuick(r, c), TOL);
        }
    }

    public void testAssignFloatMatrix2DFloatFloatFunction() {
        FloatMatrix2D Acopy = A.copy();
        A.assign(B, FloatFunctions.div);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(Acopy.getQuick(r, c) / B.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testAssignFloatMatrix2DFloatFloatFunctionIntArrayListIntArrayList() {
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                rowList.add(r);
                columnList.add(c);
            }
        }
        FloatMatrix2D Acopy = A.copy();
        A.assign(B, FloatFunctions.div, rowList, columnList);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(Acopy.getQuick(r, c) / B.getQuick(r, c), A.getQuick(r, c), TOL);
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
        FloatMatrix2D Acopy = A.copy();
        A.assign(procedure, -1.0f);
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
        FloatMatrix2D Acopy = A.copy();
        A.assign(procedure, FloatFunctions.tan);
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

    public void testForEachNonZero() {
        FloatMatrix2D Acopy = A.copy();
        IntIntFloatFunction function = new IntIntFloatFunction() {
            public float apply(int first, int second, float third) {
                return (float)Math.sqrt(third);
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
        A.setQuick(NROWS / 3, NCOLUMNS / 3, 0.7f);
        A.setQuick(NROWS / 2, NCOLUMNS / 2, 0.1f);
        float[] maxAndLoc = A.getMaxLocation();
        assertEquals(0.7f, maxAndLoc[0], TOL);
        assertEquals(NROWS / 3, (int) maxAndLoc[1]);
        assertEquals(NCOLUMNS / 3, (int) maxAndLoc[2]);
    }

    public void testMinLocation() {
        A.assign(0);
        A.setQuick(NROWS / 3, NCOLUMNS / 3, -0.7f);
        A.setQuick(NROWS / 2, NCOLUMNS / 2, -0.1f);
        float[] minAndLoc = A.getMinLocation();
        assertEquals(-0.7f, minAndLoc[0], TOL);
        assertEquals(NROWS / 3, (int) minAndLoc[1]);
        assertEquals(NCOLUMNS / 3, (int) minAndLoc[2]);
    }

    public void testGetNegativeValues() {
        A.assign(0);
        A.setQuick(NROWS / 3, NCOLUMNS / 3, -0.7f);
        A.setQuick(NROWS / 2, NCOLUMNS / 2, -0.1f);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        A.getNegativeValues(rowList, columnList, valueList);
        assertEquals(2, rowList.size());
        assertEquals(2, columnList.size());
        assertEquals(2, valueList.size());
        assertTrue(rowList.contains(NROWS / 3));
        assertTrue(rowList.contains(NROWS / 2));
        assertTrue(columnList.contains(NCOLUMNS / 3));
        assertTrue(columnList.contains(NCOLUMNS / 2));
        assertTrue(valueList.contains(-0.7f));
        assertTrue(valueList.contains(-0.1f));
    }

    public void testGetNonZeros() {
        A.assign(0);
        A.setQuick(NROWS / 3, NCOLUMNS / 3, 0.7f);
        A.setQuick(NROWS / 2, NCOLUMNS / 2, 0.1f);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        A.getNonZeros(rowList, columnList, valueList);
        assertEquals(2, rowList.size());
        assertEquals(2, columnList.size());
        assertEquals(2, valueList.size());
        assertTrue(rowList.contains(NROWS / 3));
        assertTrue(rowList.contains(NROWS / 2));
        assertTrue(columnList.contains(NCOLUMNS / 3));
        assertTrue(columnList.contains(NCOLUMNS / 2));
        assertTrue(valueList.contains(0.7f));
        assertTrue(valueList.contains(0.1f));
    }

    public void testGetPositiveValues() {
        A.assign(0);
        A.setQuick(NROWS / 3, NCOLUMNS / 3, 0.7f);
        A.setQuick(NROWS / 2, NCOLUMNS / 2, 0.1f);
        IntArrayList rowList = new IntArrayList();
        IntArrayList columnList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        A.getPositiveValues(rowList, columnList, valueList);
        assertEquals(2, rowList.size());
        assertEquals(2, columnList.size());
        assertEquals(2, valueList.size());
        assertTrue(rowList.contains(NROWS / 3));
        assertTrue(rowList.contains(NROWS / 2));
        assertTrue(columnList.contains(NCOLUMNS / 3));
        assertTrue(columnList.contains(NCOLUMNS / 2));
        assertTrue(valueList.contains(0.7f));
        assertTrue(valueList.contains(0.1f));
    }

    public void testToArray() {
        float[][] array = A.toArray();
        assertTrue(NROWS == array.length);
        for (int r = 0; r < NROWS; r++) {
            assertTrue(NCOLUMNS == array[r].length);
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(array[r][c] - A.getQuick(r, c)), TOL);
        }
    }

    public void testVectorize() {
        FloatMatrix1D Avec = A.vectorize();
        int idx = 0;
        for (int c = 0; c < NCOLUMNS; c++) {
            for (int r = 0; r < NROWS; r++) {
                assertEquals(A.getQuick(r, c), Avec.getQuick(idx++), TOL);
            }
        }
    }

    public void testViewColumn() {
        FloatMatrix1D col = A.viewColumn(NCOLUMNS / 2);
        assertEquals(NROWS, col.size());
        for (int r = 0; r < NROWS; r++) {
            assertEquals(A.getQuick(r, NCOLUMNS / 2), col.getQuick(r), TOL);
        }
    }
    
    public void testViewColumnFlip() {
        FloatMatrix2D B = A.viewColumnFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, NCOLUMNS - 1 - c), B.getQuick(r, c), TOL);
            }
        }
    }
    
    public void testViewDice() {
        FloatMatrix2D B = A.viewDice();
        assertEquals(NROWS, B.columns());
        assertEquals(NCOLUMNS, B.rows());
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(c, r), TOL);
            }
        }
    }
    
    public void testViewPart() {
        FloatMatrix2D B = A.viewPart(NROWS / 2, NCOLUMNS / 2, NROWS / 3, NCOLUMNS / 3);
        assertEquals(NROWS / 3, B.rows());
        assertEquals(NCOLUMNS / 3, B.columns());
        for (int r = 0; r < NROWS / 3; r++) {
            for (int c = 0; c < NCOLUMNS / 3; c++) {
                assertEquals(A.getQuick(NROWS / 2 + r, NCOLUMNS / 2 + c), B.getQuick(r, c), TOL);
            }
        }
    }
    
    public void testViewRow() {
        FloatMatrix1D B = A.viewRow(NROWS / 2);
        assertEquals(NCOLUMNS, B.size());
        for (int r = 0; r < NCOLUMNS; r++) {
            assertEquals(A.getQuick(NROWS / 2, r), B.getQuick(r), TOL);
        }
    }

    
    public void testViewRowFlip() {
        FloatMatrix2D B = A.viewRowFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(A.getQuick(NROWS - 1 - r, c), B.getQuick(r, c), TOL);
            }
        }
    }
    
    public void testViewSelectionFloatMatrix1DProcedure() {
        final float value = 2;
        A.assign(0);
        A.setQuick(NROWS / 4, 0, value);
        A.setQuick(NROWS / 2, 0, value);
        FloatMatrix2D B = A.viewSelection(new FloatMatrix1DProcedure() {
            public boolean apply(FloatMatrix1D element) {
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
        int[] colIndexes = new int[] { NCOLUMNS / 6, NCOLUMNS / 5, NCOLUMNS / 4, NCOLUMNS / 3, NCOLUMNS / 2, NCOLUMNS - 1};
        FloatMatrix2D B = A.viewSelection(rowIndexes, colIndexes);
        assertEquals(rowIndexes.length, B.rows());
        assertEquals(colIndexes.length, B.columns());
        for (int r = 0; r < rowIndexes.length; r++) {
            for (int c = 0; c < colIndexes.length; c++) {
                assertEquals(A.getQuick(rowIndexes[r], colIndexes[c]), B.getQuick(r, c), TOL);
            }
        }
    }
    
    public void testViewSorted() {
        FloatMatrix2D B = A.viewSorted(1);
        for (int r = 0; r < NROWS - 1; r++) {
            assertTrue(B.getQuick(r+1, 1) >= B.getQuick(r, 1));
        }
    }
    
    public void testViewStrides() {
        int rowStride = 3;
        int colStride = 5;
        FloatMatrix2D B = A.viewStrides(rowStride, colStride);
        for (int r = 0; r < B.rows(); r++) {
            for (int c = 0; c < B.columns(); c++) {
                assertEquals(A.getQuick(r * rowStride, c * colStride), B.getQuick(r, c), TOL);
            }
        }
    }
    
    public void testZMultFloatMatrix1DFloatMatrix1DFloatFloatBoolean() {
        FloatMatrix1D y = new DenseFloatMatrix1D(NCOLUMNS);
        for (int i = 0; i < y.size(); i++) {
            y.setQuick(i, (float)Math.random());
        }
        float alpha = 3;
        float beta = 5;
        FloatMatrix1D z = FloatFactory1D.dense.random(NROWS);
        float[] expected = z.toArray();
        z = A.zMult(y, z, alpha, beta, false);
        for (int r = 0; r < NROWS; r++) {
            float s = 0;
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
        expected = new float[NROWS];
        for (int r = 0; r < NROWS; r++) {
            float s = 0;
            for (int c = 0; c < NCOLUMNS; c++) {
                s += A.getQuick(r, c) * y.getQuick(c);
            }
            expected[r] = s * alpha;
        }
        for (int r = 0; r < NROWS; r++) {
            assertEquals(expected[r], z.getQuick(r), TOL);
        }
        
        //transpose
        y = new DenseFloatMatrix1D(NROWS);
        for (int i = 0; i < y.size(); i++) {
            y.setQuick(i, (float)Math.random());
        }
        z = FloatFactory1D.dense.random(NCOLUMNS);
        expected = z.toArray();
        z = A.zMult(y, z, alpha, beta, true);
        for (int r = 0; r < NCOLUMNS; r++) {
            float s = 0;
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
        expected = new float[NCOLUMNS];
        for (int r = 0; r < NCOLUMNS; r++) {
            float s = 0;
            for (int c = 0; c < NROWS; c++) {
                s += A.getQuick(c, r) * y.getQuick(c);
            }
            expected[r] = s * alpha;
        }
        for (int r = 0; r < NCOLUMNS; r++) {
            assertEquals(expected[r], z.getQuick(r), TOL);
        }
    }
    
    public void testZMultFloatMatrix2DFloatMatrix2DFloatFloatBooleanBoolean() {
        float alpha = 3;
        float beta = 5;
        FloatMatrix2D C = FloatFactory2D.dense.random(NROWS, NROWS);
        float[][] expected = C.toArray();
        C = A.zMult(Bt, C, alpha, beta, false, false);
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                float s = 0;
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
        expected = new float[NROWS][NROWS];
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                float s = 0;
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
        C = FloatFactory2D.dense.random(NCOLUMNS, NCOLUMNS);
        expected = C.toArray();
        C = A.zMult(B, C, alpha, beta, true, false);
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                float s = 0;
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
        expected = new float[NCOLUMNS][NCOLUMNS];
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                float s = 0;
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
        C = FloatFactory2D.dense.random(NROWS, NROWS);
        expected = C.toArray();
        C = A.zMult(B, C, alpha, beta, false, true);
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                float s = 0;
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
        expected = new float[NROWS][NROWS];
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                float s = 0;
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
        C = FloatFactory2D.dense.random(NCOLUMNS, NCOLUMNS);
        expected = C.toArray();
        C = A.zMult(Bt, C, alpha, beta, true, true);
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                float s = 0;
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
        expected = new float[NCOLUMNS][NCOLUMNS];
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                float s = 0;
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
        float sum = A.zSum();
        float expected = 0;
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                expected += A.getQuick(r, c);
            }
        }
        assertEquals(expected, sum, TOL);
    }
    
}
