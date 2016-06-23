package cern.colt.matrix.tfloat;

import junit.framework.TestCase;

import org.junit.Test;

import cern.colt.function.tfloat.FloatProcedure;
import cern.colt.list.tfloat.FloatArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.jet.math.tfloat.FloatFunctions;
import edu.emory.mathcs.utils.pc.ConcurrencyUtils;

public abstract class FloatMatrix1DTest extends TestCase {
    /**
     * Matrix to test
     */
    protected FloatMatrix1D A;

    /**
     * Matrix of the same size as a
     */
    protected FloatMatrix1D B;

    protected int SIZE = 2 * 17 * 5;

    protected float TOL = 1e-3f;

    /**
     * Constructor for FloatMatrix1DTest
     */
    public FloatMatrix1DTest(String arg0) {
        super(arg0);
    }

    protected void setUp() throws Exception {
        createMatrices();
        populateMatrices();
    }

    protected abstract void createMatrices() throws Exception;

    protected void populateMatrices() {
        ConcurrencyUtils.setThreadsBeginN_1D(1);

        for (int i = 0; i < (int) A.size(); i++) {
            A.setQuick(i, (float) Math.random());
        }

        for (int i = 0; i < (int) B.size(); i++) {
            B.setQuick(i, (float) Math.random());
        }
    }

    protected void tearDown() throws Exception {
        A = B = null;
    }

    public void testAggregateFloatFloatFunctionFloatFunction() {
        float expected = 0;
        for (int i = 0; i < (int) A.size(); i++) {
            float elem = A.getQuick(i);
            expected += elem * elem;
        }
        float result = A.aggregate(FloatFunctions.plus, FloatFunctions.square);
        assertEquals(expected, result, TOL);
    }

    public void testAggregateFloatFloatFunctionFloatFunctionIntArrayList() {
        IntArrayList indexList = new IntArrayList();
        for (int i = 0; i < (int) A.size(); i++) {
            indexList.add(i);
        }
        float expected = 0;
        for (int i = 0; i < (int) A.size(); i++) {
            float elem = A.getQuick(i);
            expected += elem * elem;
        }
        float result = A.aggregate(FloatFunctions.plus, FloatFunctions.square, indexList);
        assertEquals(expected, result, TOL);
    }

    public void testAggregateFloatMatrix2DFloatFloatFunctionFloatFloatFunction() {
        float expected = 0;
        for (int i = 0; i < (int) A.size(); i++) {
            float elemA = A.getQuick(i);
            float elemB = B.getQuick(i);
            expected += elemA * elemB;
        }
        float result = A.aggregate(B, FloatFunctions.plus, FloatFunctions.mult);
        assertEquals(expected, result, TOL);
    }

    public void testAssignFloat() {
        float value = (float) Math.random();
        A.assign(value);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(value, A.getQuick(i), TOL);
        }
    }

    public void testAssignFloatArray() {
        float[] expected = new float[(int) A.size()];
        for (int i = 0; i < (int) A.size(); i++) {
            expected[i] = (float) Math.random();
        }
        A.assign(expected);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(expected[i], A.getQuick(i), TOL);
        }
    }

    public void testAssignFloatFunction() {
        FloatMatrix1D Acopy = A.copy();
        A.assign(FloatFunctions.acos);
        for (int i = 0; i < (int) A.size(); i++) {
            float expected = (float) Math.acos(Acopy.getQuick(i));
            assertEquals(expected, A.getQuick(i), TOL);
        }
    }

    public void testAssignFloatMatrix1D() {
        A.assign(B);
        assertTrue(A.size() == B.size());
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(B.getQuick(i), A.getQuick(i), TOL);
        }
    }

    public void testAssignFloatMatrix1DFloatFloatFunction() {
        FloatMatrix1D Acopy = A.copy();
        A.assign(B, FloatFunctions.div);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(Acopy.getQuick(i) / B.getQuick(i), A.getQuick(i), TOL);
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
        FloatMatrix1D Acopy = A.copy();
        A.assign(procedure, -1.0f);
        for (int i = 0; i < (int) A.size(); i++) {
            if (Math.abs(Acopy.getQuick(i)) > 0.1) {
                assertEquals(-1.0, A.getQuick(i), TOL);
            } else {
                assertEquals(Acopy.getQuick(i), A.getQuick(i), TOL);
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
        FloatMatrix1D Acopy = A.copy();
        A.assign(procedure, FloatFunctions.tan);
        for (int i = 0; i < (int) A.size(); i++) {
            if (Math.abs(Acopy.getQuick(i)) > 0.1) {
                assertEquals(Math.tan(Acopy.getQuick(i)), A.getQuick(i), TOL);
            } else {
                assertEquals(Acopy.getQuick(i), A.getQuick(i), TOL);
            }
        }
    }

    public void testCardinality() {
        int card = A.cardinality();
        assertEquals((int) A.size(), card);
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
        A.setQuick((int) A.size() / 3, 0.7f);
        A.setQuick((int) A.size() / 2, 0.1f);
        float[] maxAndLoc = A.getMaxLocation();
        assertEquals(0.7, maxAndLoc[0], TOL);
        assertEquals((int) A.size() / 3, (int) maxAndLoc[1]);
    }

    public void testMinLocation() {
        A.assign(0);
        A.setQuick((int) A.size() / 3, -0.7f);
        A.setQuick((int) A.size() / 2, -0.1f);
        float[] minAndLoc = A.getMinLocation();
        assertEquals(-0.7f, minAndLoc[0], TOL);
        assertEquals((int) A.size() / 3, (int) minAndLoc[1]);
    }

    public void testGetNegativeValuesIntArrayListFloatArrayList() {
        A.assign(0);
        A.setQuick((int) A.size() / 3, -0.7f);
        A.setQuick((int) A.size() / 2, -0.1f);
        IntArrayList indexList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        A.getNegativeValues(indexList, valueList);
        assertEquals(2, indexList.size());
        assertEquals(2, valueList.size());
        assertTrue(indexList.contains((int) A.size() / 3));
        assertTrue(indexList.contains((int) A.size() / 2));
        assertTrue(valueList.contains(-0.7f));
        assertTrue(valueList.contains(-0.1f));
    }

    public void testGetNonZerosIntArrayListFloatArrayList() {
        A.assign(0);
        A.setQuick((int) A.size() / 3, 0.7f);
        A.setQuick((int) A.size() / 2, 0.1f);
        IntArrayList indexList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        A.getNonZeros(indexList, valueList);
        assertEquals(2, indexList.size());
        assertEquals(2, valueList.size());
        assertTrue(indexList.contains((int) A.size() / 3));
        assertTrue(indexList.contains((int) A.size() / 2));
        assertTrue(valueList.contains(0.7f));
        assertTrue(valueList.contains(0.1f));
    }

    public void testGetPositiveValuesIntArrayListFloatArrayList() {
        A.assign(0);
        A.setQuick((int) A.size() / 3, 0.7f);
        A.setQuick((int) A.size() / 2, 0.1f);
        IntArrayList indexList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        A.getPositiveValues(indexList, valueList);
        assertEquals(2, indexList.size());
        assertEquals(2, valueList.size());
        assertTrue(indexList.contains((int) A.size() / 3));
        assertTrue(indexList.contains((int) A.size() / 2));
        assertTrue(valueList.contains(0.7f));
        assertTrue(valueList.contains(0.1f));
    }

    public void testToArray() {
        float[] array = A.toArray();
        assertTrue((int) A.size() == array.length);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(array[i], A.getQuick(i), TOL);
        }
    }

    public void testToArrayFloatArray() {
        float[] array = new float[(int) A.size()];
        A.toArray(array);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(A.getQuick(i), array[i], TOL);
        }
    }

    public void testReshapeIntInt() {
        int rows = 10;
        int columns = 17;
        FloatMatrix2D B = A.reshape(rows, columns);
        int idx = 0;
        for (int c = 0; c < columns; c++) {
            for (int r = 0; r < rows; r++) {
                assertEquals(A.getQuick(idx++), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testReshapeIntIntInt() {
        int slices = 2;
        int rows = 5;
        int columns = 17;
        FloatMatrix3D B = A.reshape(slices, rows, columns);
        int idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int c = 0; c < columns; c++) {
                for (int r = 0; r < rows; r++) {
                    assertEquals(A.getQuick(idx++), B.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testSwap() {
        FloatMatrix1D Acopy = A.copy();
        FloatMatrix1D Bcopy = B.copy();
        A.swap(B);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(Bcopy.getQuick(i), A.getQuick(i), TOL);
            assertEquals(Acopy.getQuick(i), B.getQuick(i), TOL);
        }
    }

    public void testViewFlip() {
        FloatMatrix1D b = A.viewFlip();
        assertEquals((int) A.size(), b.size());
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(A.getQuick(i), b.getQuick((int) A.size() - 1 - i), TOL);
        }
    }

    public void testViewPart() {
        FloatMatrix1D b = A.viewPart(15, 11);
        for (int i = 0; i < 11; i++) {
            assertEquals(A.getQuick(15 + i), b.getQuick(i), TOL);
        }
    }

    public void testViewSelectionFloatProcedure() {
        FloatMatrix1D b = A.viewSelection(new FloatProcedure() {
            public boolean apply(float element) {
                return element % 2 == 0;
            }
        });
        for (int i = 0; i < b.size(); i++) {
            float el = b.getQuick(i);
            if (el % 2 != 0) {
                fail();
            }
        }
    }

    public void testViewSelectionIntArray() {
        int[] indexes = new int[] { 5, 11, 22, 37, 101 };
        FloatMatrix1D b = A.viewSelection(indexes);
        for (int i = 0; i < indexes.length; i++) {
            assertEquals(A.getQuick(indexes[i]), b.getQuick(i), TOL);
        }
    }

    public void testViewSorted() {
        FloatMatrix1D b = A.viewSorted();
        for (int i = 0; i < (int) A.size() - 1; i++) {
            assertTrue(b.getQuick(i + 1) >= b.getQuick(i));
        }
    }

    public void testViewStrides() {
        int stride = 3;
        FloatMatrix1D b = A.viewStrides(stride);
        for (int i = 0; i < b.size(); i++) {
            assertEquals(A.getQuick(i * stride), b.getQuick(i), TOL);
        }
    }

    public void testZDotProductFloatMatrix1D() {
        float product = A.zDotProduct(B);
        float expected = 0;
        for (int i = 0; i < (int) A.size(); i++) {
            expected += A.getQuick(i) * B.getQuick(i);
        }
        assertEquals(expected, product, TOL);
    }

    public void testZDotProductFloatMatrix1DIntInt() {
        float product = A.zDotProduct(B, 5, (int) B.size() - 10);
        float expected = 0;
        for (int i = 5; i < (int) A.size() - 5; i++) {
            expected += A.getQuick(i) * B.getQuick(i);
        }
        assertEquals(expected, product, TOL);
    }

    @Test
    public void testZDotProductFloatMatrix1DIntIntIntArrayList() {
        IntArrayList indexList = new IntArrayList();
        FloatArrayList valueList = new FloatArrayList();
        B.getNonZeros(indexList, valueList);
        float product = A.zDotProduct(B, 5, (int) B.size() - 10, indexList);
        float expected = 0;
        for (int i = 5; i < (int) A.size() - 5; i++) {
            expected += A.getQuick(i) * B.getQuick(i);
        }
        assertEquals(expected, product, TOL);
    }

    public void testZSum() {
        float sum = A.zSum();
        float expected = 0;
        for (int i = 0; i < (int) A.size(); i++) {
            expected += A.getQuick(i);
        }
        assertEquals(expected, sum, TOL);
    }

}
