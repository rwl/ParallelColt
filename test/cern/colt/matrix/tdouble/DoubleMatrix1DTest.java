package cern.colt.matrix.tdouble;

import junit.framework.TestCase;

import org.junit.Test;

import cern.colt.function.tdouble.DoubleProcedure;
import cern.colt.list.tdouble.DoubleArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.jet.math.tdouble.DoubleFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public abstract class DoubleMatrix1DTest extends TestCase {
    /**
     * Matrix to test
     */
    protected DoubleMatrix1D A;

    /**
     * Matrix of the same size as a
     */
    protected DoubleMatrix1D B;

    protected int SIZE = 2 * 17 * 5;

    protected double TOL = 1e-10;

    /**
     * Constructor for DoubleMatrix1DTest
     */
    public DoubleMatrix1DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void setUp() throws Exception {
        createMatrices();
        populateMatrices();
    }

    protected abstract void createMatrices() throws Exception;

    protected void populateMatrices() {
        ConcurrencyUtils.setThreadsBeginN_1D(1);

        for (int i = 0; i < SIZE; i++) {
            A.setQuick(i, Math.random());
        }

        for (int i = 0; i < SIZE; i++) {
            B.setQuick(i, Math.random());
        }
    }

    @Override
    protected void tearDown() throws Exception {
        A = B = null;
    }

    public void testAggregateDoubleDoubleFunctionDoubleFunction() {
        double expected = 0;
        for (int i = 0; i < SIZE; i++) {
            double elem = A.getQuick(i);
            expected += elem * elem;
        }
        double result = A.aggregate(DoubleFunctions.plus, DoubleFunctions.square);
        assertEquals(expected, result, TOL);
    }
    
    public void testAggregateDoubleDoubleFunctionDoubleFunctionIntArrayList() {
       IntArrayList indexList = new IntArrayList();
       for (int i = 0; i < SIZE; i++) {
                indexList.add(i);
        }
        double expected = 0;
        for (int i = 0; i < SIZE; i++) {
                double elem = A.getQuick(i);
                expected += elem * elem;
        }
        double result = A.aggregate(DoubleFunctions.plus, DoubleFunctions.square, indexList);
        assertEquals(expected, result, TOL);
    }

    public void testAggregateDoubleMatrix2DDoubleDoubleFunctionDoubleDoubleFunction() {
        double expected = 0;
        for (int i = 0; i < SIZE; i++) {
            double elemA = A.getQuick(i);
            double elemB = B.getQuick(i);
            expected += elemA * elemB;
        }
        double result = A.aggregate(B, DoubleFunctions.plus, DoubleFunctions.mult);
        assertEquals(expected, result, TOL);
    }

    public void testAssignDouble() {
        double value = Math.random();
        A.assign(value);
        for (int i = 0; i < SIZE; i++) {
            assertEquals(value, A.getQuick(i), TOL);
        }
    }

    public void testAssignDoubleArray() {
        double[] expected = new double[SIZE];
        for (int i = 0; i < SIZE; i++) {
            expected[i] = Math.random();
        }
        A.assign(expected);
        for (int i = 0; i < SIZE; i++) {
            assertEquals(expected[i], A.getQuick(i), TOL);
        }
    }

    public void testAssignDoubleFunction() {
        DoubleMatrix1D Acopy = A.copy();
        A.assign(DoubleFunctions.acos);
        for (int i = 0; i < SIZE; i++) {
            double expected = Math.acos(Acopy.getQuick(i));
            assertEquals(expected, A.getQuick(i), TOL);
        }
    }

    public void testAssignDoubleMatrix1D() {
        A.assign(B);
        assertTrue(A.size() == B.size());
        for (int i = 0; i < SIZE; i++) {
            assertEquals(B.getQuick(i), A.getQuick(i), TOL);
        }
    }

    public void testAssignDoubleMatrix1DDoubleDoubleFunction() {
        DoubleMatrix1D Acopy = A.copy();
        A.assign(B, DoubleFunctions.div);
        for (int i = 0; i < SIZE; i++) {
            assertEquals(Acopy.getQuick(i) / B.getQuick(i), A.getQuick(i), TOL);
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
        DoubleMatrix1D Acopy = A.copy();
        A.assign(procedure, -1.0);
        for (int i = 0; i < SIZE; i++) {
            if (Math.abs(Acopy.getQuick(i)) > 0.1) {
                assertEquals(-1.0, A.getQuick(i), TOL);
            } else {
                assertEquals(Acopy.getQuick(i), A.getQuick(i), TOL);
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
        DoubleMatrix1D Acopy = A.copy();
        A.assign(procedure, DoubleFunctions.tan);
        for (int i = 0; i < SIZE; i++) {
            if (Math.abs(Acopy.getQuick(i)) > 0.1) {
                assertEquals(Math.tan(Acopy.getQuick(i)), A.getQuick(i), TOL);
            } else {
                assertEquals(Acopy.getQuick(i), A.getQuick(i), TOL);
            }
        }
    }

    public void testCardinality() {
        int card = A.cardinality();
        assertEquals(SIZE, card);
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

    public void testMaxLocation() {
        A.assign(0);
        A.setQuick(SIZE / 3, 0.7);
        A.setQuick(SIZE / 2, 0.1);
        double[] maxAndLoc = A.getMaxLocation();
        assertEquals(0.7, maxAndLoc[0], TOL);
        assertEquals(SIZE / 3, (int) maxAndLoc[1]);
    }

    public void testMinLocation() {
        A.assign(0);
        A.setQuick(SIZE / 3, -0.7);
        A.setQuick(SIZE / 2, -0.1);
        double[] minAndLoc = A.getMinLocation();
        assertEquals(-0.7, minAndLoc[0], TOL);
        assertEquals(SIZE / 3, (int) minAndLoc[1]);
    }

    public void testGetNegativeValuesIntArrayListDoubleArrayList() {
        A.assign(0);
        A.setQuick(SIZE / 3, -0.7);
        A.setQuick(SIZE / 2, -0.1);
        IntArrayList indexList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        A.getNegativeValues(indexList, valueList);
        assertEquals(2, indexList.size());
        assertEquals(2, valueList.size());
        assertTrue(indexList.contains(SIZE / 3));
        assertTrue(indexList.contains(SIZE / 2));
        assertTrue(valueList.contains(-0.7));
        assertTrue(valueList.contains(-0.1));
    }

    public void testGetNonZerosIntArrayListDoubleArrayList() {
        A.assign(0);
        A.setQuick(SIZE / 3, 0.7);
        A.setQuick(SIZE / 2, 0.1);
        IntArrayList indexList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        A.getNonZeros(indexList, valueList);
        assertEquals(2, indexList.size());
        assertEquals(2, valueList.size());
        assertTrue(indexList.contains(SIZE / 3));
        assertTrue(indexList.contains(SIZE / 2));
        assertTrue(valueList.contains(0.7));
        assertTrue(valueList.contains(0.1));
    }

    public void testGetPositiveValuesIntArrayListDoubleArrayList() {
        A.assign(0);
        A.setQuick(SIZE / 3, 0.7);
        A.setQuick(SIZE / 2, 0.1);
        IntArrayList indexList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        A.getPositiveValues(indexList, valueList);
        assertEquals(2, indexList.size());
        assertEquals(2, valueList.size());
        assertTrue(indexList.contains(SIZE / 3));
        assertTrue(indexList.contains(SIZE / 2));
        assertTrue(valueList.contains(0.7));
        assertTrue(valueList.contains(0.1));
    }

    public void testToArray() {
        double[] array = A.toArray();
        assertTrue(SIZE == array.length);
        for (int i = 0; i < SIZE; i++) {
            assertEquals(array[i], A.getQuick(i), TOL);
        }
    }

    public void testToArrayDoubleArray() {
        double[] array = new double[SIZE];
        A.toArray(array);
        for (int i = 0; i < SIZE; i++) {
            assertEquals(A.getQuick(i), array[i], TOL);
        }
    }

    public void testReshapeIntInt() {
        int rows = 10;
        int cols = 17;
        DoubleMatrix2D B = A.reshape(rows, cols);
        int idx = 0;
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                assertEquals(A.getQuick(idx++), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testReshapeIntIntInt() {
        int slices = 2;
        int rows = 5;
        int cols = 17;
        DoubleMatrix3D B = A.reshape(slices, rows, cols);
        int idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int c = 0; c < cols; c++) {
                for (int r = 0; r < rows; r++) {
                    assertEquals(A.getQuick(idx++), B.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testSwap() {
        DoubleMatrix1D Acopy = A.copy();
        DoubleMatrix1D Bcopy = B.copy();
        A.swap(B);
        for (int i = 0; i < SIZE; i++) {
            assertEquals(Bcopy.getQuick(i), A.getQuick(i), TOL);
            assertEquals(Acopy.getQuick(i), B.getQuick(i), TOL);
        }
    }

    public void testViewFlip() {
        DoubleMatrix1D b = A.viewFlip();
        assertEquals(SIZE, b.size());
        for (int i = 0; i < SIZE; i++) {
            assertEquals(A.getQuick(i), b.getQuick(SIZE - 1 - i), TOL);
        }
    }

    public void testViewPart() {
        DoubleMatrix1D b = A.viewPart(15, 11);
        for (int i = 0; i < 11; i++) {
            assertEquals(A.getQuick(15 + i), b.getQuick(i), TOL);
        }
    }

    public void testViewSelectionDoubleProcedure() {
        DoubleMatrix1D b = A.viewSelection(new DoubleProcedure() {
            public boolean apply(double element) {
                return element % 2 == 0;
            }
        });
        for (int i = 0; i < b.size(); i++) {
            double el = b.getQuick(i);
            if (el % 2 != 0) {
                fail();
            }
        }
    }

    public void testViewSelectionIntArray() {
        int[] indexes = new int[] { 5, 11, 22, 37, 101 };
        DoubleMatrix1D b = A.viewSelection(indexes);
        for (int i = 0; i < indexes.length; i++) {
            assertEquals(A.getQuick(indexes[i]), b.getQuick(i), TOL);
        }
    }

    public void testViewSorted() {
        DoubleMatrix1D b = A.viewSorted();
        for (int i = 0; i < SIZE - 1; i++) {
            assertTrue(b.getQuick(i + 1) >= b.getQuick(i));
        }
    }

    public void testViewStrides() {
        int stride = 3;
        DoubleMatrix1D b = A.viewStrides(stride);
        for (int i = 0; i < b.size(); i++) {
            assertEquals(A.getQuick(i * stride), b.getQuick(i), TOL);
        }
    }

    public void testZDotProductDoubleMatrix1D() {
        double product = A.zDotProduct(B);
        double expected = 0;
        for (int i = 0; i < SIZE; i++) {
            expected += A.getQuick(i) * B.getQuick(i);
        }
        assertEquals(expected, product, TOL);
    }

    public void testZDotProductDoubleMatrix1DIntInt() {
        double product = A.zDotProduct(B, 5, B.size() - 10);
        double expected = 0;
        for (int i = 5; i < SIZE - 5; i++) {
            expected += A.getQuick(i) * B.getQuick(i);
        }
        assertEquals(expected, product, TOL);
    }

    @Test
    public void testZDotProductDoubleMatrix1DIntIntIntArrayList() {
        IntArrayList indexList = new IntArrayList();
        DoubleArrayList valueList = new DoubleArrayList();
        B.getNonZeros(indexList, valueList);
        double product = A.zDotProduct(B, 5, B.size() - 10, indexList);
        double expected = 0;
        for (int i = 5; i < SIZE - 5; i++) {
            expected += A.getQuick(i) * B.getQuick(i);
        }
        assertEquals(expected, product, TOL);
    }

    public void testZSum() {
        double sum = A.zSum();
        double expected = 0;
        for (int i = 0; i < SIZE; i++) {
            expected += A.getQuick(i);
        }
        assertEquals(expected, sum, TOL);
    }

}
