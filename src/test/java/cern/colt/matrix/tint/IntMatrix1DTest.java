package cern.colt.matrix.tint;

import java.util.Random;

import junit.framework.TestCase;

import org.junit.Test;

import cern.colt.function.tint.IntProcedure;
import cern.colt.list.tint.IntArrayList;
import cern.jet.math.tint.IntFunctions;
import edu.emory.mathcs.utils.pc.ConcurrencyUtils;

public abstract class IntMatrix1DTest extends TestCase {
    /**
     * Matrix to test
     */
    protected IntMatrix1D A;

    /**
     * Matrix of the same size as a
     */
    protected IntMatrix1D B;

    protected int SIZE = 2 * 17 * 5;

    protected Random rand = new Random(0);

    /**
     * Constructor for IntMatrix1DTest
     */
    public IntMatrix1DTest(String arg0) {
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
            A.setQuick(i, (int) Math.max(1, rand.nextInt() % A.size()));
        }

        for (int i = 0; i < (int) B.size(); i++) {
            B.setQuick(i, (int) Math.max(1, rand.nextInt() % A.size()));
        }
    }

    protected void tearDown() throws Exception {
        A = B = null;
    }

    public void testAggregateIntIntFunctionIntFunction() {
        int expected = 0;
        for (int i = 0; i < (int) A.size(); i++) {
            int elem = A.getQuick(i);
            expected += elem * elem;
        }
        int result = A.aggregate(IntFunctions.plus, IntFunctions.square);
        assertEquals(expected, result);
    }

    public void testAggregateIntIntFunctionIntFunctionIntArrayList() {
        IntArrayList indexList = new IntArrayList();
        for (int i = 0; i < (int) A.size(); i++) {
            indexList.add(i);
        }
        int expected = 0;
        for (int i = 0; i < (int) A.size(); i++) {
            int elem = A.getQuick(i);
            expected += elem * elem;
        }
        int result = A.aggregate(IntFunctions.plus, IntFunctions.square, indexList);
        assertEquals(expected, result);
    }

    public void testAggregateIntMatrix2DIntIntFunctionIntIntFunction() {
        int expected = 0;
        for (int i = 0; i < (int) A.size(); i++) {
            int elemA = A.getQuick(i);
            int elemB = B.getQuick(i);
            expected += elemA * elemB;
        }
        int result = A.aggregate(B, IntFunctions.plus, IntFunctions.mult);
        assertEquals(expected, result);
    }

    public void testAssignInt() {
        int value = rand.nextInt();
        A.assign(value);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(value, A.getQuick(i));
        }
    }

    public void testAssignIntArray() {
        int[] expected = new int[(int) A.size()];
        for (int i = 0; i < (int) A.size(); i++) {
            expected[i] = rand.nextInt();
        }
        A.assign(expected);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(expected[i], A.getQuick(i));
        }
    }

    public void testAssignIntFunction() {
        IntMatrix1D Acopy = A.copy();
        A.assign(IntFunctions.neg);
        for (int i = 0; i < (int) A.size(); i++) {
            int expected = -Acopy.getQuick(i);
            assertEquals(expected, A.getQuick(i));
        }
    }

    public void testAssignIntMatrix1D() {
        A.assign(B);
        assertTrue(A.size() == B.size());
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(B.getQuick(i), A.getQuick(i));
        }
    }

    public void testAssignIntMatrix1DIntIntFunction() {
        IntMatrix1D Acopy = A.copy();
        A.assign(B, IntFunctions.plus);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(Acopy.getQuick(i) + B.getQuick(i), A.getQuick(i));
        }
    }

    public void testAssignIntProcedureInt() {
        IntProcedure procedure = new IntProcedure() {
            public boolean apply(int element) {
                if (Math.abs(element) > 1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        IntMatrix1D Acopy = A.copy();
        A.assign(procedure, -1);
        for (int i = 0; i < (int) A.size(); i++) {
            if (Math.abs(Acopy.getQuick(i)) > 1) {
                assertEquals(-1, A.getQuick(i));
            } else {
                assertEquals(Acopy.getQuick(i), A.getQuick(i));
            }
        }
    }

    public void testAssignIntProcedureIntFunction() {
        IntProcedure procedure = new IntProcedure() {
            public boolean apply(int element) {
                if (Math.abs(element) > 1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        IntMatrix1D Acopy = A.copy();
        A.assign(procedure, IntFunctions.neg);
        for (int i = 0; i < (int) A.size(); i++) {
            if (Math.abs(Acopy.getQuick(i)) > 1) {
                assertEquals(-Acopy.getQuick(i), A.getQuick(i));
            } else {
                assertEquals(Acopy.getQuick(i), A.getQuick(i));
            }
        }
    }

    public void testCardinality() {
        int card = A.cardinality();
        int expected = 0;
        for (int i = 0; i < A.size(); i++) {
            if (A.getQuick(i) != 0)
                expected++;
        }
        assertEquals(expected, card);
    }

    public void testEqualsInt() {
        int value = 1;
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
        A.setQuick((int) A.size() / 3, 7);
        A.setQuick((int) A.size() / 2, 1);
        int[] maxAndLoc = A.getMaxLocation();
        assertEquals(7, maxAndLoc[0]);
        assertEquals((int) A.size() / 3, (int) maxAndLoc[1]);
    }

    public void testMinLocation() {
        A.assign(0);
        A.setQuick((int) A.size() / 3, -7);
        A.setQuick((int) A.size() / 2, -1);
        int[] minAndLoc = A.getMinLocation();
        assertEquals(-7, minAndLoc[0]);
        assertEquals((int) A.size() / 3, (int) minAndLoc[1]);
    }

    public void testGetNegativeValuesIntArrayListIntArrayList() {
        A.assign(0);
        A.setQuick((int) A.size() / 3, -7);
        A.setQuick((int) A.size() / 2, -1);
        IntArrayList indexList = new IntArrayList();
        IntArrayList valueList = new IntArrayList();
        A.getNegativeValues(indexList, valueList);
        assertEquals(2, indexList.size());
        assertEquals(2, valueList.size());
        assertTrue(indexList.contains((int) A.size() / 3));
        assertTrue(indexList.contains((int) A.size() / 2));
        assertTrue(valueList.contains(-7));
        assertTrue(valueList.contains(-1));
    }

    public void testGetNonZerosIntArrayListIntArrayList() {
        A.assign(0);
        A.setQuick((int) A.size() / 3, 7);
        A.setQuick((int) A.size() / 2, 1);
        IntArrayList indexList = new IntArrayList();
        IntArrayList valueList = new IntArrayList();
        A.getNonZeros(indexList, valueList);
        assertEquals(2, indexList.size());
        assertEquals(2, valueList.size());
        assertTrue(indexList.contains((int) A.size() / 3));
        assertTrue(indexList.contains((int) A.size() / 2));
        assertTrue(valueList.contains(7));
        assertTrue(valueList.contains(1));
    }

    public void testGetPositiveValuesIntArrayListIntArrayList() {
        A.assign(0);
        A.setQuick((int) A.size() / 3, 7);
        A.setQuick((int) A.size() / 2, 1);
        IntArrayList indexList = new IntArrayList();
        IntArrayList valueList = new IntArrayList();
        A.getPositiveValues(indexList, valueList);
        assertEquals(2, indexList.size());
        assertEquals(2, valueList.size());
        assertTrue(indexList.contains((int) A.size() / 3));
        assertTrue(indexList.contains((int) A.size() / 2));
        assertTrue(valueList.contains(7));
        assertTrue(valueList.contains(1));
    }

    public void testToArray() {
        int[] array = A.toArray();
        assertTrue((int) A.size() == array.length);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(array[i], A.getQuick(i));
        }
    }

    public void testToArrayIntArray() {
        int[] array = new int[(int) A.size()];
        A.toArray(array);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(A.getQuick(i), array[i]);
        }
    }

    public void testReshapeIntInt() {
        int rows = 10;
        int columns = 17;
        IntMatrix2D B = A.reshape(rows, columns);
        int idx = 0;
        for (int c = 0; c < columns; c++) {
            for (int r = 0; r < rows; r++) {
                assertEquals(A.getQuick(idx++), B.getQuick(r, c));
            }
        }
    }

    public void testReshapeIntIntInt() {
        int slices = 2;
        int rows = 5;
        int columns = 17;
        IntMatrix3D B = A.reshape(slices, rows, columns);
        int idx = 0;
        for (int s = 0; s < slices; s++) {
            for (int c = 0; c < columns; c++) {
                for (int r = 0; r < rows; r++) {
                    assertEquals(A.getQuick(idx++), B.getQuick(s, r, c));
                }
            }
        }
    }

    public void testSwap() {
        IntMatrix1D Acopy = A.copy();
        IntMatrix1D Bcopy = B.copy();
        A.swap(B);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(Bcopy.getQuick(i), A.getQuick(i));
            assertEquals(Acopy.getQuick(i), B.getQuick(i));
        }
    }

    public void testViewFlip() {
        IntMatrix1D b = A.viewFlip();
        assertEquals((int) A.size(), b.size());
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(A.getQuick(i), b.getQuick((int) A.size() - 1 - i));
        }
    }

    public void testViewPart() {
        IntMatrix1D b = A.viewPart(15, 11);
        for (int i = 0; i < 11; i++) {
            assertEquals(A.getQuick(15 + i), b.getQuick(i));
        }
    }

    public void testViewSelectionIntProcedure() {
        IntMatrix1D b = A.viewSelection(new IntProcedure() {
            public boolean apply(int element) {
                return element % 2 == 0;
            }
        });
        for (int i = 0; i < b.size(); i++) {
            int el = b.getQuick(i);
            if (el % 2 != 0) {
                fail();
            }
        }
    }

    public void testViewSelectionIntArray() {
        int[] indexes = new int[] { 5, 11, 22, 37, 101 };
        IntMatrix1D b = A.viewSelection(indexes);
        for (int i = 0; i < indexes.length; i++) {
            assertEquals(A.getQuick(indexes[i]), b.getQuick(i));
        }
    }

    public void testViewSorted() {
        IntMatrix1D b = A.viewSorted();
        for (int i = 0; i < (int) A.size() - 1; i++) {
            assertTrue(b.getQuick(i + 1) >= b.getQuick(i));
        }
    }

    public void testViewStrides() {
        int stride = 3;
        IntMatrix1D b = A.viewStrides(stride);
        for (int i = 0; i < b.size(); i++) {
            assertEquals(A.getQuick(i * stride), b.getQuick(i));
        }
    }

    public void testZDotProductIntMatrix1D() {
        int product = A.zDotProduct(B);
        int expected = 0;
        for (int i = 0; i < (int) A.size(); i++) {
            expected += A.getQuick(i) * B.getQuick(i);
        }
        assertEquals(expected, product);
    }

    public void testZDotProductIntMatrix1DIntInt() {
        int product = A.zDotProduct(B, 5, (int) B.size() - 10);
        int expected = 0;
        for (int i = 5; i < (int) A.size() - 5; i++) {
            expected += A.getQuick(i) * B.getQuick(i);
        }
        assertEquals(expected, product);
    }

    @Test
    public void testZDotProductIntMatrix1DIntIntIntArrayList() {
        IntArrayList indexList = new IntArrayList();
        IntArrayList valueList = new IntArrayList();
        B.getNonZeros(indexList, valueList);
        int product = A.zDotProduct(B, 5, (int) B.size() - 10, indexList);
        int expected = 0;
        for (int i = 5; i < (int) A.size() - 5; i++) {
            expected += A.getQuick(i) * B.getQuick(i);
        }
        assertEquals(expected, product);
    }

    public void testZSum() {
        int sum = A.zSum();
        int expected = 0;
        for (int i = 0; i < (int) A.size(); i++) {
            expected += A.getQuick(i);
        }
        assertEquals(expected, sum);
    }

}
