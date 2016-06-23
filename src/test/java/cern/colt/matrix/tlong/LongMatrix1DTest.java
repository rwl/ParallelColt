package cern.colt.matrix.tlong;

import java.util.Random;

import junit.framework.TestCase;

import org.junit.Test;

import cern.colt.function.tlong.LongProcedure;
import cern.colt.list.tint.IntArrayList;
import cern.colt.list.tlong.LongArrayList;
import cern.jet.math.tlong.LongFunctions;
import edu.emory.mathcs.utils.pc.ConcurrencyUtils;

public abstract class LongMatrix1DTest extends TestCase {
    /**
     * Matrix to test
     */
    protected LongMatrix1D A;

    /**
     * Matrix of the same size as a
     */
    protected LongMatrix1D B;

    protected int SIZE = 2 * 17 * 5;

    protected Random rand = new Random(0);

    /**
     * Constructor for LongMatrix1DTest
     */
    public LongMatrix1DTest(String arg0) {
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
            A.setQuick(i, Math.max(1, rand.nextLong() % A.size()));
        }

        for (int i = 0; i < (int) B.size(); i++) {
            B.setQuick(i, Math.max(1, rand.nextLong() % A.size()));
        }
    }

    protected void tearDown() throws Exception {
        A = B = null;
    }

    public void testAggregateLongLongFunctionLongFunction() {
        long expected = 0;
        for (int i = 0; i < (int) A.size(); i++) {
            long elem = A.getQuick(i);
            expected += elem * elem;
        }
        long result = A.aggregate(LongFunctions.plus, LongFunctions.square);
        assertEquals(expected, result);
    }

    public void testAggregateLongLongFunctionLongFunctionIntArrayList() {
        IntArrayList indexList = new IntArrayList();
        for (int i = 0; i < (int) A.size(); i++) {
            indexList.add(i);
        }
        long expected = 0;
        for (int i = 0; i < (int) A.size(); i++) {
            long elem = A.getQuick(i);
            expected += elem * elem;
        }
        long result = A.aggregate(LongFunctions.plus, LongFunctions.square, indexList);
        assertEquals(expected, result);
    }

    public void testAggregateLongMatrix2DLongLongFunctionLongLongFunction() {
        long expected = 0;
        for (int i = 0; i < (int) A.size(); i++) {
            long elemA = A.getQuick(i);
            long elemB = B.getQuick(i);
            expected += elemA * elemB;
        }
        long result = A.aggregate(B, LongFunctions.plus, LongFunctions.mult);
        assertEquals(expected, result);
    }

    public void testAssignLong() {
        long value = rand.nextLong();
        A.assign(value);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(value, A.getQuick(i));
        }
    }

    public void testAssignLongArray() {
        long[] expected = new long[(int) A.size()];
        for (int i = 0; i < (int) A.size(); i++) {
            expected[i] = rand.nextLong();
        }
        A.assign(expected);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(expected[i], A.getQuick(i));
        }
    }

    public void testAssignLongFunction() {
        LongMatrix1D Acopy = A.copy();
        A.assign(LongFunctions.neg);
        for (int i = 0; i < (int) A.size(); i++) {
            long expected = -Acopy.getQuick(i);
            assertEquals(expected, A.getQuick(i));
        }
    }

    public void testAssignLongMatrix1D() {
        A.assign(B);
        assertTrue(A.size() == B.size());
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(B.getQuick(i), A.getQuick(i));
        }
    }

    public void testAssignLongMatrix1DLongLongFunction() {
        LongMatrix1D Acopy = A.copy();
        A.assign(B, LongFunctions.plus);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(Acopy.getQuick(i) + B.getQuick(i), A.getQuick(i));
        }
    }

    public void testAssignLongProcedureLong() {
        LongProcedure procedure = new LongProcedure() {
            public boolean apply(long element) {
                if (Math.abs(element) > 1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        LongMatrix1D Acopy = A.copy();
        A.assign(procedure, -1);
        for (int i = 0; i < (int) A.size(); i++) {
            if (Math.abs(Acopy.getQuick(i)) > 1) {
                assertEquals(-1, A.getQuick(i));
            } else {
                assertEquals(Acopy.getQuick(i), A.getQuick(i));
            }
        }
    }

    public void testAssignLongProcedureLongFunction() {
        LongProcedure procedure = new LongProcedure() {
            public boolean apply(long element) {
                if (Math.abs(element) > 1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        LongMatrix1D Acopy = A.copy();
        A.assign(procedure, LongFunctions.neg);
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

    public void testEqualsLong() {
        long value = 1;
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
        long[] maxAndLoc = A.getMaxLocation();
        assertEquals(7, maxAndLoc[0]);
        assertEquals((int) A.size() / 3, (int) maxAndLoc[1]);
    }

    public void testMinLocation() {
        A.assign(0);
        A.setQuick((int) A.size() / 3, -7);
        A.setQuick((int) A.size() / 2, -1);
        long[] minAndLoc = A.getMinLocation();
        assertEquals(-7, minAndLoc[0]);
        assertEquals((int) A.size() / 3, (int) minAndLoc[1]);
    }

    public void testGetNegativeValuesIntArrayListLongArrayList() {
        A.assign(0);
        A.setQuick((int) A.size() / 3, -7);
        A.setQuick((int) A.size() / 2, -1);
        IntArrayList indexList = new IntArrayList();
        LongArrayList valueList = new LongArrayList();
        A.getNegativeValues(indexList, valueList);
        assertEquals(2, indexList.size());
        assertEquals(2, valueList.size());
        assertTrue(indexList.contains((int) A.size() / 3));
        assertTrue(indexList.contains((int) A.size() / 2));
        assertTrue(valueList.contains(-7));
        assertTrue(valueList.contains(-1));
    }

    public void testGetNonZerosIntArrayListLongArrayList() {
        A.assign(0);
        A.setQuick((int) A.size() / 3, 7);
        A.setQuick((int) A.size() / 2, 1);
        IntArrayList indexList = new IntArrayList();
        LongArrayList valueList = new LongArrayList();
        A.getNonZeros(indexList, valueList);
        assertEquals(2, indexList.size());
        assertEquals(2, valueList.size());
        assertTrue(indexList.contains((int) A.size() / 3));
        assertTrue(indexList.contains((int) A.size() / 2));
        assertTrue(valueList.contains(7));
        assertTrue(valueList.contains(1));
    }

    public void testGetPositiveValuesIntArrayListLongArrayList() {
        A.assign(0);
        A.setQuick((int) A.size() / 3, 7);
        A.setQuick((int) A.size() / 2, 1);
        IntArrayList indexList = new IntArrayList();
        LongArrayList valueList = new LongArrayList();
        A.getPositiveValues(indexList, valueList);
        assertEquals(2, indexList.size());
        assertEquals(2, valueList.size());
        assertTrue(indexList.contains((int) A.size() / 3));
        assertTrue(indexList.contains((int) A.size() / 2));
        assertTrue(valueList.contains(7));
        assertTrue(valueList.contains(1));
    }

    public void testToArray() {
        long[] array = A.toArray();
        assertTrue((int) A.size() == array.length);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(array[i], A.getQuick(i));
        }
    }

    public void testToArrayLongArray() {
        long[] array = new long[(int) A.size()];
        A.toArray(array);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(A.getQuick(i), array[i]);
        }
    }

    public void testReshapeIntInt() {
        int rows = 10;
        int columns = 17;
        LongMatrix2D B = A.reshape(rows, columns);
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
        LongMatrix3D B = A.reshape(slices, rows, columns);
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
        LongMatrix1D Acopy = A.copy();
        LongMatrix1D Bcopy = B.copy();
        A.swap(B);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(Bcopy.getQuick(i), A.getQuick(i));
            assertEquals(Acopy.getQuick(i), B.getQuick(i));
        }
    }

    public void testViewFlip() {
        LongMatrix1D b = A.viewFlip();
        assertEquals((int) A.size(), b.size());
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(A.getQuick(i), b.getQuick((int) A.size() - 1 - i));
        }
    }

    public void testViewPart() {
        LongMatrix1D b = A.viewPart(15, 11);
        for (int i = 0; i < 11; i++) {
            assertEquals(A.getQuick(15 + i), b.getQuick(i));
        }
    }

    public void testViewSelectionLongProcedure() {
        LongMatrix1D b = A.viewSelection(new LongProcedure() {
            public boolean apply(long element) {
                return element % 2 == 0;
            }
        });
        for (int i = 0; i < b.size(); i++) {
            long el = b.getQuick(i);
            if (el % 2 != 0) {
                fail();
            }
        }
    }

    public void testViewSelectionIntArray() {
        int[] indexes = new int[] { 5, 11, 22, 37, 101 };
        LongMatrix1D b = A.viewSelection(indexes);
        for (int i = 0; i < indexes.length; i++) {
            assertEquals(A.getQuick(indexes[i]), b.getQuick(i));
        }
    }

    public void testViewSorted() {
        LongMatrix1D b = A.viewSorted();
        for (int i = 0; i < (int) A.size() - 1; i++) {
            assertTrue(b.getQuick(i + 1) >= b.getQuick(i));
        }
    }

    public void testViewStrides() {
        int stride = 3;
        LongMatrix1D b = A.viewStrides(stride);
        for (int i = 0; i < b.size(); i++) {
            assertEquals(A.getQuick(i * stride), b.getQuick(i));
        }
    }

    public void testZDotProductLongMatrix1D() {
        long product = A.zDotProduct(B);
        long expected = 0;
        for (int i = 0; i < (int) A.size(); i++) {
            expected += A.getQuick(i) * B.getQuick(i);
        }
        assertEquals(expected, product);
    }

    public void testZDotProductLongMatrix1DIntInt() {
        long product = A.zDotProduct(B, 5, (int) B.size() - 10);
        long expected = 0;
        for (int i = 5; i < (int) A.size() - 5; i++) {
            expected += A.getQuick(i) * B.getQuick(i);
        }
        assertEquals(expected, product);
    }

    @Test
    public void testZDotProductLongMatrix1DIntIntIntArrayList() {
        IntArrayList indexList = new IntArrayList();
        LongArrayList valueList = new LongArrayList();
        B.getNonZeros(indexList, valueList);
        long product = A.zDotProduct(B, 5, (int) B.size() - 10, indexList);
        long expected = 0;
        for (int i = 5; i < (int) A.size() - 5; i++) {
            expected += A.getQuick(i) * B.getQuick(i);
        }
        assertEquals(expected, product);
    }

    public void testZSum() {
        long sum = A.zSum();
        long expected = 0;
        for (int i = 0; i < (int) A.size(); i++) {
            expected += A.getQuick(i);
        }
        assertEquals(expected, sum);
    }

}
