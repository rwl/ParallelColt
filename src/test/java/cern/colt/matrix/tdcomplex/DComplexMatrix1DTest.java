package cern.colt.matrix.tdcomplex;

import java.util.ArrayList;

import junit.framework.TestCase;
import cern.colt.function.tdcomplex.DComplexProcedure;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tdouble.DoubleFactory1D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.jet.math.tdcomplex.DComplex;
import cern.jet.math.tdcomplex.DComplexFunctions;
import edu.emory.mathcs.utils.pc.ConcurrencyUtils;

public abstract class DComplexMatrix1DTest extends TestCase {
    /**
     * Matrix to test
     */
    protected DComplexMatrix1D A;

    /**
     * Matrix of the same size as a
     */
    protected DComplexMatrix1D B;

    protected int SIZE = 2 * 17 * 5;

    protected double TOL = 1e-10;

    protected cern.jet.math.tdouble.DoubleFunctions F = cern.jet.math.tdouble.DoubleFunctions.functions;

    /**
     * Constructor for DoubleMatrix1DTest
     */
    public DComplexMatrix1DTest(String arg0) {
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
            A.setQuick(i, new double[] { Math.random(), Math.random() });
        }

        for (int i = 0; i < (int) A.size(); i++) {
            B.setQuick(i, new double[] { Math.random(), Math.random() });
        }
    }

    protected void tearDown() throws Exception {
        A = B = null;
    }

    public void testAggregateDoubleDoubleFunctionDoubleFunction() {
        double[] expected = new double[2];
        for (int i = 0; i < (int) A.size(); i++) {
            expected = DComplex.plus(expected, DComplex.square(A.getQuick(i)));
        }
        double[] result = A.aggregate(DComplexFunctions.plus, DComplexFunctions.square);
        assertEquals(expected, result, TOL);
    }

    public void testAggregateComplexMatrix1DComplexComplexFunctionComplexComplexFunction() {
        double[] actual = A.aggregate(B, DComplexFunctions.plus, DComplexFunctions.mult);
        double[] expected = new double[2];
        for (int i = 0; i < (int) A.size(); i++) {
            expected = DComplex.plus(expected, DComplex.mult(A.getQuick(i), B.getQuick(i)));
        }
        assertEquals(expected, actual, TOL);
    }

    public void testAssignComplexComplexFunction() {
        DComplexMatrix1D Acopy = A.copy();
        A.assign(DComplexFunctions.acos);
        for (int i = 0; i < (int) A.size(); i++) {
            double[] expected = DComplex.acos(Acopy.getQuick(i));
            assertEquals(expected, A.getQuick(i), TOL);
        }
    }

    public void testAssignComplexMatrix1D() {
        A.assign(B);
        assertTrue(A.size() == B.size());
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(B.getQuick(i), A.getQuick(i), TOL);
        }
    }

    public void testAssignComplexMatrix1DComplexComplexFunction() {
        DComplexMatrix1D Acopy = A.copy();
        A.assign(B, DComplexFunctions.div);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(DComplex.div(Acopy.getQuick(i), B.getQuick(i)), A.getQuick(i), TOL);
        }
    }

    public void testAssignComplexProcedureComplexComplexFunction() {
        DComplexProcedure procedure = new DComplexProcedure() {
            public boolean apply(double[] element) {
                if (DComplex.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        DComplexMatrix1D Acopy = A.copy();
        A.assign(procedure, DComplexFunctions.tan);
        for (int i = 0; i < (int) A.size(); i++) {
            if (DComplex.abs(Acopy.getQuick(i)) > 0.1) {
                assertEquals(DComplex.tan(Acopy.getQuick(i)), A.getQuick(i), TOL);
            } else {
                assertEquals(Acopy.getQuick(i), A.getQuick(i), TOL);
            }
        }
    }

    public void testAssignComplexProcedureDoubleArray() {
        DComplexProcedure procedure = new DComplexProcedure() {
            public boolean apply(double[] element) {
                if (DComplex.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        DComplexMatrix1D Acopy = A.copy();
        double[] value = new double[] { -1, -1 };
        A.assign(procedure, value);
        for (int i = 0; i < (int) A.size(); i++) {
            if (DComplex.abs(Acopy.getQuick(i)) > 0.1) {
                assertEquals(value, A.getQuick(i), TOL);
            } else {
                assertEquals(Acopy.getQuick(i), A.getQuick(i), TOL);
            }
        }
    }

    public void testAssignComplexRealFunction() {
        DComplexMatrix1D Acopy = A.copy();
        A.assign(DComplexFunctions.abs);
        for (int i = 0; i < (int) A.size(); i++) {
            double[] elem = A.getQuick(i);
            assertEquals(DComplex.abs(Acopy.getQuick(i)), elem[0], TOL);
            assertEquals(0, elem[1], TOL);
        }
    }

    public void testAssignDoubleArray() {
        double[] expected = new double[2 * (int) A.size()];
        for (int i = 0; i < 2 * (int) A.size(); i++) {
            expected[i] = Math.random();
        }
        A.assign(expected);
        for (int i = 0; i < (int) A.size(); i++) {
            double[] elem = A.getQuick(i);
            assertEquals(expected[2 * i], elem[0], TOL);
            assertEquals(expected[2 * i + 1], elem[1], TOL);
        }

    }

    public void testAssignDoubleDouble() {
        double re = Math.random();
        double im = Math.random();
        A.assign(re, im);
        for (int i = 0; i < (int) A.size(); i++) {
            double[] elem = A.getQuick(i);
            assertEquals(re, elem[0], TOL);
            assertEquals(im, elem[1], TOL);
        }
    }

    public void testAssignImaginary() {
        DComplexMatrix1D Acopy = A.copy();
        DoubleMatrix1D Im = DoubleFactory1D.dense.random((int) A.size());
        A.assignImaginary(Im);
        for (int i = 0; i < (int) A.size(); i++) {
            double[] elem = A.getQuick(i);
            assertEquals(Acopy.getQuick(i)[0], elem[0], TOL);
            assertEquals(Im.getQuick(i), elem[1], TOL);
        }
    }

    public void testAssignReal() {
        DComplexMatrix1D Acopy = A.copy();
        DoubleMatrix1D Re = DoubleFactory1D.dense.random((int) A.size());
        A.assignReal(Re);
        for (int i = 0; i < (int) A.size(); i++) {
            double[] elem = A.getQuick(i);
            assertEquals(Acopy.getQuick(i)[1], elem[1], TOL);
            assertEquals(Re.getQuick(i), elem[0], TOL);
        }
    }

    public void testCardinality() {
        int card = A.cardinality();
        assertEquals((int) A.size(), card);
    }

    public void testEqualsDouble() {
        double[] value = new double[] { 1, 2 };
        A.assign(value[0], value[1]);
        boolean eq = A.equals(value);
        assertTrue(eq);
        eq = A.equals(new double[] { 2, 2 });
        assertFalse(eq);
    }

    public void testEqualsObject() {
        boolean eq = A.equals(A);
        assertTrue(eq);
        eq = A.equals(B);
        assertFalse(eq);
    }

    public void testGetImaginaryPart() {
        DoubleMatrix1D Im = A.getImaginaryPart();
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(A.getQuick(i)[1], Im.getQuick(i), TOL);
        }
    }

    public void testGetRealPart() {
        DoubleMatrix1D Re = A.getRealPart();
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(A.getQuick(i)[0], Re.getQuick(i), TOL);
        }
    }

    public void testGetNonZerosIntArrayListArrayListOfdouble() {
        IntArrayList indexList = new IntArrayList();
        ArrayList<double[]> valueList = new ArrayList<double[]>();
        A.getNonZeros(indexList, valueList);
        assertEquals((int) A.size(), indexList.size());
        assertEquals((int) A.size(), valueList.size());
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(A.getQuick(indexList.get(i)), valueList.get(i), TOL);
            assertTrue(valueList.get(i)[0] != 0 || valueList.get(i)[1] != 0);
        }
    }

    public void testReshapeIntInt() {
        int rows = 10;
        int columns = 17;
        DComplexMatrix2D B = A.reshape(rows, columns);
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
        DComplexMatrix3D B = A.reshape(slices, rows, columns);
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
        DComplexMatrix1D Acopy = A.copy();
        DComplexMatrix1D Bcopy = B.copy();
        A.swap(B);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(Bcopy.getQuick(i), A.getQuick(i), TOL);
            assertEquals(Acopy.getQuick(i), B.getQuick(i), TOL);
        }
    }

    public void testToArray() {
        double[] array = A.toArray();
        for (int i = 0; i < (int) A.size(); i++) {
            double[] elem = A.getQuick(i);
            assertEquals(elem[0], array[2 * i], TOL);
            assertEquals(elem[1], array[2 * i + 1], TOL);
        }
    }

    public void testToArrayDoubleArray() {
        double[] array = new double[2 * (int) A.size()];
        A.toArray(array);
        for (int i = 0; i < (int) A.size(); i++) {
            double[] elem = A.getQuick(i);
            assertEquals(elem[0], array[2 * i], TOL);
            assertEquals(elem[1], array[2 * i + 1], TOL);
        }
    }

    public void testViewFlip() {
        DComplexMatrix1D B = A.viewFlip();
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(A.getQuick((int) A.size() - 1 - i), B.getQuick(i), TOL);
        }
    }

    public void testViewPart() {
        DComplexMatrix1D B = A.viewPart((int) A.size() / 2, (int) A.size() / 3);
        for (int i = 0; i < (int) A.size() / 3; i++) {
            assertEquals(A.getQuick((int) A.size() / 2 + i), B.getQuick(i), TOL);
        }
    }

    public void testViewSelectionComplexProcedure() {
        DComplexMatrix1D B = A.viewSelection(new DComplexProcedure() {
            public boolean apply(double[] element) {
                if (element[0] < element[1]) {
                    return true;
                } else {
                    return false;
                }
            }
        });
        for (int i = 0; i < B.size(); i++) {
            double[] el = B.getQuick(i);
            if (el[0] >= el[1]) {
                fail();
            }
        }
    }

    public void testViewSelectionIntArray() {
        int[] indexes = new int[] { (int) A.size() / 6, (int) A.size() / 5, (int) A.size() / 4, (int) A.size() / 3,
                (int) A.size() / 2 };
        DComplexMatrix1D B = A.viewSelection(indexes);
        for (int i = 0; i < indexes.length; i++) {
            assertEquals(A.getQuick(indexes[i]), B.getQuick(i), TOL);
        }
    }

    public void testViewStrides() {
        int stride = 3;
        DComplexMatrix1D B = A.viewStrides(stride);
        for (int i = 0; i < B.size(); i++) {
            assertEquals(A.getQuick(i * stride), B.getQuick(i), TOL);
        }
    }

    public void testZDotProductComplexMatrix1D() {
        double[] actual = A.zDotProduct(B);
        double[] expected = new double[2];
        for (int i = 0; i < (int) A.size(); i++) {
            expected = DComplex.plus(expected, DComplex.mult(DComplex.conj(B.getQuick(i)), A.getQuick(i)));
        }
        assertEquals(expected, actual, TOL);
    }

    public void testZDotProductComplexMatrix1DIntInt() {
        double[] actual = A.zDotProduct(B, 5, (int) B.size() - 10);
        double[] expected = new double[2];
        for (int i = 5; i < (int) A.size() - 5; i++) {
            expected = DComplex.plus(expected, DComplex.mult(DComplex.conj(B.getQuick(i)), A.getQuick(i)));
        }
        assertEquals(expected, actual, TOL);
    }

    public void testZDotProductComplexMatrix1DIntIntIntArrayList() {
        IntArrayList indexList = new IntArrayList();
        ArrayList<double[]> valueList = new ArrayList<double[]>();
        B.getNonZeros(indexList, valueList);
        double[] actual = A.zDotProduct(B, 5, (int) B.size() - 10, indexList);
        double[] expected = new double[2];
        for (int i = 5; i < (int) A.size() - 5; i++) {
            expected = DComplex.plus(expected, DComplex.mult(A.getQuick(i), DComplex.conj(B.getQuick(i))));
        }
        assertEquals(expected, actual, TOL);
    }

    public void testZSum() {
        double[] actual = A.zSum();
        double[] expected = new double[2];
        for (int i = 0; i < (int) A.size(); i++) {
            expected = DComplex.plus(expected, A.getQuick(i));
        }
        assertEquals(expected, actual, TOL);
    }

    protected void assertEquals(double[] expected, double[] actual, double tol) {
        for (int i = 0; i < actual.length; i++) {
            assertEquals(expected[i], actual[i], tol);
        }
    }

}
