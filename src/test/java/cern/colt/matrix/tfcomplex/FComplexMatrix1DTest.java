package cern.colt.matrix.tfcomplex;

import java.util.ArrayList;

import junit.framework.TestCase;
import cern.colt.function.tfcomplex.FComplexProcedure;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tfloat.FloatFactory1D;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.jet.math.tfcomplex.FComplex;
import cern.jet.math.tfcomplex.FComplexFunctions;
import edu.emory.mathcs.utils.pc.ConcurrencyUtils;

public abstract class FComplexMatrix1DTest extends TestCase {
    /**
     * Matrix to test
     */
    protected FComplexMatrix1D A;

    /**
     * Matrix of the same size as a
     */
    protected FComplexMatrix1D B;

    protected int SIZE = 2 * 17 * 5;

    protected float TOL = 1e-3f;

    protected cern.jet.math.tfloat.FloatFunctions F = cern.jet.math.tfloat.FloatFunctions.functions;

    /**
     * Constructor for FloatMatrix1DTest
     */
    public FComplexMatrix1DTest(String arg0) {
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
            A.setQuick(i, new float[] { (float) Math.random(), (float) Math.random() });
        }

        for (int i = 0; i < (int) B.size(); i++) {
            B.setQuick(i, new float[] { (float) Math.random(), (float) Math.random() });
        }
    }

    protected void tearDown() throws Exception {
        A = B = null;
    }

    public void testAggregateFloatFloatFunctionFloatFunction() {
        float[] expected = new float[2];
        for (int i = 0; i < (int) A.size(); i++) {
            expected = FComplex.plus(expected, FComplex.square(A.getQuick(i)));
        }
        float[] result = A.aggregate(FComplexFunctions.plus, FComplexFunctions.square);
        assertEquals(expected, result, TOL);
    }

    public void testAggregateComplexMatrix1FComplexComplexFunctionComplexComplexFunction() {
        float[] actual = A.aggregate(B, FComplexFunctions.plus, FComplexFunctions.mult);
        float[] expected = new float[2];
        for (int i = 0; i < (int) A.size(); i++) {
            expected = FComplex.plus(expected, FComplex.mult(A.getQuick(i), B.getQuick(i)));
        }
        assertEquals(expected, actual, TOL);
    }

    public void testAssignComplexComplexFunction() {
        FComplexMatrix1D Acopy = A.copy();
        A.assign(FComplexFunctions.acos);
        for (int i = 0; i < (int) A.size(); i++) {
            float[] expected = FComplex.acos(Acopy.getQuick(i));
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

    public void testAssignComplexMatrix1FComplexComplexFunction() {
        FComplexMatrix1D Acopy = A.copy();
        A.assign(B, FComplexFunctions.div);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(FComplex.div(Acopy.getQuick(i), B.getQuick(i)), A.getQuick(i), TOL);
        }
    }

    public void testAssignComplexProcedureComplexComplexFunction() {
        FComplexProcedure procedure = new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        FComplexMatrix1D Acopy = A.copy();
        A.assign(procedure, FComplexFunctions.tan);
        for (int i = 0; i < (int) A.size(); i++) {
            if (FComplex.abs(Acopy.getQuick(i)) > 0.1) {
                assertEquals(FComplex.tan(Acopy.getQuick(i)), A.getQuick(i), TOL);
            } else {
                assertEquals(Acopy.getQuick(i), A.getQuick(i), TOL);
            }
        }
    }

    public void testAssignComplexProcedureFloatArray() {
        FComplexProcedure procedure = new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 0.1) {
                    return true;
                } else {
                    return false;
                }
            }
        };
        FComplexMatrix1D Acopy = A.copy();
        float[] value = new float[] { -1, -1 };
        A.assign(procedure, value);
        for (int i = 0; i < (int) A.size(); i++) {
            if (FComplex.abs(Acopy.getQuick(i)) > 0.1) {
                assertEquals(value, A.getQuick(i), TOL);
            } else {
                assertEquals(Acopy.getQuick(i), A.getQuick(i), TOL);
            }
        }
    }

    public void testAssignComplexRealFunction() {
        FComplexMatrix1D Acopy = A.copy();
        A.assign(FComplexFunctions.abs);
        for (int i = 0; i < (int) A.size(); i++) {
            float[] elem = A.getQuick(i);
            assertEquals(FComplex.abs(Acopy.getQuick(i)), elem[0], TOL);
            assertEquals(0, elem[1], TOL);
        }
    }

    public void testAssignFloatArray() {
        float[] expected = new float[2 * (int) A.size()];
        for (int i = 0; i < 2 * (int) A.size(); i++) {
            expected[i] = (float) Math.random();
        }
        A.assign(expected);
        for (int i = 0; i < (int) A.size(); i++) {
            float[] elem = A.getQuick(i);
            assertEquals(expected[2 * i], elem[0], TOL);
            assertEquals(expected[2 * i + 1], elem[1], TOL);
        }

    }

    public void testAssignFloatFloat() {
        float re = (float) Math.random();
        float im = (float) Math.random();
        A.assign(re, im);
        for (int i = 0; i < (int) A.size(); i++) {
            float[] elem = A.getQuick(i);
            assertEquals(re, elem[0], TOL);
            assertEquals(im, elem[1], TOL);
        }
    }

    public void testAssignImaginary() {
        FComplexMatrix1D Acopy = A.copy();
        FloatMatrix1D Im = FloatFactory1D.dense.random((int) A.size());
        A.assignImaginary(Im);
        for (int i = 0; i < (int) A.size(); i++) {
            float[] elem = A.getQuick(i);
            assertEquals(Acopy.getQuick(i)[0], elem[0], TOL);
            assertEquals(Im.getQuick(i), elem[1], TOL);
        }
    }

    public void testAssignReal() {
        FComplexMatrix1D Acopy = A.copy();
        FloatMatrix1D Re = FloatFactory1D.dense.random((int) A.size());
        A.assignReal(Re);
        for (int i = 0; i < (int) A.size(); i++) {
            float[] elem = A.getQuick(i);
            assertEquals(Acopy.getQuick(i)[1], elem[1], TOL);
            assertEquals(Re.getQuick(i), elem[0], TOL);
        }
    }

    public void testCardinality() {
        int card = A.cardinality();
        assertEquals((int) A.size(), card);
    }

    public void testEqualsFloat() {
        float[] value = new float[] { 1, 2 };
        A.assign(value[0], value[1]);
        boolean eq = A.equals(value);
        assertTrue(eq);
        eq = A.equals(new float[] { 2, 2 });
        assertFalse(eq);
    }

    public void testEqualsObject() {
        boolean eq = A.equals(A);
        assertTrue(eq);
        eq = A.equals(B);
        assertFalse(eq);
    }

    public void testGetImaginaryPart() {
        FloatMatrix1D Im = A.getImaginaryPart();
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(A.getQuick(i)[1], Im.getQuick(i), TOL);
        }
    }

    public void testGetRealPart() {
        FloatMatrix1D Re = A.getRealPart();
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(A.getQuick(i)[0], Re.getQuick(i), TOL);
        }
    }

    public void testGetNonZerosIntArrayListArrayListOffloat() {
        IntArrayList indexList = new IntArrayList();
        ArrayList<float[]> valueList = new ArrayList<float[]>();
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
        FComplexMatrix2D B = A.reshape(rows, columns);
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
        FComplexMatrix3D B = A.reshape(slices, rows, columns);
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
        FComplexMatrix1D Acopy = A.copy();
        FComplexMatrix1D Bcopy = B.copy();
        A.swap(B);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(Bcopy.getQuick(i), A.getQuick(i), TOL);
            assertEquals(Acopy.getQuick(i), B.getQuick(i), TOL);
        }
    }

    public void testToArray() {
        float[] array = A.toArray();
        for (int i = 0; i < (int) A.size(); i++) {
            float[] elem = A.getQuick(i);
            assertEquals(elem[0], array[2 * i], TOL);
            assertEquals(elem[1], array[2 * i + 1], TOL);
        }
    }

    public void testToArrayFloatArray() {
        float[] array = new float[2 * (int) A.size()];
        A.toArray(array);
        for (int i = 0; i < (int) A.size(); i++) {
            float[] elem = A.getQuick(i);
            assertEquals(elem[0], array[2 * i], TOL);
            assertEquals(elem[1], array[2 * i + 1], TOL);
        }
    }

    public void testViewFlip() {
        FComplexMatrix1D B = A.viewFlip();
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(A.getQuick((int) A.size() - 1 - i), B.getQuick(i), TOL);
        }
    }

    public void testViewPart() {
        FComplexMatrix1D B = A.viewPart((int) A.size() / 2, (int) A.size() / 3);
        for (int i = 0; i < (int) A.size() / 3; i++) {
            assertEquals(A.getQuick((int) A.size() / 2 + i), B.getQuick(i), TOL);
        }
    }

    public void testViewSelectionComplexProcedure() {
        FComplexMatrix1D B = A.viewSelection(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (element[0] < element[1]) {
                    return true;
                } else {
                    return false;
                }
            }
        });
        for (int i = 0; i < B.size(); i++) {
            float[] el = B.getQuick(i);
            if (el[0] >= el[1]) {
                fail();
            }
        }
    }

    public void testViewSelectionIntArray() {
        int[] indexes = new int[] { (int) A.size() / 6, (int) A.size() / 5, (int) A.size() / 4, (int) A.size() / 3,
                (int) A.size() / 2 };
        FComplexMatrix1D B = A.viewSelection(indexes);
        for (int i = 0; i < indexes.length; i++) {
            assertEquals(A.getQuick(indexes[i]), B.getQuick(i), TOL);
        }
    }

    public void testViewStrides() {
        int stride = 3;
        FComplexMatrix1D B = A.viewStrides(stride);
        for (int i = 0; i < B.size(); i++) {
            assertEquals(A.getQuick(i * stride), B.getQuick(i), TOL);
        }
    }

    public void testZDotProductComplexMatrix1D() {
        float[] actual = A.zDotProduct(B);
        float[] expected = new float[2];
        for (int i = 0; i < (int) A.size(); i++) {
            expected = FComplex.plus(expected, FComplex.mult(A.getQuick(i), FComplex.conj(B.getQuick(i))));
        }
        assertEquals(expected, actual, TOL);
    }

    public void testZDotProductComplexMatrix1DIntInt() {
        float[] actual = A.zDotProduct(B, 5, (int) B.size() - 10);
        float[] expected = new float[2];
        for (int i = 5; i < (int) A.size() - 5; i++) {
            expected = FComplex.plus(expected, FComplex.mult(A.getQuick(i), FComplex.conj(B.getQuick(i))));
        }
        assertEquals(expected, actual, TOL);
    }

    public void testZDotProductComplexMatrix1DIntIntIntArrayList() {
        IntArrayList indexList = new IntArrayList();
        ArrayList<float[]> valueList = new ArrayList<float[]>();
        B.getNonZeros(indexList, valueList);
        float[] actual = A.zDotProduct(B, 5, (int) B.size() - 10, indexList);
        float[] expected = new float[2];
        for (int i = 5; i < (int) A.size() - 5; i++) {
            expected = FComplex.plus(expected, FComplex.mult(A.getQuick(i), FComplex.conj(B.getQuick(i))));
        }
        assertEquals(expected, actual, TOL);
    }

    public void testZSum() {
        float[] actual = A.zSum();
        float[] expected = new float[2];
        for (int i = 0; i < (int) A.size(); i++) {
            expected = FComplex.plus(expected, A.getQuick(i));
        }
        assertEquals(expected, actual, TOL);
    }

    protected void assertEquals(float[] expected, float[] actual, float tol) {
        for (int i = 0; i < actual.length; i++) {
            assertEquals(expected[i], actual[i], tol);
        }
    }

}
