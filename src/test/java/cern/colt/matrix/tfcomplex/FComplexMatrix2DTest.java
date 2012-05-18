package cern.colt.matrix.tfcomplex;

import java.util.ArrayList;

import junit.framework.TestCase;
import cern.colt.function.tfcomplex.FComplexProcedure;
import cern.colt.function.tfcomplex.IntIntFComplexFunction;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix1D;
import cern.colt.matrix.tfloat.FloatFactory2D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.jet.math.tfcomplex.FComplex;
import cern.jet.math.tfcomplex.FComplexFunctions;
import edu.emory.mathcs.utils.ConcurrencyUtils;

public abstract class FComplexMatrix2DTest extends TestCase {
    /**
     * Matrix to test
     */
    protected FComplexMatrix2D A;

    /**
     * Matrix of the same size as A
     */
    protected FComplexMatrix2D B;

    /**
     * Matrix of the size A.columns() x A.rows()
     */
    protected FComplexMatrix2D Bt;

    protected int NROWS = 13;

    protected int NCOLUMNS = 17;

    protected float TOL = 1e-3f;

    /**
     * Constructor for FloatMatrix2DTest
     */
    public FComplexMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void setUp() throws Exception {
        createMatrices();
        populateMatrices();
    }

    protected abstract void createMatrices() throws Exception;

    protected void populateMatrices() {
        ConcurrencyUtils.setThreadsBeginN_2D(1);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                A.setQuick(r, c, new float[] { (float) Math.random(), (float) Math.random() });
            }
        }

        for (int r = 0; r < B.rows(); r++) {
            for (int c = 0; c < B.columns(); c++) {
                B.setQuick(r, c, new float[] { (float) Math.random(), (float) Math.random() });
            }
        }

        for (int r = 0; r < Bt.rows(); r++) {
            for (int c = 0; c < Bt.columns(); c++) {
                Bt.setQuick(r, c, new float[] { (float) Math.random(), (float) Math.random() });
            }
        }
    }

    protected void tearDown() throws Exception {
        A = B = Bt = null;
    }

    public void testAggregateComplexComplexComplexFunctionComplexComplexFunction() {
        float[] actual = A.aggregate(FComplexFunctions.plus, FComplexFunctions.square);
        float[] expected = new float[2];
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                expected = FComplex.plus(expected, FComplex.square(A.getQuick(r, c)));
            }
        }
        assertEquals(expected, actual, TOL);
    }

    public void testAggregateComplexMatrix2FComplexComplexComplexFunctionComplexComplexComplexFunction() {
        float[] actual = A.aggregate(B, FComplexFunctions.plus, FComplexFunctions.mult);
        float[] expected = new float[2];
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                expected = FComplex.plus(expected, FComplex.mult(A.getQuick(r, c), B.getQuick(r, c)));
            }
        }
        assertEquals(expected, actual, TOL);
    }

    public void testAssignComplexComplexFunction() {
        FComplexMatrix2D Acopy = A.copy();
        A.assign(FComplexFunctions.acos);
        float[] tmp;
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                tmp = FComplex.acos(Acopy.getQuick(r, c));
                assertEquals(tmp, A.getQuick(r, c), TOL);
            }
        }
    }

    public void testAssignComplexMatrix2D() {
        A.assign(B);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(B.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testAssignComplexMatrix2FComplexComplexComplexFunction() {
        FComplexMatrix2D Acopy = A.copy();
        A.assign(B, FComplexFunctions.div);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(FComplex.div(Acopy.getQuick(r, c), B.getQuick(r, c)), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testAssignComplexProcedureComplexComplexFunction() {
        FComplexMatrix2D Acopy = A.copy();
        A.assign(new FComplexProcedure() {
            public boolean apply(float[] element) {
                if (FComplex.abs(element) > 3) {
                    return true;
                } else {
                    return false;
                }
            }
        }, FComplexFunctions.tan);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                if (FComplex.abs(Acopy.getQuick(r, c)) > 3) {
                    assertEquals(FComplex.tan(Acopy.getQuick(r, c)), A.getQuick(r, c), TOL);
                } else {
                    assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
                }
            }
        }
    }

    public void testAssignComplexProcedureFloatArray() {
        FComplexMatrix2D Acopy = A.copy();
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
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                if (FComplex.abs(A.getQuick(r, c)) > 3) {
                    assertEquals(value, A.getQuick(r, c), TOL);
                } else {
                    assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
                }
            }
        }
    }

    public void testAssignComplexRealFunction() {
        FComplexMatrix2D Acopy = A.copy();
        A.assign(FComplexFunctions.abs);
        float[] tmp;
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                tmp = A.getQuick(r, c);
                assertEquals(FComplex.abs(Acopy.getQuick(r, c)), tmp[0], TOL);
                assertEquals(0, tmp[1], TOL);
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
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                float[] elem = A.getQuick(r, c);
                assertEquals(expected[idx], elem[0], TOL);
                assertEquals(expected[idx + 1], elem[1], TOL);
                idx += 2;
            }
        }
    }

    public void testAssignFloatArrayArray() {
        float[][] expected = new float[A.rows()][2 * A.columns()];
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < 2 * A.columns(); c++) {
                expected[r][c] = (float) Math.random();
            }
        }
        A.assign(expected);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                float[] elem = A.getQuick(r, c);
                assertEquals(expected[r][2 * c], elem[0], TOL);
                assertEquals(expected[r][2 * c + 1], elem[1], TOL);
            }
        }
    }

    public void testAssignFloatFloat() {
        float[] value = new float[] { (float) Math.random(), (float) Math.random() };
        A.assign(value[0], value[1]);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                float[] elem = A.getQuick(r, c);
                assertEquals(value, elem, TOL);
            }
        }
    }

    public void testAssignImaginary() {
        FloatMatrix2D Im = FloatFactory2D.dense.random(A.rows(), A.columns());
        FComplexMatrix2D Acopy = A.copy();
        A.assignImaginary(Im);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(Acopy.getQuick(r, c)[0], A.getQuick(r, c)[0], TOL);
                assertEquals(Im.getQuick(r, c), A.getQuick(r, c)[1], TOL);
            }
        }
    }

    public void testAssignReal() {
        FloatMatrix2D Re = FloatFactory2D.dense.random(A.rows(), A.columns());
        FComplexMatrix2D Acopy = A.copy();
        A.assignReal(Re);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(Acopy.getQuick(r, c)[1], A.getQuick(r, c)[1], TOL);
                assertEquals(Re.getQuick(r, c), A.getQuick(r, c)[0], TOL);
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
        assertEquals(true, eq);
        eq = A.equals(new float[] { value[0] + 1, value[1] + 1 });
        assertEquals(false, eq);
    }

    public void testEqualsObject() {
        boolean eq = A.equals(A);
        assertEquals(true, eq);
        eq = A.equals(B);
        assertEquals(false, eq);
    }

    public void testForEachNonZero() {
        FComplexMatrix2D Acopy = A.copy();
        IntIntFComplexFunction function = new IntIntFComplexFunction() {
            public float[] apply(int first, int second, float[] third) {
                return FComplex.sqrt(third);
            }
        };
        A.forEachNonZero(function);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(FComplex.sqrt(Acopy.getQuick(r, c)), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testGetConjugateTranspose() {
        FComplexMatrix2D Aconj = A.getConjugateTranspose();
        assertEquals(A.rows(), Aconj.columns());
        assertEquals(A.columns(), Aconj.rows());
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(r, c)[0], Aconj.getQuick(c, r)[0], TOL);
                assertEquals(-A.getQuick(r, c)[1], Aconj.getQuick(c, r)[1], TOL);
            }
        }
    }

    public void testGetImaginaryPart() {
        FloatMatrix2D Im = A.getImaginaryPart();
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(r, c)[1], Im.getQuick(r, c), TOL);
            }
        }
    }

    public void testGetNonZeros() {
        IntArrayList rowList = new IntArrayList();
        IntArrayList colList = new IntArrayList();
        ArrayList<float[]> valueList = new ArrayList<float[]>();
        A.getNonZeros(rowList, colList, valueList);
        assertEquals(A.size(), rowList.size());
        assertEquals(A.size(), colList.size());
        assertEquals(A.size(), valueList.size());
        int idx = 0;
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(rowList.get(idx), colList.get(idx)), valueList.get(idx), TOL);
                idx++;
            }
        }
    }

    public void testGetRealPart() {
        FloatMatrix2D Re = A.getRealPart();
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(r, c)[0], Re.getQuick(r, c), TOL);
            }
        }
    }

    public void testToArray() {
        float[][] array = A.toArray();
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(r, c)[0], array[r][2 * c], TOL);
                assertEquals(A.getQuick(r, c)[1], array[r][2 * c + 1], TOL);
            }
        }
    }

    public void testVectorize() {
        FComplexMatrix1D B = A.vectorize();
        int idx = 0;
        for (int c = 0; c < A.columns(); c++) {
            for (int r = 0; r < A.rows(); r++) {
                assertEquals(A.getQuick(r, c), B.getQuick(idx++), TOL);
            }
        }
    }

    public void testViewColumn() {
        FComplexMatrix1D B = A.viewColumn(A.columns() / 2);
        assertEquals(A.rows(), B.size());
        for (int r = 0; r < A.rows(); r++) {
            assertEquals(A.getQuick(r, A.columns() / 2), B.getQuick(r), TOL);
        }
    }

    public void testViewColumnFlip() {
        FComplexMatrix2D B = A.viewColumnFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(r, A.columns() - 1 - c), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewDice() {
        FComplexMatrix2D B = A.viewDice();
        assertEquals(A.rows(), B.columns());
        assertEquals(A.columns(), B.rows());
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(r, c), B.getQuick(c, r), TOL);
            }
        }
    }

    public void testViewPart() {
        FComplexMatrix2D B = A.viewPart(A.rows() / 2, A.columns() / 2, A.rows() / 3, A.columns() / 3);
        for (int r = 0; r < A.rows() / 3; r++) {
            for (int c = 0; c < A.columns() / 3; c++) {
                assertEquals(A.getQuick(A.rows() / 2 + r, A.columns() / 2 + c), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewRow() {
        FComplexMatrix1D B = A.viewRow(A.rows() / 2);
        assertEquals(A.columns(), B.size());
        for (int c = 0; c < A.columns(); c++) {
            assertEquals(A.getQuick(A.rows() / 2, c), B.getQuick(c), TOL);
        }
    }

    public void testViewRowFlip() {
        FComplexMatrix2D B = A.viewRowFlip();
        assertEquals(A.size(), B.size());
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(A.getQuick(A.rows() - 1 - r, c), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewSelectionComplexMatrix1DProcedure() {
        final float[] value = new float[] { (float) Math.random(), (float) Math.random() };
        A.setQuick(A.rows() / 3, 0, value);
        A.setQuick(A.rows() / 2, 0, value);
        FComplexMatrix2D B = A.viewSelection(new FComplexMatrix1DProcedure() {
            public boolean apply(FComplexMatrix1D element) {
                return FComplex.isEqual(element.getQuick(0), value, TOL);
            }
        });
        assertEquals(2, B.rows());
        assertEquals(A.columns(), B.columns());
        assertEquals(A.getQuick(A.rows() / 3, 0), B.getQuick(0, 0), TOL);
        assertEquals(A.getQuick(A.rows() / 2, 0), B.getQuick(1, 0), TOL);
    }

    public void testViewSelectionIntArrayIntArray() {
        int[] rowIndexes = new int[] { A.rows() / 6, A.rows() / 5, A.rows() / 4, A.rows() / 3, A.rows() / 2 };
        int[] colIndexes = new int[] { A.columns() / 6, A.columns() / 5, A.columns() / 4, A.columns() / 3,
                A.columns() / 2, A.columns() - 1 };
        FComplexMatrix2D B = A.viewSelection(rowIndexes, colIndexes);
        assertEquals(rowIndexes.length, B.rows());
        assertEquals(colIndexes.length, B.columns());
        for (int r = 0; r < rowIndexes.length; r++) {
            for (int c = 0; c < colIndexes.length; c++) {
                assertEquals(A.getQuick(rowIndexes[r], colIndexes[c]), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testViewStrides() {
        int rowStride = 3;
        int colStride = 5;
        FComplexMatrix2D B = A.viewStrides(rowStride, colStride);
        for (int r = 0; r < B.rows(); r++) {
            for (int c = 0; c < B.columns(); c++) {
                assertEquals(A.getQuick(r * rowStride, c * colStride), B.getQuick(r, c), TOL);
            }
        }
    }

    public void testZMultFComplexMatrix1DFComplexMatrix1DFComplexFComplexBoolean() {
        FComplexMatrix1D y = new DenseFComplexMatrix1D(A.columns());
        for (int i = 0; i < y.size(); i++) {
            y.setQuick(i, new float[] { (float) Math.random(), (float) Math.random() });
        }
        float[] alpha = new float[] { 3, 2 };
        float[] beta = new float[] { 5, 4 };
        FComplexMatrix1D z = null;
        z = A.zMult(y, z, alpha, beta, false);
        float[] expected = new float[2 * A.rows()];
        float[] tmp = new float[2];
        for (int r = 0; r < A.rows(); r++) {
            float[] s = new float[2];
            for (int c = 0; c < A.columns(); c++) {
                s = FComplex.plus(s, FComplex.mult(A.getQuick(r, c), y.getQuick(c)));
            }
            tmp[0] = expected[2 * r];
            tmp[1] = expected[2 * r + 1];
            tmp = FComplex.mult(beta, tmp);
            tmp = FComplex.plus(tmp, FComplex.mult(alpha, s));
            expected[2 * r] = tmp[0];
            expected[2 * r + 1] = tmp[1];
        }

        for (int r = 0; r < A.rows(); r++) {
            assertEquals(expected[2 * r], z.getQuick(r)[0], TOL);
            assertEquals(expected[2 * r + 1], z.getQuick(r)[1], TOL);
        }
        //transpose
        y = new DenseFComplexMatrix1D(A.rows());
        for (int i = 0; i < y.size(); i++) {
            y.setQuick(i, new float[] { (float) Math.random(), (float) Math.random() });
        }
        z = null;
        z = A.zMult(y, z, alpha, beta, true);
        expected = new float[2 * A.columns()];
        for (int r = 0; r < A.columns(); r++) {
            float[] s = new float[2];
            for (int c = 0; c < A.rows(); c++) {
                s = FComplex.plus(s, FComplex.mult(FComplex.conj(A.getQuick(c, r)), y.getQuick(c)));
            }
            tmp[0] = expected[2 * r];
            tmp[1] = expected[2 * r + 1];
            tmp = FComplex.mult(beta, tmp);
            tmp = FComplex.plus(tmp, FComplex.mult(alpha, s));
            expected[2 * r] = tmp[0];
            expected[2 * r + 1] = tmp[1];
        }
        for (int r = 0; r < A.columns(); r++) {
            assertEquals(expected[2 * r], z.getQuick(r)[0], TOL);
            assertEquals(expected[2 * r + 1], z.getQuick(r)[1], TOL);
        }
    }

    public void testZMultFloatMatrix2DFloatMatrix2DFloatFloatBooleanBoolean() {
        float[] alpha = new float[] { 3, 2 };
        float[] beta = new float[] { 5, 4 };
        float[] tmp = new float[2];
        FComplexMatrix2D C = null;
        C = A.zMult(Bt, C, alpha, beta, false, false);
        float[][] expected = new float[A.rows()][2 * A.rows()];
        for (int j = 0; j < A.rows(); j++) {
            for (int i = 0; i < A.rows(); i++) {
                float[] s = new float[2];
                for (int k = 0; k < A.columns(); k++) {
                    s = FComplex.plus(s, FComplex.mult(A.getQuick(i, k), Bt.getQuick(k, j)));
                }
                tmp[0] = expected[i][2 * j];
                tmp[1] = expected[i][2 * j + 1];
                tmp = FComplex.mult(tmp, beta);
                tmp = FComplex.plus(tmp, FComplex.mult(s, alpha));
                expected[i][2 * j] = tmp[0];
                expected[i][2 * j + 1] = tmp[1];
            }
        }
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.rows(); c++) {
                assertEquals(expected[r][2 * c], C.getQuick(r, c)[0], TOL);
                assertEquals(expected[r][2 * c + 1], C.getQuick(r, c)[1], TOL);
            }
        }

        //transposeA
        C = null;
        C = A.zMult(B, C, alpha, beta, true, false);
        expected = new float[A.columns()][2 * A.columns()];
        for (int j = 0; j < A.columns(); j++) {
            for (int i = 0; i < A.columns(); i++) {
                float[] s = new float[2];
                for (int k = 0; k < A.rows(); k++) {
                    s = FComplex.plus(s, FComplex.mult(FComplex.conj(A.getQuick(k, i)), B.getQuick(k, j)));
                }
                tmp[0] = expected[i][2 * j];
                tmp[1] = expected[i][2 * j + 1];
                tmp = FComplex.mult(tmp, beta);
                tmp = FComplex.plus(tmp, FComplex.mult(s, alpha));
                expected[i][2 * j] = tmp[0];
                expected[i][2 * j + 1] = tmp[1];
            }
        }
        for (int r = 0; r < A.columns(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(expected[r][2 * c], C.getQuick(r, c)[0], TOL);
                assertEquals(expected[r][2 * c + 1], C.getQuick(r, c)[1], TOL);
            }
        }
        //transposeB
        C = null;
        C = A.zMult(B, C, alpha, beta, false, true);
        expected = new float[A.rows()][2 * A.rows()];
        for (int j = 0; j < A.rows(); j++) {
            for (int i = 0; i < A.rows(); i++) {
                float[] s = new float[2];
                for (int k = 0; k < A.columns(); k++) {
                    s = FComplex.plus(s, FComplex.mult(A.getQuick(i, k), FComplex.conj(B.getQuick(j, k))));
                }
                tmp[0] = expected[i][2 * j];
                tmp[1] = expected[i][2 * j + 1];
                tmp = FComplex.mult(tmp, beta);
                tmp = FComplex.plus(tmp, FComplex.mult(s, alpha));
                expected[i][2 * j] = tmp[0];
                expected[i][2 * j + 1] = tmp[1];
            }
        }
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.rows(); c++) {
                assertEquals(expected[r][2 * c], C.getQuick(r, c)[0], TOL);
                assertEquals(expected[r][2 * c + 1], C.getQuick(r, c)[1], TOL);
            }
        }
        //transposeA and transposeB
        C = null;
        C = A.zMult(Bt, C, alpha, beta, true, true);
        expected = new float[A.columns()][2 * A.columns()];
        for (int j = 0; j < A.columns(); j++) {
            for (int i = 0; i < A.columns(); i++) {
                float[] s = new float[2];
                for (int k = 0; k < A.rows(); k++) {
                    s = FComplex.plus(s, FComplex.mult(FComplex.conj(A.getQuick(k, i)), FComplex
                            .conj(Bt.getQuick(j, k))));
                }
                tmp[0] = expected[i][2 * j];
                tmp[1] = expected[i][2 * j + 1];
                tmp = FComplex.mult(tmp, beta);
                tmp = FComplex.plus(tmp, FComplex.mult(s, alpha));
                expected[i][2 * j] = tmp[0];
                expected[i][2 * j + 1] = tmp[1];
            }
        }
        for (int r = 0; r < A.columns(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(expected[r][2 * c], C.getQuick(r, c)[0], TOL);
                assertEquals(expected[r][2 * c + 1], C.getQuick(r, c)[1], TOL);
            }
        }

    }

    public void testZSum() {
        float[] actual = A.zSum();
        float[] expected = new float[2];
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                expected = FComplex.plus(expected, A.getQuick(r, c));
            }
        }
        assertEquals(expected, actual, TOL);
    }

    protected void assertEquals(float[] expected, float[] actual, float tol) {
        for (int i = 0; i < actual.length; i++) {
            assertEquals(expected[i], actual[i], tol);
        }
    }

}
