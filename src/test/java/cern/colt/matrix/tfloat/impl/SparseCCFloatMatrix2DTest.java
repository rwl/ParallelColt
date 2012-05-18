package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfloat.FloatMatrix2DTest;

public class SparseCCFloatMatrix2DTest extends FloatMatrix2DTest {

    public SparseCCFloatMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseCCFloatMatrix2D(NROWS, NCOLUMNS);
        B = new SparseCCFloatMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseCCFloatMatrix2D(NCOLUMNS, NROWS);
    }

    //    public void testZMultFloatMatrix2DFloatMatrix2DFloatFloatBooleanBoolean() {
    //        float alpha = 2;
    //        float beta = 5;
    //        FloatMatrix2D C = new SparseCCFloatMatrix2D(NROWS, NROWS, NROWS * NROWS);
    //        float[][] expected = C.toArray();
    //        C = A.zMult(Bt, C, alpha, beta, false, false);
    //        for (int j = 0; j < A.rows(); j++) {
    //            for (int i = 0; i < A.rows(); i++) {
    //                float s = 0;
    //                for (int k = 0; k < A.columns(); k++) {
    //                    s += A.getQuick(i, k) * Bt.getQuick(k, j);
    //                }
    //                expected[i][j] = s * alpha + expected[i][j] * beta;
    //            }
    //        }
    //        for (int r = 0; r < A.rows(); r++) {
    //            for (int c = 0; c < A.rows(); c++) {
    //                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
    //            }
    //        }
    //
    //        //---
    //        C = null;
    //        C = A.zMult(Bt, C, alpha, beta, false, false);
    //        expected = new float[A.rows()][A.rows()];
    //        for (int j = 0; j < A.rows(); j++) {
    //            for (int i = 0; i < A.rows(); i++) {
    //                float s = 0;
    //                for (int k = 0; k < A.columns(); k++) {
    //                    s += A.getQuick(i, k) * Bt.getQuick(k, j);
    //                }
    //                expected[i][j] = s * alpha;
    //            }
    //        }
    //        for (int r = 0; r < A.rows(); r++) {
    //            for (int c = 0; c < A.rows(); c++) {
    //                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
    //            }
    //        }
    //
    //        //transposeA
    //        C = new SparseCCFloatMatrix2D(A.columns(), A.columns(), A.columns() * A.columns());
    //        expected = C.toArray();
    //        C = A.zMult(B, C, alpha, beta, true, false);
    //        for (int j = 0; j < A.columns(); j++) {
    //            for (int i = 0; i < A.columns(); i++) {
    //                float s = 0;
    //                for (int k = 0; k < A.rows(); k++) {
    //                    s += A.getQuick(k, i) * B.getQuick(k, j);
    //                }
    //                expected[i][j] = s * alpha + expected[i][j] * beta;
    //            }
    //        }
    //        for (int r = 0; r < A.columns(); r++) {
    //            for (int c = 0; c < A.columns(); c++) {
    //                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
    //            }
    //        }
    //        //---
    //        C = null;
    //        C = A.zMult(B, C, alpha, beta, true, false);
    //        expected = new float[A.columns()][A.columns()];
    //        for (int j = 0; j < A.columns(); j++) {
    //            for (int i = 0; i < A.columns(); i++) {
    //                float s = 0;
    //                for (int k = 0; k < A.rows(); k++) {
    //                    s += A.getQuick(k, i) * B.getQuick(k, j);
    //                }
    //                expected[i][j] = s * alpha;
    //            }
    //        }
    //        for (int r = 0; r < A.columns(); r++) {
    //            for (int c = 0; c < A.columns(); c++) {
    //                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
    //            }
    //        }
    //
    //        //transposeB
    //        C = new SparseCCFloatMatrix2D(A.rows(), A.rows(), A.rows() * A.rows());
    //        expected = C.toArray();
    //        C = A.zMult(B, C, alpha, beta, false, true);
    //        for (int j = 0; j < A.rows(); j++) {
    //            for (int i = 0; i < A.rows(); i++) {
    //                float s = 0;
    //                for (int k = 0; k < A.columns(); k++) {
    //                    s += A.getQuick(i, k) * B.getQuick(j, k);
    //                }
    //                expected[i][j] = s * alpha + expected[i][j] * beta;
    //            }
    //        }
    //        for (int r = 0; r < A.rows(); r++) {
    //            for (int c = 0; c < A.rows(); c++) {
    //                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
    //            }
    //        }
    //        //---
    //        C = null;
    //        C = A.zMult(B, C, alpha, beta, false, true);
    //        expected = new float[A.rows()][A.rows()];
    //        for (int j = 0; j < A.rows(); j++) {
    //            for (int i = 0; i < A.rows(); i++) {
    //                float s = 0;
    //                for (int k = 0; k < A.columns(); k++) {
    //                    s += A.getQuick(i, k) * B.getQuick(j, k);
    //                }
    //                expected[i][j] = s * alpha;
    //            }
    //        }
    //        for (int r = 0; r < A.rows(); r++) {
    //            for (int c = 0; c < A.rows(); c++) {
    //                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
    //            }
    //        }
    //        //transposeA and transposeB
    //        C = new SparseCCFloatMatrix2D(A.columns(), A.columns(), A.columns() * A.columns());
    //        expected = C.toArray();
    //        C = A.zMult(Bt, C, alpha, beta, true, true);
    //        for (int j = 0; j < A.columns(); j++) {
    //            for (int i = 0; i < A.columns(); i++) {
    //                float s = 0;
    //                for (int k = 0; k < A.rows(); k++) {
    //                    s += A.getQuick(k, i) * Bt.getQuick(j, k);
    //                }
    //                expected[i][j] = s * alpha + expected[i][j] * beta;
    //            }
    //        }
    //        for (int r = 0; r < A.columns(); r++) {
    //            for (int c = 0; c < A.columns(); c++) {
    //                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
    //            }
    //        }
    //        //---
    //        C = null;
    //        C = A.zMult(Bt, C, alpha, beta, true, true);
    //        expected = new float[A.columns()][A.columns()];
    //        for (int j = 0; j < A.columns(); j++) {
    //            for (int i = 0; i < A.columns(); i++) {
    //                float s = 0;
    //                for (int k = 0; k < A.rows(); k++) {
    //                    s += A.getQuick(k, i) * Bt.getQuick(j, k);
    //                }
    //                expected[i][j] = s * alpha;
    //            }
    //        }
    //        for (int r = 0; r < A.columns(); r++) {
    //            for (int c = 0; c < A.columns(); c++) {
    //                assertEquals(expected[r][c], C.getQuick(r, c), TOL);
    //            }
    //        }
    //
    //    }

}
