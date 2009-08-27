package cern.colt.matrix.tlong.impl;

import cern.colt.matrix.tlong.LongMatrix2DTest;

public class SparseRCLongMatrix2DTest extends LongMatrix2DTest {

    public SparseRCLongMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseRCLongMatrix2D(NROWS, NCOLUMNS);
        B = new SparseRCLongMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseRCLongMatrix2D(NCOLUMNS, NROWS);
    }

    //    public void testZMultLongMatrix2DLongMatrix2DLongLongBooleanBoolean() {
    //        long alpha = 2;
    //        long beta = 5;
    //        LongMatrix2D C = new SparseRCLongMatrix2D(NROWS, NROWS, NROWS * NROWS);
    //        long[][] expected = C.toArray();
    //        C = A.zMult(Bt, C, alpha, beta, false, false);
    //        for (int j = 0; j < A.rows(); j++) {
    //            for (int i = 0; i < A.rows(); i++) {
    //                long s = 0;
    //                for (int k = 0; k < A.columns(); k++) {
    //                    s += A.getQuick(i, k) * Bt.getQuick(k, j);
    //                }
    //                expected[i][j] = s * alpha + expected[i][j] * beta;
    //            }
    //        }
    //        for (int r = 0; r < A.rows(); r++) {
    //            for (int c = 0; c < A.rows(); c++) {
    //                assertEquals(expected[r][c], C.getQuick(r, c));
    //            }
    //        }
    //
    //        //---
    //        C = null;
    //        C = A.zMult(Bt, C, alpha, beta, false, false);
    //        expected = new long[A.rows()][A.rows()];
    //        for (int j = 0; j < A.rows(); j++) {
    //            for (int i = 0; i < A.rows(); i++) {
    //                long s = 0;
    //                for (int k = 0; k < A.columns(); k++) {
    //                    s += A.getQuick(i, k) * Bt.getQuick(k, j);
    //                }
    //                expected[i][j] = s * alpha;
    //            }
    //        }
    //        for (int r = 0; r < A.rows(); r++) {
    //            for (int c = 0; c < A.rows(); c++) {
    //                assertEquals(expected[r][c], C.getQuick(r, c));
    //            }
    //        }
    //
    //        //transposeA
    //        C = new SparseRCLongMatrix2D(A.columns(), A.columns(), A.columns() * A.columns());
    //        expected = C.toArray();
    //        C = A.zMult(B, C, alpha, beta, true, false);
    //        for (int j = 0; j < A.columns(); j++) {
    //            for (int i = 0; i < A.columns(); i++) {
    //                long s = 0;
    //                for (int k = 0; k < A.rows(); k++) {
    //                    s += A.getQuick(k, i) * B.getQuick(k, j);
    //                }
    //                expected[i][j] = s * alpha + expected[i][j] * beta;
    //            }
    //        }
    //        for (int r = 0; r < A.columns(); r++) {
    //            for (int c = 0; c < A.columns(); c++) {
    //                assertEquals(expected[r][c], C.getQuick(r, c));
    //            }
    //        }
    //        //---
    //        C = null;
    //        C = A.zMult(B, C, alpha, beta, true, false);
    //        expected = new long[A.columns()][A.columns()];
    //        for (int j = 0; j < A.columns(); j++) {
    //            for (int i = 0; i < A.columns(); i++) {
    //                long s = 0;
    //                for (int k = 0; k < A.rows(); k++) {
    //                    s += A.getQuick(k, i) * B.getQuick(k, j);
    //                }
    //                expected[i][j] = s * alpha;
    //            }
    //        }
    //        for (int r = 0; r < A.columns(); r++) {
    //            for (int c = 0; c < A.columns(); c++) {
    //                assertEquals(expected[r][c], C.getQuick(r, c));
    //            }
    //        }
    //
    //        //transposeB
    //        C = new SparseRCLongMatrix2D(A.rows(), A.rows(), A.rows() * A.rows());
    //        expected = C.toArray();
    //        C = A.zMult(B, C, alpha, beta, false, true);
    //        for (int j = 0; j < A.rows(); j++) {
    //            for (int i = 0; i < A.rows(); i++) {
    //                long s = 0;
    //                for (int k = 0; k < A.columns(); k++) {
    //                    s += A.getQuick(i, k) * B.getQuick(j, k);
    //                }
    //                expected[i][j] = s * alpha + expected[i][j] * beta;
    //            }
    //        }
    //        for (int r = 0; r < A.rows(); r++) {
    //            for (int c = 0; c < A.rows(); c++) {
    //                assertEquals(expected[r][c], C.getQuick(r, c));
    //            }
    //        }
    //        //---
    //        C = null;
    //        C = A.zMult(B, C, alpha, beta, false, true);
    //        expected = new long[A.rows()][A.rows()];
    //        for (int j = 0; j < A.rows(); j++) {
    //            for (int i = 0; i < A.rows(); i++) {
    //                long s = 0;
    //                for (int k = 0; k < A.columns(); k++) {
    //                    s += A.getQuick(i, k) * B.getQuick(j, k);
    //                }
    //                expected[i][j] = s * alpha;
    //            }
    //        }
    //        for (int r = 0; r < A.rows(); r++) {
    //            for (int c = 0; c < A.rows(); c++) {
    //                assertEquals(expected[r][c], C.getQuick(r, c));
    //            }
    //        }
    //        //transposeA and transposeB
    //        C = new SparseRCLongMatrix2D(A.columns(), A.columns(), A.columns() * A.columns());
    //        expected = C.toArray();
    //        C = A.zMult(Bt, C, alpha, beta, true, true);
    //        for (int j = 0; j < A.columns(); j++) {
    //            for (int i = 0; i < A.columns(); i++) {
    //                long s = 0;
    //                for (int k = 0; k < A.rows(); k++) {
    //                    s += A.getQuick(k, i) * Bt.getQuick(j, k);
    //                }
    //                expected[i][j] = s * alpha + expected[i][j] * beta;
    //            }
    //        }
    //        for (int r = 0; r < A.columns(); r++) {
    //            for (int c = 0; c < A.columns(); c++) {
    //                assertEquals(expected[r][c], C.getQuick(r, c));
    //            }
    //        }
    //        //---
    //        C = null;
    //        C = A.zMult(Bt, C, alpha, beta, true, true);
    //        expected = new long[A.columns()][A.columns()];
    //        for (int j = 0; j < A.columns(); j++) {
    //            for (int i = 0; i < A.columns(); i++) {
    //                long s = 0;
    //                for (int k = 0; k < A.rows(); k++) {
    //                    s += A.getQuick(k, i) * Bt.getQuick(j, k);
    //                }
    //                expected[i][j] = s * alpha;
    //            }
    //        }
    //        for (int r = 0; r < A.columns(); r++) {
    //            for (int c = 0; c < A.columns(); c++) {
    //                assertEquals(expected[r][c], C.getQuick(r, c));
    //            }
    //        }
    //
    //    }

}
