package cern.colt.matrix.tdouble.impl;

import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix2DTest;

public class SparseRCDoubleMatrix2DTest extends DoubleMatrix2DTest {

    public SparseRCDoubleMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseRCDoubleMatrix2D(NROWS, NCOLUMNS);
        B = new SparseRCDoubleMatrix2D(NROWS, NCOLUMNS);
        Bt = new SparseRCDoubleMatrix2D(NCOLUMNS, NROWS);
    }

    @Override
    public void testZMultDoubleMatrix2DDoubleMatrix2DDoubleDoubleBooleanBoolean() {
        //        NROWS = 6;
        //        NCOLUMNS = 6;
        //        A = new SparseDoubleMatrix2D(NROWS, NCOLUMNS);
        //        A.setQuick(0, 0, 10);
        //        A.setQuick(0, 4, -2);
        //        A.setQuick(1, 0, 3);
        //        A.setQuick(1, 1, 9);
        //        A.setQuick(1, 5, 3);
        //        A.setQuick(2, 1, 7);
        //        A.setQuick(2, 2, 8);
        //        A.setQuick(2, 3, 7);
        //        A.setQuick(3, 0, 3);
        //        A.setQuick(3, 2, 8);
        //        A.setQuick(3, 3, 7);
        //        A.setQuick(3, 4, 5);
        //        A.setQuick(4, 1, 8);
        //        A.setQuick(4, 3, 9);
        //        A.setQuick(4, 4, 9);
        //        A.setQuick(4, 5, 13);
        //        A.setQuick(5, 1, 4);
        //        A.setQuick(5, 4, 2);
        //        A.setQuick(5, 5, -1);
        //        Bt = A.viewDice().copy();
        //        A = ((SparseDoubleMatrix2D)A).convertToRCDoubleMatrix2D();
        //        Bt = ((SparseDoubleMatrix2D)Bt).convertToRCDoubleMatrix2D();

        double alpha = 2;
        double beta = 5;
        DoubleMatrix2D C = new SparseRCDoubleMatrix2D(NROWS, NROWS, NROWS * NROWS);
        double[][] expected = C.toArray();
        C = A.zMult(Bt, C, alpha, beta, false, false);
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                double s = 0;
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
        expected = new double[NROWS][NROWS];
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                double s = 0;
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
        C = new SparseRCDoubleMatrix2D(NCOLUMNS, NCOLUMNS, NCOLUMNS * NCOLUMNS);
        expected = C.toArray();
        C = A.zMult(B, C, alpha, beta, true, false);
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                double s = 0;
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
        expected = new double[NCOLUMNS][NCOLUMNS];
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                double s = 0;
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
        C = new SparseRCDoubleMatrix2D(NROWS, NROWS, NROWS * NROWS);
        expected = C.toArray();
        C = A.zMult(B, C, alpha, beta, false, true);
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                double s = 0;
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
        expected = new double[NROWS][NROWS];
        for (int j = 0; j < NROWS; j++) {
            for (int i = 0; i < NROWS; i++) {
                double s = 0;
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
        C = new SparseRCDoubleMatrix2D(NCOLUMNS, NCOLUMNS, NCOLUMNS * NCOLUMNS);
        expected = C.toArray();
        C = A.zMult(Bt, C, alpha, beta, true, true);
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                double s = 0;
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
        expected = new double[NCOLUMNS][NCOLUMNS];
        for (int j = 0; j < NCOLUMNS; j++) {
            for (int i = 0; i < NCOLUMNS; i++) {
                double s = 0;
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

}
