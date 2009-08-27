package cern.colt.matrix.tfcomplex.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfcomplex.FComplexMatrix2DTest;

public class DenseColumnFComplexMatrix2DTest extends FComplexMatrix2DTest {
    public DenseColumnFComplexMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseColumnFComplexMatrix2D(NROWS, NCOLUMNS);
        B = new DenseColumnFComplexMatrix2D(NROWS, NCOLUMNS);
        Bt = new DenseColumnFComplexMatrix2D(NCOLUMNS, NROWS);
    }

    public void testAssignFloatArray() {
        float[] expected = new float[2 * (int) A.size()];
        for (int i = 0; i < 2 * A.size(); i++) {
            expected[i] = (float) Math.random();
        }
        A.assign(expected);
        int idx = 0;
        for (int c = 0; c < A.columns(); c++) {
            for (int r = 0; r < A.rows(); r++) {
                float[] elem = A.getQuick(r, c);
                assertEquals(expected[idx], elem[0], TOL);
                assertEquals(expected[idx + 1], elem[1], TOL);
                idx += 2;
            }
        }
    }

    public void testAssignFloatArrayArray() {
        float[][] expected = new float[A.columns()][2 * A.rows()];
        for (int c = 0; c < A.columns(); c++) {
            for (int r = 0; r < 2 * A.rows(); r++) {
                expected[c][r] = (float) Math.random();
            }
        }
        A.assign(expected);
        for (int c = 0; c < A.columns(); c++) {
            for (int r = 0; r < A.rows(); r++) {
                float[] elem = A.getQuick(r, c);
                assertEquals(expected[c][2 * r], elem[0], TOL);
                assertEquals(expected[c][2 * r + 1], elem[1], TOL);
            }
        }
    }

    public void testFft2() {
        FComplexMatrix2D Acopy = A.copy();
        ((DenseColumnFComplexMatrix2D) A).fft2();
        ((DenseColumnFComplexMatrix2D) A).ifft2(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testFftColumns() {
        FComplexMatrix2D Acopy = A.copy();
        ((DenseColumnFComplexMatrix2D) A).fftColumns();
        ((DenseColumnFComplexMatrix2D) A).ifftColumns(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testFftRows() {
        FComplexMatrix2D Acopy = A.copy();
        ((DenseColumnFComplexMatrix2D) A).fftRows();
        ((DenseColumnFComplexMatrix2D) A).ifftRows(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }
}
