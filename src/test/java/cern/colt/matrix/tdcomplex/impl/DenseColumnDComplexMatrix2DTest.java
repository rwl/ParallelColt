package cern.colt.matrix.tdcomplex.impl;

import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2DTest;

public class DenseColumnDComplexMatrix2DTest extends DComplexMatrix2DTest {
    public DenseColumnDComplexMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseColumnDComplexMatrix2D(NROWS, NCOLUMNS);
        B = new DenseColumnDComplexMatrix2D(NROWS, NCOLUMNS);
        Bt = new DenseColumnDComplexMatrix2D(NCOLUMNS, NROWS);
    }

    public void testAssignDoubleArray() {
        double[] expected = new double[2 * (int) A.size()];
        for (int i = 0; i < 2 * A.size(); i++) {
            expected[i] = Math.random();
        }
        A.assign(expected);
        int idx = 0;
        for (int c = 0; c < A.columns(); c++) {
            for (int r = 0; r < A.rows(); r++) {
                double[] elem = A.getQuick(r, c);
                assertEquals(expected[idx], elem[0], TOL);
                assertEquals(expected[idx + 1], elem[1], TOL);
                idx += 2;
            }
        }
    }

    public void testAssignFloatArray() {
        float[] expected = new float[NROWS * 2 * NCOLUMNS];
        for (int i = 0; i < 2 * A.size(); i++) {
            expected[i] = (float) Math.random();
        }
        A.assign(expected);
        int idx = 0;
        for (int c = 0; c < A.columns(); c++) {
            for (int r = 0; r < A.rows(); r++) {
                double[] elem = A.getQuick(r, c);
                assertEquals(expected[idx], elem[0], TOL);
                assertEquals(expected[idx + 1], elem[1], TOL);
                idx += 2;
            }
        }
    }

    public void testAssignDoubleArrayArray() {
        double[][] expected = new double[NCOLUMNS][2 * NROWS];
        for (int c = 0; c < NCOLUMNS; c++) {
            for (int r = 0; r < 2 * NROWS; r++) {
                expected[c][r] = Math.random();
            }
        }
        A.assign(expected);
        for (int c = 0; c < A.columns(); c++) {
            for (int r = 0; r < A.rows(); r++) {
                double[] elem = A.getQuick(r, c);
                assertEquals(expected[c][2 * r], elem[0], TOL);
                assertEquals(expected[c][2 * r + 1], elem[1], TOL);
            }
        }
    }

    public void testFft2() {
        DComplexMatrix2D Acopy = A.copy();
        ((DenseColumnDComplexMatrix2D) A).fft2();
        ((DenseColumnDComplexMatrix2D) A).ifft2(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testFftColumns() {
        DComplexMatrix2D Acopy = A.copy();
        ((DenseColumnDComplexMatrix2D) A).fftColumns();
        ((DenseColumnDComplexMatrix2D) A).ifftColumns(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testFftRows() {
        DComplexMatrix2D Acopy = A.copy();
        ((DenseColumnDComplexMatrix2D) A).fftRows();
        ((DenseColumnDComplexMatrix2D) A).ifftRows(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }
}
