package cern.colt.matrix.tfcomplex.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfcomplex.FComplexMatrix2DTest;

public class LargeDenseFComplexMatrix2DTest extends FComplexMatrix2DTest {
    public LargeDenseFComplexMatrix2DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLargeFComplexMatrix2D(NROWS, NCOLUMNS);
        B = new DenseLargeFComplexMatrix2D(NROWS, NCOLUMNS);
        Bt = new DenseLargeFComplexMatrix2D(NCOLUMNS, NROWS);
    }

    public void testFft2() {
        FComplexMatrix2D Acopy = A.copy();
        ((WrapperFComplexMatrix2D) A).fft2();
        ((WrapperFComplexMatrix2D) A).ifft2(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testFftColumns() {
        FComplexMatrix2D Acopy = A.copy();
        ((WrapperFComplexMatrix2D) A).fftColumns();
        ((WrapperFComplexMatrix2D) A).ifftColumns(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testFftRows() {
        FComplexMatrix2D Acopy = A.copy();
        ((WrapperFComplexMatrix2D) A).fftRows();
        ((WrapperFComplexMatrix2D) A).ifftRows(true);
        for (int r = 0; r < A.rows(); r++) {
            for (int c = 0; c < A.columns(); c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }
}
