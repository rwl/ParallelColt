package cern.colt.matrix.tfcomplex.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfcomplex.FComplexMatrix2DTest;

public class DenseFComplexMatrix2DTest extends FComplexMatrix2DTest {
    public DenseFComplexMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new DenseFComplexMatrix2D(NROWS, NCOLUMNS);
        B = new DenseFComplexMatrix2D(NROWS, NCOLUMNS);
        Bt = new DenseFComplexMatrix2D(NCOLUMNS, NROWS);
    }
    
    public void testFft2() {
        FComplexMatrix2D Acopy = A.copy();
        ((DenseFComplexMatrix2D) A).fft2();
        ((DenseFComplexMatrix2D) A).ifft2(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testFftColumns() {
        FComplexMatrix2D Acopy = A.copy();
        ((DenseFComplexMatrix2D) A).fftColumns();
        ((DenseFComplexMatrix2D) A).ifftColumns(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }
    
    public void testFftRows() {
        FComplexMatrix2D Acopy = A.copy();
        ((DenseFComplexMatrix2D) A).fftRows();
        ((DenseFComplexMatrix2D) A).ifftRows(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }
}
