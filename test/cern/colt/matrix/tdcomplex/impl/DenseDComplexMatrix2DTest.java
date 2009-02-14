package cern.colt.matrix.tdcomplex.impl;

import cern.colt.matrix.tdcomplex.DComplexMatrix2D;
import cern.colt.matrix.tdcomplex.DComplexMatrix2DTest;

public class DenseDComplexMatrix2DTest extends DComplexMatrix2DTest {
    public DenseDComplexMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new DenseDComplexMatrix2D(NROWS, NCOLUMNS);
        B = new DenseDComplexMatrix2D(NROWS, NCOLUMNS);
        Bt = new DenseDComplexMatrix2D(NCOLUMNS, NROWS);
    }
    
    public void testFft2() {
        DComplexMatrix2D Acopy = A.copy();
        ((DenseDComplexMatrix2D) A).fft2();
        ((DenseDComplexMatrix2D) A).ifft2(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testFftColumns() {
        DComplexMatrix2D Acopy = A.copy();
        ((DenseDComplexMatrix2D) A).fftColumns();
        ((DenseDComplexMatrix2D) A).ifftColumns(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }
    
    public void testFftRows() {
        DComplexMatrix2D Acopy = A.copy();
        ((DenseDComplexMatrix2D) A).fftRows();
        ((DenseDComplexMatrix2D) A).ifftRows(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }
}
