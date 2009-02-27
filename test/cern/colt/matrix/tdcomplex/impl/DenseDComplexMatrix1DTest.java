package cern.colt.matrix.tdcomplex.impl;

import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.DComplexMatrix1DTest;

public class DenseDComplexMatrix1DTest extends DComplexMatrix1DTest {

    public DenseDComplexMatrix1DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new DenseDComplexMatrix1D(SIZE);
        B = new DenseDComplexMatrix1D(SIZE);
    }

    public void testFft() {
        DComplexMatrix1D Acopy = A.copy();
        ((DenseDComplexMatrix1D) A).fft();
        ((DenseDComplexMatrix1D) A).ifft(true);
        for (int i = 0; i < SIZE; i++) {
            assertEquals(Acopy.getQuick(i), A.getQuick(i), TOL);
        }
    }

}
