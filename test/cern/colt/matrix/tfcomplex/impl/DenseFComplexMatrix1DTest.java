package cern.colt.matrix.tfcomplex.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.FComplexMatrix1DTest;

public class DenseFComplexMatrix1DTest extends FComplexMatrix1DTest {

    public DenseFComplexMatrix1DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseFComplexMatrix1D(SIZE);
        B = new DenseFComplexMatrix1D(SIZE);
    }

    public void testFft() {
        FComplexMatrix1D Acopy = A.copy();
        ((DenseFComplexMatrix1D) A).fft();
        ((DenseFComplexMatrix1D) A).ifft(true);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(Acopy.getQuick(i), A.getQuick(i), TOL);
        }
    }

}
