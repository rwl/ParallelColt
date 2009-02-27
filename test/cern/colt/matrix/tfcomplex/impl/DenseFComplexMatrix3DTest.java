package cern.colt.matrix.tfcomplex.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix3D;
import cern.colt.matrix.tfcomplex.FComplexMatrix3DTest;

public class DenseFComplexMatrix3DTest extends FComplexMatrix3DTest {
    public DenseFComplexMatrix3DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new DenseFComplexMatrix3D(NSLICES, NROWS, NCOLUMNS);
        B = new DenseFComplexMatrix3D(NSLICES, NROWS, NCOLUMNS);
    }

    public void testFft3() {
        FComplexMatrix3D Acopy = A.copy();
        ((DenseFComplexMatrix3D) A).fft3();
        ((DenseFComplexMatrix3D) A).ifft3(true);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testFft2Slices() {
        FComplexMatrix3D Acopy = A.copy();
        ((DenseFComplexMatrix3D) A).fft2Slices();
        ((DenseFComplexMatrix3D) A).ifft2Slices(true);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                }
            }
        }
    }
}
