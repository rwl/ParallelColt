package cern.colt.matrix.tfcomplex.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix3D;
import cern.colt.matrix.tfcomplex.FComplexMatrix3DTest;

public class LargeDenseFComplexMatrix3DTest extends FComplexMatrix3DTest {
    public LargeDenseFComplexMatrix3DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLargeFComplexMatrix3D(NSLICES, NROWS, NCOLUMNS);
        B = new DenseLargeFComplexMatrix3D(NSLICES, NROWS, NCOLUMNS);
    }

    public void testFft3() {
        FComplexMatrix3D Acopy = A.copy();
        ((WrapperFComplexMatrix3D) A).fft3();
        ((WrapperFComplexMatrix3D) A).ifft3(true);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testFft2Slices() {
        FComplexMatrix3D Acopy = A.copy();
        ((WrapperFComplexMatrix3D) A).fft2Slices();
        ((WrapperFComplexMatrix3D) A).ifft2Slices(true);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                }
            }
        }
    }
}
