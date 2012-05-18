package cern.colt.matrix.tdcomplex.impl;

import cern.colt.matrix.tdcomplex.DComplexMatrix3D;
import cern.colt.matrix.tdcomplex.DComplexMatrix3DTest;

public class LargeDenseDComplexMatrix3DTest extends DComplexMatrix3DTest {
    public LargeDenseDComplexMatrix3DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLargeDComplexMatrix3D(NSLICES, NROWS, NCOLUMNS);
        B = new DenseLargeDComplexMatrix3D(NSLICES, NROWS, NCOLUMNS);
    }

    public void testFft3() {
        DComplexMatrix3D Acopy = A.copy();
        ((WrapperDComplexMatrix3D) A).fft3();
        ((WrapperDComplexMatrix3D) A).ifft3(true);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testFft2Slices() {
        DComplexMatrix3D Acopy = A.copy();
        ((WrapperDComplexMatrix3D) A).fft2Slices();
        ((WrapperDComplexMatrix3D) A).ifft2Slices(true);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                }
            }
        }
    }
}
