package cern.colt.matrix.tfcomplex.impl;

public class LargeDenseFComplexMatrix3DViewTest extends LargeDenseFComplexMatrix3DTest {
    public LargeDenseFComplexMatrix3DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLargeFComplexMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
        B = new DenseLargeFComplexMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
    }

}
