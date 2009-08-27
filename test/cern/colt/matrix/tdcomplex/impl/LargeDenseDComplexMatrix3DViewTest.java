package cern.colt.matrix.tdcomplex.impl;

public class LargeDenseDComplexMatrix3DViewTest extends LargeDenseDComplexMatrix3DTest {
    public LargeDenseDComplexMatrix3DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLargeDComplexMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
        B = new DenseLargeDComplexMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
    }

}
