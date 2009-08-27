package cern.colt.matrix.tdcomplex.impl;

public class DenseDComplexMatrix3DViewTest extends DenseDComplexMatrix3DTest {
    public DenseDComplexMatrix3DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseDComplexMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
        B = new DenseDComplexMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
    }

}
