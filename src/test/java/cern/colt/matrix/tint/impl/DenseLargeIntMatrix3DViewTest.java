package cern.colt.matrix.tint.impl;

public class DenseLargeIntMatrix3DViewTest extends DenseLargeIntMatrix3DTest {

    public DenseLargeIntMatrix3DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLargeIntMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
        B = new DenseLargeIntMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
    }
}
