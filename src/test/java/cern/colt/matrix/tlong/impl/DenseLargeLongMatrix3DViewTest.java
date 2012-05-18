package cern.colt.matrix.tlong.impl;

public class DenseLargeLongMatrix3DViewTest extends DenseLargeLongMatrix3DTest {

    public DenseLargeLongMatrix3DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLargeLongMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
        B = new DenseLargeLongMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
    }
}
