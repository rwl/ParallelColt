package cern.colt.matrix.tfloat.impl;

public class DenseLargeFloatMatrix3DViewTest extends DenseLargeFloatMatrix3DTest {

    public DenseLargeFloatMatrix3DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLargeFloatMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
        B = new DenseLargeFloatMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
    }
}
