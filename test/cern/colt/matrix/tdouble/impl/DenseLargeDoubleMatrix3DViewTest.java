package cern.colt.matrix.tdouble.impl;

public class DenseLargeDoubleMatrix3DViewTest extends DenseLargeDoubleMatrix3DTest {

    public DenseLargeDoubleMatrix3DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLargeDoubleMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
        B = new DenseLargeDoubleMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
    }
}
