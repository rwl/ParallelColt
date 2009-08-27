package cern.colt.matrix.tfloat.impl;

public class DenseLargeFloatMatrix2DViewTest extends DenseLargeFloatMatrix2DTest {

    public DenseLargeFloatMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseLargeFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new DenseLargeFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new DenseLargeFloatMatrix2D(NROWS, NCOLUMNS).viewDice();
    }
}
