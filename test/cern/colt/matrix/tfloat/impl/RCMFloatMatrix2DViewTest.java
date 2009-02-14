package cern.colt.matrix.tfloat.impl;

public class RCMFloatMatrix2DViewTest extends RCMFloatMatrix2DTest {

    public RCMFloatMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new RCMFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new RCMFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new RCMFloatMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
