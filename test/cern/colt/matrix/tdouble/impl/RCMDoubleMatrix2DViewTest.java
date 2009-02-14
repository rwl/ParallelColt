package cern.colt.matrix.tdouble.impl;

public class RCMDoubleMatrix2DViewTest extends RCMDoubleMatrix2DTest {

    public RCMDoubleMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new RCMDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new RCMDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new RCMDoubleMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
