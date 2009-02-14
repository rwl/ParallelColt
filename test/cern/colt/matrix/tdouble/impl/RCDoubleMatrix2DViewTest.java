package cern.colt.matrix.tdouble.impl;


public class RCDoubleMatrix2DViewTest extends RCDoubleMatrix2DTest {

    public RCDoubleMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new RCDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new RCDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new RCDoubleMatrix2D(NROWS, NCOLUMNS).viewDice();        
    }

}
