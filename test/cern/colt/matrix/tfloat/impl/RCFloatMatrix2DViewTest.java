package cern.colt.matrix.tfloat.impl;


public class RCFloatMatrix2DViewTest extends RCFloatMatrix2DTest {

    public RCFloatMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new RCFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new RCFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new RCFloatMatrix2D(NROWS, NCOLUMNS).viewDice();        
    }

}
