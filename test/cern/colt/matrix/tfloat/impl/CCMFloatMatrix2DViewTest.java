package cern.colt.matrix.tfloat.impl;


public class CCMFloatMatrix2DViewTest extends CCMFloatMatrix2DTest {

    public CCMFloatMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new CCMFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new CCMFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new CCMFloatMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
