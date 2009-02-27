package cern.colt.matrix.tfloat.impl;

public class CCFloatMatrix2DViewTest extends CCFloatMatrix2DTest {

    public CCFloatMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new CCFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new CCFloatMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new CCFloatMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
