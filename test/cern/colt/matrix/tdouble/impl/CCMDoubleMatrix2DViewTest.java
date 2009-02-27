package cern.colt.matrix.tdouble.impl;

public class CCMDoubleMatrix2DViewTest extends CCMDoubleMatrix2DTest {

    public CCMDoubleMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new CCMDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new CCMDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new CCMDoubleMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
