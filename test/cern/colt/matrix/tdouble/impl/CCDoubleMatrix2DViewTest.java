package cern.colt.matrix.tdouble.impl;


public class CCDoubleMatrix2DViewTest extends CCDoubleMatrix2DTest {

    public CCDoubleMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new CCDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new CCDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new CCDoubleMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
