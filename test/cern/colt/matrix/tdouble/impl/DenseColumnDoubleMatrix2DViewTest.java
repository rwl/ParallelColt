package cern.colt.matrix.tdouble.impl;

public class DenseColumnDoubleMatrix2DViewTest extends DenseColumnDoubleMatrix2DTest {

    public DenseColumnDoubleMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new DenseColumnDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new DenseColumnDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new DenseColumnDoubleMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
