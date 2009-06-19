package cern.colt.matrix.tdouble.impl;

public class SparseRCMDoubleMatrix2DViewTest extends SparseRCMDoubleMatrix2DTest {

    public SparseRCMDoubleMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseRCMDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseRCMDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseRCMDoubleMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
