package cern.colt.matrix.tdouble.impl;

public class SparseDoubleMatrix2DViewTest extends SparseDoubleMatrix2DTest {

    public SparseDoubleMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseDoubleMatrix2D(NROWS, NCOLUMNS).viewDice();
    }
}