package cern.colt.matrix.tdouble.impl;

public class SparseCCMDoubleMatrix2DViewTest extends SparseCCMDoubleMatrix2DTest {

    public SparseCCMDoubleMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseCCMDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseCCMDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseCCMDoubleMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
