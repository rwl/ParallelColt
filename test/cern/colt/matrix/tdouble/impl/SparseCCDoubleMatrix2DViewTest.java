package cern.colt.matrix.tdouble.impl;

public class SparseCCDoubleMatrix2DViewTest extends SparseCCDoubleMatrix2DTest {

    public SparseCCDoubleMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new SparseCCDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseCCDoubleMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseCCDoubleMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
