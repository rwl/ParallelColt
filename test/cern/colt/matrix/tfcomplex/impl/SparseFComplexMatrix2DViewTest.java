package cern.colt.matrix.tfcomplex.impl;

public class SparseFComplexMatrix2DViewTest extends SparseFComplexMatrix2DTest {
    public SparseFComplexMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseFComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseFComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseFComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
