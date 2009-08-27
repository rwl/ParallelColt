package cern.colt.matrix.tfcomplex.impl;

public class DenseColumnFComplexMatrix2DViewTest extends DenseColumnFComplexMatrix2DTest {
    public DenseColumnFComplexMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseColumnFComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new DenseColumnFComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new DenseColumnFComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
