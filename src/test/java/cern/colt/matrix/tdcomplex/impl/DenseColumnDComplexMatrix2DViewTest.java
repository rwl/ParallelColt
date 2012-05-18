package cern.colt.matrix.tdcomplex.impl;

public class DenseColumnDComplexMatrix2DViewTest extends DenseColumnDComplexMatrix2DTest {
    public DenseColumnDComplexMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseColumnDComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new DenseColumnDComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new DenseColumnDComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
