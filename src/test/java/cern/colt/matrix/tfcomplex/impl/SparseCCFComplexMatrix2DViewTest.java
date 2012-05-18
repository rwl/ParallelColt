package cern.colt.matrix.tfcomplex.impl;

public class SparseCCFComplexMatrix2DViewTest extends SparseCCFComplexMatrix2DTest {

    public SparseCCFComplexMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new SparseCCFComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        B = new SparseCCFComplexMatrix2D(NCOLUMNS, NROWS).viewDice();
        Bt = new SparseCCFComplexMatrix2D(NROWS, NCOLUMNS).viewDice();
    }

}
