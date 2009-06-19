package cern.colt.matrix.tdouble.impl;

public class DiagonalDoubleMatrix2DViewTest extends DiagonalDoubleMatrix2DTest {

    public DiagonalDoubleMatrix2DViewTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        DINDEX = 3;
        A = new DiagonalDoubleMatrix2D(NCOLUMNS, NROWS, -DINDEX);
        DLENGTH = ((DiagonalDoubleMatrix2D) A).diagonalLength();
        A = A.viewDice();
        B = new DiagonalDoubleMatrix2D(NCOLUMNS, NROWS, -DINDEX).viewDice();
        Bt = new DiagonalDoubleMatrix2D(NROWS, NCOLUMNS, DINDEX).viewDice();
    }

}
