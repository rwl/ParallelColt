package cern.colt.matrix.tdouble.impl;

public class DenseDoubleMatrix3DViewTest extends DenseDoubleMatrix3DTest {
    
    public DenseDoubleMatrix3DViewTest(String arg0) {
        super(arg0);
    }
    
    @Override
    protected void createMatrices() throws Exception {
        A = new DenseDoubleMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
        B = new DenseDoubleMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
    }

}
