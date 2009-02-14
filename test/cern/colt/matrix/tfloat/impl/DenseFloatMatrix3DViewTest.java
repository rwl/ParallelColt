package cern.colt.matrix.tfloat.impl;

public class DenseFloatMatrix3DViewTest extends DenseFloatMatrix3DTest {
    
    public DenseFloatMatrix3DViewTest(String arg0) {
        super(arg0);
    }
    
    @Override
    protected void createMatrices() throws Exception {
        A = new DenseFloatMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
        B = new DenseFloatMatrix3D(NCOLUMNS, NROWS, NSLICES).viewDice(2, 1, 0);
    }

}
