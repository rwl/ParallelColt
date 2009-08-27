package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix1D;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix1DTest;

public class DenseFloatMatrix1DTest extends FloatMatrix1DTest {

    public DenseFloatMatrix1DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseFloatMatrix1D(SIZE);
        B = new DenseFloatMatrix1D(SIZE);
    }

    public void testDct() {
        FloatMatrix1D Acopy = A.copy();
        ((DenseFloatMatrix1D) A).dct(true);
        ((DenseFloatMatrix1D) A).idct(true);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(Acopy.getQuick(i), A.getQuick(i), TOL);
        }
    }

    public void testDst() {
        FloatMatrix1D Acopy = A.copy();
        ((DenseFloatMatrix1D) A).dst(true);
        ((DenseFloatMatrix1D) A).idst(true);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(Acopy.getQuick(i), A.getQuick(i), TOL);
        }
    }

    public void testDht() {
        FloatMatrix1D Acopy = A.copy();
        ((DenseFloatMatrix1D) A).dht();
        ((DenseFloatMatrix1D) A).idht(true);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(Acopy.getQuick(i), A.getQuick(i), TOL);
        }
    }

    public void testFft() {
        FloatMatrix1D Acopy = A.copy();
        ((DenseFloatMatrix1D) A).fft();
        ((DenseFloatMatrix1D) A).ifft(true);
        for (int i = 0; i < (int) A.size(); i++) {
            assertEquals(Acopy.getQuick(i), A.getQuick(i), TOL);
        }
    }

    public void testGetFft() {
        FloatMatrix1D Acopy = A.copy();
        FComplexMatrix1D ac = ((DenseFloatMatrix1D) A).getFft();
        ((DenseFComplexMatrix1D) ac).ifft(true);
        for (int i = 0; i < (int) A.size(); i++) {
            float[] elem = ac.getQuick(i);
            assertEquals(Acopy.getQuick(i), elem[0], TOL);
            assertEquals(0, elem[1], TOL);
        }
    }

    public void testGetIfft() {
        FloatMatrix1D Acopy = A.copy();
        FComplexMatrix1D ac = ((DenseFloatMatrix1D) A).getIfft(true);
        ((DenseFComplexMatrix1D) ac).fft();
        for (int i = 0; i < (int) A.size(); i++) {
            float[] elem = ac.getQuick(i);
            assertEquals(Acopy.getQuick(i), elem[0], TOL);
            assertEquals(0, elem[1], TOL);
        }
    }

}
