package cern.colt.matrix.tdouble.impl;

import cern.colt.matrix.tdcomplex.DComplexMatrix1D;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix1DTest;

public class DenseDoubleMatrix1DTest extends DoubleMatrix1DTest {

    public DenseDoubleMatrix1DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new DenseDoubleMatrix1D(SIZE);
        B = new DenseDoubleMatrix1D(SIZE);
    }

    public void testDct() {
        DoubleMatrix1D Acopy = A.copy();
        ((DenseDoubleMatrix1D) A).dct(true);
        ((DenseDoubleMatrix1D) A).idct(true);
        for (int i = 0; i < SIZE; i++) {
            assertEquals(Acopy.getQuick(i), A.getQuick(i), TOL);
        }
    }

    public void testDst() {
        DoubleMatrix1D Acopy = A.copy();
        ((DenseDoubleMatrix1D) A).dst(true);
        ((DenseDoubleMatrix1D) A).idst(true);
        for (int i = 0; i < SIZE; i++) {
            assertEquals(Acopy.getQuick(i), A.getQuick(i), TOL);
        }
    }

    public void testDht() {
        DoubleMatrix1D Acopy = A.copy();
        ((DenseDoubleMatrix1D) A).dht();
        ((DenseDoubleMatrix1D) A).idht(true);
        for (int i = 0; i < SIZE; i++) {
            assertEquals(Acopy.getQuick(i), A.getQuick(i), TOL);
        }
    }

    public void testFft() {
        DoubleMatrix1D Acopy = A.copy();
        ((DenseDoubleMatrix1D) A).fft();
        ((DenseDoubleMatrix1D) A).ifft(true);
        for (int i = 0; i < SIZE; i++) {
            assertEquals(Acopy.getQuick(i), A.getQuick(i), TOL);
        }
    }

    public void testGetFft() {
        DoubleMatrix1D Acopy = A.copy();
        DComplexMatrix1D ac = ((DenseDoubleMatrix1D) A).getFft();
        ((DenseDComplexMatrix1D) ac).ifft(true);
        for (int i = 0; i < SIZE; i++) {
            double[] elem = ac.getQuick(i);
            assertEquals(Acopy.getQuick(i), elem[0], TOL);
            assertEquals(0, elem[1], TOL);
        }
    }

    public void testGetIfft() {
        DoubleMatrix1D Acopy = A.copy();
        DComplexMatrix1D ac = ((DenseDoubleMatrix1D) A).getIfft(true);
        ((DenseDComplexMatrix1D) ac).fft();
        for (int i = 0; i < SIZE; i++) {
            double[] elem = ac.getQuick(i);
            assertEquals(Acopy.getQuick(i), elem[0], TOL);
            assertEquals(0, elem[1], TOL);
        }
    }

}