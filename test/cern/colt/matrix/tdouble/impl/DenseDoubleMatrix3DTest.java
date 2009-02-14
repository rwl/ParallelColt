package cern.colt.matrix.tdouble.impl;

import cern.colt.matrix.tdcomplex.DComplexMatrix3D;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix3D;
import cern.colt.matrix.tdouble.DoubleMatrix3D;
import cern.colt.matrix.tdouble.DoubleMatrix3DTest;

public class DenseDoubleMatrix3DTest extends DoubleMatrix3DTest {

    public DenseDoubleMatrix3DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new DenseDoubleMatrix3D(NSLICES, NROWS, NCOLUMNS);
        B = new DenseDoubleMatrix3D(NSLICES, NROWS, NCOLUMNS);
    }

    public void testDct3() {
        DoubleMatrix3D Acopy = A.copy();
        ((DenseDoubleMatrix3D) A).dct3(true);
        ((DenseDoubleMatrix3D) A).idct3(true);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(0, Math.abs(Acopy.getQuick(s, r, c) - A.getQuick(s, r, c)), TOL);
                }
            }
        }
    }

    public void testDst3() {
        DoubleMatrix3D Acopy = A.copy();
        ((DenseDoubleMatrix3D) A).dst3(true);
        ((DenseDoubleMatrix3D) A).idst3(true);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(0, Math.abs(Acopy.getQuick(s, r, c) - A.getQuick(s, r, c)), TOL);
                }
            }
        }
    }
    
    public void testDht3() {
        DoubleMatrix3D Acopy = A.copy();
        ((DenseDoubleMatrix3D) A).dht3();
        ((DenseDoubleMatrix3D) A).idht3(true);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(0, Math.abs(Acopy.getQuick(s, r, c) - A.getQuick(s, r, c)), TOL);
                }
            }
        }
    }
    
    public void testFft3() {
    	int nslices = 16;
    	int nrows = 32;
    	int ncolumns = 64;
        DoubleMatrix3D A = new DenseDoubleMatrix3D(nslices, nrows, ncolumns);
        DoubleMatrix3D Acopy = A.copy();
        ((DenseDoubleMatrix3D) A).fft3();
        ((DenseDoubleMatrix3D) A).ifft3(true);
        for (int s = 0; s < nslices; s++) {
            for (int r = 0; r < nrows; r++) {
                for (int c = 0; c < ncolumns; c++) {
                    assertEquals(0, Math.abs(Acopy.getQuick(s, r, c) - A.getQuick(s, r, c)), TOL);
                }
            }
        }
    }
    
    public void testDct2Slices() {
        DoubleMatrix3D Acopy = A.copy();
        ((DenseDoubleMatrix3D)A).dct2Slices(true);
        ((DenseDoubleMatrix3D)A).idct2Slices(true);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(0, Math.abs(Acopy.getQuick(s, r, c) - A.getQuick(s, r, c)), TOL);
                }
            }
        }
    }

    public void testDst2Slices() {
        DoubleMatrix3D Acopy = A.copy();
        ((DenseDoubleMatrix3D)A).dst2Slices(true);
        ((DenseDoubleMatrix3D)A).idst2Slices(true);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(0, Math.abs(Acopy.getQuick(s, r, c) - A.getQuick(s, r, c)), TOL);
                }
            }
        }
    }
    
    public void testDft2Slices() {
        DoubleMatrix3D Acopy = A.copy();
        ((DenseDoubleMatrix3D)A).dht2Slices();
        ((DenseDoubleMatrix3D)A).idht2Slices(true);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(0, Math.abs(Acopy.getQuick(s, r, c) - A.getQuick(s, r, c)), TOL);
                }
            }
        }
    }

    public void testGetFft3() {
        DComplexMatrix3D Ac = ((DenseDoubleMatrix3D)A).getFft3();
        ((DenseDComplexMatrix3D)Ac).ifft3(true);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    double[] elem  = Ac.getQuick(s, r, c);
                    assertEquals(A.getQuick(s, r, c), elem[0], TOL);
                    assertEquals(0, elem[1], TOL);                    
                }
            }
        }
    }


    public void testGetIfft3() {
        DComplexMatrix3D Ac = ((DenseDoubleMatrix3D)A).getIfft3(true);
        ((DenseDComplexMatrix3D)Ac).fft3();
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    double[] elem  = Ac.getQuick(s, r, c);
                    assertEquals(A.getQuick(s, r, c),  elem[0], TOL);
                    assertEquals(0, elem[1], TOL);                    
                }
            }
        }
    }
    
    public void testGetFft2Slices() {
        DComplexMatrix3D Ac = ((DenseDoubleMatrix3D)A).getFft2Slices();
        ((DenseDComplexMatrix3D)Ac).ifft2Slices(true);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    double[] elem  = Ac.getQuick(s, r, c);
                    assertEquals(0, Math.abs(A.getQuick(s, r, c) - elem[0]), TOL);
                    assertEquals(0, elem[1], TOL);                    
                }
            }
        }
    }

    public void testGetIfft2Slices() {
        DComplexMatrix3D Ac = ((DenseDoubleMatrix3D)A).getIfft2Slices(true);
        ((DenseDComplexMatrix3D)Ac).fft2Slices();
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    double[] elem  = Ac.getQuick(s, r, c);
                    assertEquals(0, Math.abs(A.getQuick(s, r, c) - elem[0]), TOL);
                    assertEquals(0, elem[1], TOL);                    
                }
            }
        }
    }

    
    
}
