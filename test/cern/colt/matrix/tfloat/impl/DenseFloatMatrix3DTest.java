package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix3D;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix3D;
import cern.colt.matrix.tfloat.FloatMatrix3D;
import cern.colt.matrix.tfloat.FloatMatrix3DTest;

public class DenseFloatMatrix3DTest extends FloatMatrix3DTest {

    public DenseFloatMatrix3DTest(String arg0) {
        super(arg0);
    }

    protected void createMatrices() throws Exception {
        A = new DenseFloatMatrix3D(NSLICES, NROWS, NCOLUMNS);
        B = new DenseFloatMatrix3D(NSLICES, NROWS, NCOLUMNS);
    }

    public void testDct3() {
        FloatMatrix3D Acopy = A.copy();
        ((DenseFloatMatrix3D) A).dct3(true);
        ((DenseFloatMatrix3D) A).idct3(true);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(0, Math.abs(Acopy.getQuick(s, r, c) - A.getQuick(s, r, c)), TOL);
                }
            }
        }
    }

    public void testDst3() {
        FloatMatrix3D Acopy = A.copy();
        ((DenseFloatMatrix3D) A).dst3(true);
        ((DenseFloatMatrix3D) A).idst3(true);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(0, Math.abs(Acopy.getQuick(s, r, c) - A.getQuick(s, r, c)), TOL);
                }
            }
        }
    }

    public void testDht3() {
        FloatMatrix3D Acopy = A.copy();
        ((DenseFloatMatrix3D) A).dht3();
        ((DenseFloatMatrix3D) A).idht3(true);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(0, Math.abs(Acopy.getQuick(s, r, c) - A.getQuick(s, r, c)), TOL);
                }
            }
        }
    }

    public void testFft3() {
        int nslices = 16;
        int nrows = 32;
        int ncolumns = 64;
        FloatMatrix3D A = new DenseFloatMatrix3D(nslices, nrows, ncolumns);
        FloatMatrix3D Acopy = A.copy();
        ((DenseFloatMatrix3D) A).fft3();
        ((DenseFloatMatrix3D) A).ifft3(true);
        for (int s = 0; s < nslices; s++) {
            for (int r = 0; r < nrows; r++) {
                for (int c = 0; c < ncolumns; c++) {
                    assertEquals(0, Math.abs(Acopy.getQuick(s, r, c) - A.getQuick(s, r, c)), TOL);
                }
            }
        }
    }

    public void testDct2Slices() {
        FloatMatrix3D Acopy = A.copy();
        ((DenseFloatMatrix3D) A).dct2Slices(true);
        ((DenseFloatMatrix3D) A).idct2Slices(true);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(0, Math.abs(Acopy.getQuick(s, r, c) - A.getQuick(s, r, c)), TOL);
                }
            }
        }
    }

    public void testDst2Slices() {
        FloatMatrix3D Acopy = A.copy();
        ((DenseFloatMatrix3D) A).dst2Slices(true);
        ((DenseFloatMatrix3D) A).idst2Slices(true);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(0, Math.abs(Acopy.getQuick(s, r, c) - A.getQuick(s, r, c)), TOL);
                }
            }
        }
    }

    public void testDft2Slices() {
        FloatMatrix3D Acopy = A.copy();
        ((DenseFloatMatrix3D) A).dht2Slices();
        ((DenseFloatMatrix3D) A).idht2Slices(true);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    assertEquals(0, Math.abs(Acopy.getQuick(s, r, c) - A.getQuick(s, r, c)), TOL);
                }
            }
        }
    }

    public void testGetFft3() {
        FComplexMatrix3D Ac = ((DenseFloatMatrix3D) A).getFft3();
        ((DenseFComplexMatrix3D) Ac).ifft3(true);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    float[] elem = Ac.getQuick(s, r, c);
                    assertEquals(A.getQuick(s, r, c), elem[0], TOL);
                    assertEquals(0, elem[1], TOL);
                }
            }
        }
    }

    public void testGetIfft3() {
        FComplexMatrix3D Ac = ((DenseFloatMatrix3D) A).getIfft3(true);
        ((DenseFComplexMatrix3D) Ac).fft3();
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    float[] elem = Ac.getQuick(s, r, c);
                    assertEquals(A.getQuick(s, r, c), elem[0], TOL);
                    assertEquals(0, elem[1], TOL);
                }
            }
        }
    }

    public void testGetFft2Slices() {
        FComplexMatrix3D Ac = ((DenseFloatMatrix3D) A).getFft2Slices();
        ((DenseFComplexMatrix3D) Ac).ifft2Slices(true);
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    float[] elem = Ac.getQuick(s, r, c);
                    assertEquals(0, Math.abs(A.getQuick(s, r, c) - elem[0]), TOL);
                    assertEquals(0, elem[1], TOL);
                }
            }
        }
    }

    public void testGetIfft2Slices() {
        FComplexMatrix3D Ac = ((DenseFloatMatrix3D) A).getIfft2Slices(true);
        ((DenseFComplexMatrix3D) Ac).fft2Slices();
        for (int s = 0; s < A.slices(); s++) {
            for (int r = 0; r < A.rows(); r++) {
                for (int c = 0; c < A.columns(); c++) {
                    float[] elem = Ac.getQuick(s, r, c);
                    assertEquals(0, Math.abs(A.getQuick(s, r, c) - elem[0]), TOL);
                    assertEquals(0, elem[1], TOL);
                }
            }
        }
    }

}
