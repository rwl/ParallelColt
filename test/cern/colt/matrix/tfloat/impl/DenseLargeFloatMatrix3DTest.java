package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix3D;
import cern.colt.matrix.tfcomplex.impl.DenseLargeFComplexMatrix3D;
import cern.colt.matrix.tfloat.FloatMatrix3D;
import cern.colt.matrix.tfloat.FloatMatrix3DTest;

public class DenseLargeFloatMatrix3DTest extends FloatMatrix3DTest {

    public DenseLargeFloatMatrix3DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new DenseLargeFloatMatrix3D(NSLICES, NROWS, NCOLUMNS);
        B = new DenseLargeFloatMatrix3D(NSLICES, NROWS, NCOLUMNS);
    }

    public void testDct3() {
        FloatMatrix3D Acopy = A.copy();
        ((WrapperFloatMatrix3D) A).dct3(true);
        ((WrapperFloatMatrix3D) A).idct3(true);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testDst3() {
        FloatMatrix3D Acopy = A.copy();
        ((WrapperFloatMatrix3D) A).dst3(true);
        ((WrapperFloatMatrix3D) A).idst3(true);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testDht3() {
        FloatMatrix3D Acopy = A.copy();
        ((WrapperFloatMatrix3D) A).dht3();
        ((WrapperFloatMatrix3D) A).idht3(true);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testFft3() {
        int nslices = 16;
        int nrows = 32;
        int ncolumns = 64;
        FloatMatrix3D A = new DenseLargeFloatMatrix3D(nslices, nrows, ncolumns);
        FloatMatrix3D Acopy = A.copy();
        ((WrapperFloatMatrix3D) A).fft3();
        ((WrapperFloatMatrix3D) A).ifft3(true);
        for (int s = 0; s < nslices; s++) {
            for (int r = 0; r < nrows; r++) {
                for (int c = 0; c < ncolumns; c++) {
                    assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                }
            }
        }

        A = A.viewDice(2, 1, 0);
        Acopy = A.copy();
        ((WrapperFloatMatrix3D) A).fft3();
        ((WrapperFloatMatrix3D) A).ifft3(true);
        for (int s = 0; s < ncolumns; s++) {
            for (int r = 0; r < nrows; r++) {
                for (int c = 0; c < nslices; c++) {
                    assertEquals(0, Math.abs(Acopy.getQuick(s, r, c) - A.getQuick(s, r, c)), TOL);
                }
            }
        }

    }

    public void testDct2Slices() {
        FloatMatrix3D Acopy = A.copy();
        ((WrapperFloatMatrix3D) A).dct2Slices(true);
        ((WrapperFloatMatrix3D) A).idct2Slices(true);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testDst2Slices() {
        FloatMatrix3D Acopy = A.copy();
        ((WrapperFloatMatrix3D) A).dst2Slices(true);
        ((WrapperFloatMatrix3D) A).idst2Slices(true);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testDft2Slices() {
        FloatMatrix3D Acopy = A.copy();
        ((WrapperFloatMatrix3D) A).dht2Slices();
        ((WrapperFloatMatrix3D) A).idht2Slices(true);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    assertEquals(Acopy.getQuick(s, r, c), A.getQuick(s, r, c), TOL);
                }
            }
        }
    }

    public void testGetFft3() {
        FComplexMatrix3D Ac = ((WrapperFloatMatrix3D) A).getFft3();
        ((DenseLargeFComplexMatrix3D) Ac).ifft3(true);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    float[] elem = Ac.getQuick(s, r, c);
                    assertEquals(A.getQuick(s, r, c), elem[0], TOL);
                    assertEquals(0, elem[1], TOL);
                }
            }
        }
    }

    public void testGetIfft3() {
        FComplexMatrix3D Ac = ((WrapperFloatMatrix3D) A).getIfft3(true);
        ((DenseLargeFComplexMatrix3D) Ac).fft3();
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    float[] elem = Ac.getQuick(s, r, c);
                    assertEquals(A.getQuick(s, r, c), elem[0], TOL);
                    assertEquals(0, elem[1], TOL);
                }
            }
        }
    }

    public void testGetFft2Slices() {
        FComplexMatrix3D Ac = ((WrapperFloatMatrix3D) A).getFft2Slices();
        ((DenseLargeFComplexMatrix3D) Ac).ifft2Slices(true);
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    float[] elem = Ac.getQuick(s, r, c);
                    assertEquals(A.getQuick(s, r, c), elem[0], TOL);
                    assertEquals(0, elem[1], TOL);
                }
            }
        }
    }

    public void testGetIfft2Slices() {
        FComplexMatrix3D Ac = ((WrapperFloatMatrix3D) A).getIfft2Slices(true);
        ((DenseLargeFComplexMatrix3D) Ac).fft2Slices();
        for (int s = 0; s < NSLICES; s++) {
            for (int r = 0; r < NROWS; r++) {
                for (int c = 0; c < NCOLUMNS; c++) {
                    float[] elem = Ac.getQuick(s, r, c);
                    assertEquals(A.getQuick(s, r, c), elem[0], TOL);
                    assertEquals(0, elem[1], TOL);
                }
            }
        }
    }
}
