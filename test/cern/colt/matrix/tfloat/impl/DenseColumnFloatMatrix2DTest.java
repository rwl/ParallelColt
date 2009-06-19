package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix2DTest;

public class DenseColumnFloatMatrix2DTest extends FloatMatrix2DTest {

    public DenseColumnFloatMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new DenseColumnFloatMatrix2D(NROWS, NCOLUMNS);
        B = new DenseColumnFloatMatrix2D(NROWS, NCOLUMNS);
        Bt = new DenseColumnFloatMatrix2D(NCOLUMNS, NROWS);
    }

    public void testAssignFloatArray() {
        float[] expected = new float[(int) A.size()];
        for (int i = 0; i < A.size(); i++) {
            expected[i] = (float) Math.random();
        }
        A.assign(expected);
        int idx = 0;
        for (int c = 0; c < A.columns(); c++) {
            for (int r = 0; r < A.rows(); r++) {
                assertEquals(0, Math.abs(expected[idx++] - A.getQuick(r, c)), TOL);
            }
        }
    }

    public void testDct2() {
        FloatMatrix2D Acopy = A.copy();
        ((DenseColumnFloatMatrix2D) A).dct2(true);
        ((DenseColumnFloatMatrix2D) A).idct2(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDctColumns() {
        FloatMatrix2D Acopy = A.copy();
        ((DenseColumnFloatMatrix2D) A).dctColumns(true);
        ((DenseColumnFloatMatrix2D) A).idctColumns(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDctRows() {
        FloatMatrix2D Acopy = A.copy();
        ((DenseColumnFloatMatrix2D) A).dctRows(true);
        ((DenseColumnFloatMatrix2D) A).idctRows(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDht2() {
        FloatMatrix2D Acopy = A.copy();
        ((DenseColumnFloatMatrix2D) A).dht2();
        ((DenseColumnFloatMatrix2D) A).idht2(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDhtColumns() {
        FloatMatrix2D Acopy = A.copy();
        ((DenseColumnFloatMatrix2D) A).dhtColumns();
        ((DenseColumnFloatMatrix2D) A).idhtColumns(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDhtRows() {
        FloatMatrix2D Acopy = A.copy();
        ((DenseColumnFloatMatrix2D) A).dhtRows();
        ((DenseColumnFloatMatrix2D) A).idhtRows(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDst2() {
        FloatMatrix2D Acopy = A.copy();
        ((DenseColumnFloatMatrix2D) A).dst2(true);
        ((DenseColumnFloatMatrix2D) A).idst2(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDstColumns() {
        FloatMatrix2D Acopy = A.copy();
        ((DenseColumnFloatMatrix2D) A).dstColumns(true);
        ((DenseColumnFloatMatrix2D) A).idstColumns(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDstRows() {
        FloatMatrix2D Acopy = A.copy();
        ((DenseColumnFloatMatrix2D) A).dstRows(true);
        ((DenseColumnFloatMatrix2D) A).idstRows(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testFft2() {
        int nrows = 64;
        int ncolumns = 128;
        FloatMatrix2D A = new DenseColumnFloatMatrix2D(nrows, ncolumns);
        FloatMatrix2D Acopy = A.copy();
        ((DenseColumnFloatMatrix2D) A).fft2();
        ((DenseColumnFloatMatrix2D) A).ifft2(true);
        for (int r = 0; r < nrows; r++) {
            for (int c = 0; c < ncolumns; c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testGetFft2() {
        FloatMatrix2D Acopy = A.copy();
        FComplexMatrix2D Ac = ((DenseColumnFloatMatrix2D) A).getFft2();
        ((DenseFComplexMatrix2D) Ac).ifft2(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                float[] elemAc = Ac.getQuick(r, c);
                assertEquals(Acopy.getQuick(r, c), elemAc[0], TOL);
                assertEquals(0, elemAc[1], TOL);
            }
        }
    }

    public void testGetIfft2() {
        FloatMatrix2D Acopy = A.copy();
        FComplexMatrix2D Ac = ((DenseColumnFloatMatrix2D) A).getIfft2(true);
        ((DenseFComplexMatrix2D) Ac).fft2();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                float[] elemAc = Ac.getQuick(r, c);
                assertEquals(Acopy.getQuick(r, c), elemAc[0], TOL);
                assertEquals(0, elemAc[1], TOL);
            }
        }
    }

    public void testGetFftColumns() {
        FloatMatrix2D Acopy = A.copy();
        FComplexMatrix2D Ac = ((DenseColumnFloatMatrix2D) A).getFftColumns();
        ((DenseFComplexMatrix2D) Ac).ifftColumns(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                float[] elemAc = Ac.getQuick(r, c);
                assertEquals(Acopy.getQuick(r, c), elemAc[0], TOL);
                assertEquals(0, elemAc[1], TOL);
            }
        }
    }

    public void testGetIfftColumns() {
        FloatMatrix2D Acopy = A.copy();
        FComplexMatrix2D Ac = ((DenseColumnFloatMatrix2D) A).getIfftColumns(true);
        ((DenseFComplexMatrix2D) Ac).fftColumns();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                float[] elemAc = Ac.getQuick(r, c);
                assertEquals(Acopy.getQuick(r, c), elemAc[0], TOL);
                assertEquals(0, elemAc[1], TOL);
            }
        }
    }

    public void testGetFftRows() {
        FloatMatrix2D Acopy = A.copy();
        FComplexMatrix2D Ac = ((DenseColumnFloatMatrix2D) A).getFftRows();
        ((DenseFComplexMatrix2D) Ac).ifftRows(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                float[] elemAc = Ac.getQuick(r, c);
                assertEquals(Acopy.getQuick(r, c), elemAc[0], TOL);
                assertEquals(0, elemAc[1], TOL);
            }
        }
    }

    public void testGetIfftRows() {
        FloatMatrix2D Acopy = A.copy();
        FComplexMatrix2D Ac = ((DenseColumnFloatMatrix2D) A).getIfftRows(true);
        ((DenseFComplexMatrix2D) Ac).fftRows();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                float[] elemAc = Ac.getQuick(r, c);
                assertEquals(Acopy.getQuick(r, c), elemAc[0], TOL);
                assertEquals(0, elemAc[1], TOL);
            }
        }
    }

}
