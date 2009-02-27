package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfcomplex.impl.DenseFComplexMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix2DTest;

public class DenseColFloatMatrix2DTest extends FloatMatrix2DTest {

    public DenseColFloatMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new DenseColFloatMatrix2D(NROWS, NCOLUMNS);
        B = new DenseColFloatMatrix2D(NROWS, NCOLUMNS);
        Bt = new DenseColFloatMatrix2D(NCOLUMNS, NROWS);
    }

    public void testAssignFloatArray() {
        float[] expected = new float[NROWS * NCOLUMNS];
        for (int i = 0; i < NROWS * NCOLUMNS; i++) {
            expected[i] = (float) Math.random();
        }
        A.assign(expected);
        int idx = 0;
        for (int c = 0; c < NCOLUMNS; c++) {
            for (int r = 0; r < NROWS; r++) {
                assertEquals(expected[idx++], A.getQuick(r, c), TOL);
            }
        }
    }

    public void testDct2() {
        FloatMatrix2D Acopy = A.copy();
        ((DenseColFloatMatrix2D) A).dct2(true);
        ((DenseColFloatMatrix2D) A).idct2(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDctColumns() {
        FloatMatrix2D Acopy = A.copy();
        ((DenseColFloatMatrix2D) A).dctColumns(true);
        ((DenseColFloatMatrix2D) A).idctColumns(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDctRows() {
        FloatMatrix2D Acopy = A.copy();
        ((DenseColFloatMatrix2D) A).dctRows(true);
        ((DenseColFloatMatrix2D) A).idctRows(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDht2() {
        FloatMatrix2D Acopy = A.copy();
        ((DenseColFloatMatrix2D) A).dht2();
        ((DenseColFloatMatrix2D) A).idht2(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDhtColumns() {
        FloatMatrix2D Acopy = A.copy();
        ((DenseColFloatMatrix2D) A).dhtColumns();
        ((DenseColFloatMatrix2D) A).idhtColumns(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDhtRows() {
        FloatMatrix2D Acopy = A.copy();
        ((DenseColFloatMatrix2D) A).dhtRows();
        ((DenseColFloatMatrix2D) A).idhtRows(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDst2() {
        FloatMatrix2D Acopy = A.copy();
        ((DenseColFloatMatrix2D) A).dst2(true);
        ((DenseColFloatMatrix2D) A).idst2(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDstColumns() {
        FloatMatrix2D Acopy = A.copy();
        ((DenseColFloatMatrix2D) A).dstColumns(true);
        ((DenseColFloatMatrix2D) A).idstColumns(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDstRows() {
        FloatMatrix2D Acopy = A.copy();
        ((DenseColFloatMatrix2D) A).dstRows(true);
        ((DenseColFloatMatrix2D) A).idstRows(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testFft2() {
        int nrows = 64;
        int ncolumns = 128;
        FloatMatrix2D A = new DenseColFloatMatrix2D(nrows, ncolumns);
        FloatMatrix2D Acopy = A.copy();
        ((DenseColFloatMatrix2D) A).fft2();
        ((DenseColFloatMatrix2D) A).ifft2(true);
        for (int r = 0; r < nrows; r++) {
            for (int c = 0; c < ncolumns; c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testGetFft2() {
        FloatMatrix2D Acopy = A.copy();
        FComplexMatrix2D Ac = ((DenseColFloatMatrix2D) A).getFft2();
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
        FComplexMatrix2D Ac = ((DenseColFloatMatrix2D) A).getIfft2(true);
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
        FComplexMatrix2D Ac = ((DenseColFloatMatrix2D) A).getFftColumns();
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
        FComplexMatrix2D Ac = ((DenseColFloatMatrix2D) A).getIfftColumns(true);
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
        FComplexMatrix2D Ac = ((DenseColFloatMatrix2D) A).getFftRows();
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
        FComplexMatrix2D Ac = ((DenseColFloatMatrix2D) A).getIfftRows(true);
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
