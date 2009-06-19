package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfcomplex.FComplexMatrix2D;
import cern.colt.matrix.tfcomplex.impl.DenseLargeFComplexMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix2DTest;

public class DenseLargeFloatMatrix2DTest extends FloatMatrix2DTest {

    public DenseLargeFloatMatrix2DTest(String arg0) {
        super(arg0);
    }

    @Override
    protected void createMatrices() throws Exception {
        A = new DenseLargeFloatMatrix2D(NROWS, NCOLUMNS);
        B = new DenseLargeFloatMatrix2D(NROWS, NCOLUMNS);
        Bt = new DenseLargeFloatMatrix2D(NCOLUMNS, NROWS);
    }

    public void testDct2() {
        FloatMatrix2D Acopy = A.copy();
        ((WrapperFloatMatrix2D) A).dct2(true);
        ((WrapperFloatMatrix2D) A).idct2(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDctColumns() {
        FloatMatrix2D Acopy = A.copy();
        ((WrapperFloatMatrix2D) A).dctColumns(true);
        ((WrapperFloatMatrix2D) A).idctColumns(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDctRows() {
        FloatMatrix2D Acopy = A.copy();
        ((WrapperFloatMatrix2D) A).dctRows(true);
        ((WrapperFloatMatrix2D) A).idctRows(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDht2() {
        FloatMatrix2D Acopy = A.copy();
        ((WrapperFloatMatrix2D) A).dht2();
        ((WrapperFloatMatrix2D) A).idht2(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDhtColumns() {
        FloatMatrix2D Acopy = A.copy();
        ((WrapperFloatMatrix2D) A).dhtColumns();
        ((WrapperFloatMatrix2D) A).idhtColumns(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDhtRows() {
        FloatMatrix2D Acopy = A.copy();
        ((WrapperFloatMatrix2D) A).dhtRows();
        ((WrapperFloatMatrix2D) A).idhtRows(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDst2() {
        FloatMatrix2D Acopy = A.copy();
        ((WrapperFloatMatrix2D) A).dst2(true);
        ((WrapperFloatMatrix2D) A).idst2(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDstColumns() {
        FloatMatrix2D Acopy = A.copy();
        ((WrapperFloatMatrix2D) A).dstColumns(true);
        ((WrapperFloatMatrix2D) A).idstColumns(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testDstRows() {
        FloatMatrix2D Acopy = A.copy();
        ((WrapperFloatMatrix2D) A).dstRows(true);
        ((WrapperFloatMatrix2D) A).idstRows(true);
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++)
                assertEquals(0, Math.abs(Acopy.getQuick(r, c) - A.getQuick(r, c)), TOL);
        }
    }

    public void testFft2() {
        int nrows = 64;
        int ncolumns = 128;
        FloatMatrix2D A = new DenseLargeFloatMatrix2D(nrows, ncolumns);
        FloatMatrix2D Acopy = A.copy();
        ((DenseLargeFloatMatrix2D) A).fft2();
        ((DenseLargeFloatMatrix2D) A).ifft2(true);
        for (int r = 0; r < nrows; r++) {
            for (int c = 0; c < ncolumns; c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }

        A = A.viewDice();
        Acopy = A.copy();
        ((WrapperFloatMatrix2D) A).fft2();
        ((WrapperFloatMatrix2D) A).ifft2(true);
        for (int r = 0; r < ncolumns; r++) {
            for (int c = 0; c < nrows; c++) {
                assertEquals(Acopy.getQuick(r, c), A.getQuick(r, c), TOL);
            }
        }
    }

    public void testGetFft2() {
        FloatMatrix2D Acopy = A.copy();
        FComplexMatrix2D Ac = ((WrapperFloatMatrix2D) A).getFft2();
        ((DenseLargeFComplexMatrix2D) Ac).ifft2(true);
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
        FComplexMatrix2D Ac = ((WrapperFloatMatrix2D) A).getIfft2(true);
        ((DenseLargeFComplexMatrix2D) Ac).fft2();
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
        FComplexMatrix2D Ac = ((WrapperFloatMatrix2D) A).getFftColumns();
        ((DenseLargeFComplexMatrix2D) Ac).ifftColumns(true);
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
        FComplexMatrix2D Ac = ((WrapperFloatMatrix2D) A).getIfftColumns(true);
        ((DenseLargeFComplexMatrix2D) Ac).fftColumns();
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
        FComplexMatrix2D Ac = ((WrapperFloatMatrix2D) A).getFftRows();
        ((DenseLargeFComplexMatrix2D) Ac).ifftRows(true);
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
        FComplexMatrix2D Ac = ((WrapperFloatMatrix2D) A).getIfftRows(true);
        ((DenseLargeFComplexMatrix2D) Ac).fftRows();
        for (int r = 0; r < NROWS; r++) {
            for (int c = 0; c < NCOLUMNS; c++) {
                float[] elemAc = Ac.getQuick(r, c);
                assertEquals(Acopy.getQuick(r, c), elemAc[0], TOL);
                assertEquals(0, elemAc[1], TOL);
            }
        }
    }
}
