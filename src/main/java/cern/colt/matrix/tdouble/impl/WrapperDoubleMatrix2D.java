/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tdouble.impl;

import java.util.concurrent.Future;

import cern.colt.list.tdouble.DoubleArrayList;
import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tdcomplex.impl.DenseLargeDComplexMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * 2-d matrix holding <tt>double</tt> elements; either a view wrapping another
 * matrix or a matrix whose views are wrappers.
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 04/14/2000
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class WrapperDoubleMatrix2D extends DoubleMatrix2D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;
    /*
     * The elements of the matrix.
     */
    protected DoubleMatrix2D content;

    public WrapperDoubleMatrix2D(DoubleMatrix2D newContent) {
        if (newContent != null)
            try {
                setUp(newContent.rows(), newContent.columns());
            } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
                if (!"matrix too large".equals(exc.getMessage()))
                    throw exc;
            }
        this.content = newContent;
    }

    public DoubleMatrix2D assign(final double[] values) {
        if (content instanceof DiagonalDoubleMatrix2D) {
            int dlength = ((DiagonalDoubleMatrix2D) content).dlength;
            final double[] elems = ((DiagonalDoubleMatrix2D) content).elements;
            if (values.length != dlength)
                throw new IllegalArgumentException("Must have same length: length=" + values.length + " dlength="
                        + dlength);
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (dlength >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, dlength);
                Future<?>[] futures = new Future[nthreads];
                int k = dlength / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstIdx = j * k;
                    final int lastIdx = (j == nthreads - 1) ? dlength : firstIdx + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {

                        public void run() {
                            for (int i = firstIdx; i < lastIdx; i++) {
                                elems[i] = values[i];
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                for (int i = 0; i < dlength; i++) {
                    elems[i] = values[i];
                }
            }
            return this;
        } else {
            return super.assign(values);
        }
    }

    public DoubleMatrix2D assign(final float[] values) {
        if (content instanceof DiagonalDoubleMatrix2D) {
            int dlength = ((DiagonalDoubleMatrix2D) content).dlength;
            final double[] elems = ((DiagonalDoubleMatrix2D) content).elements;
            if (values.length != dlength)
                throw new IllegalArgumentException("Must have same length: length=" + values.length + " dlength="
                        + dlength);
            int nthreads = ConcurrencyUtils.getNumberOfThreads();
            if ((nthreads > 1) && (dlength >= ConcurrencyUtils.getThreadsBeginN_2D())) {
                nthreads = Math.min(nthreads, dlength);
                Future<?>[] futures = new Future[nthreads];
                int k = dlength / nthreads;
                for (int j = 0; j < nthreads; j++) {
                    final int firstIdx = j * k;
                    final int lastIdx = (j == nthreads - 1) ? dlength : firstIdx + k;
                    futures[j] = ConcurrencyUtils.submit(new Runnable() {

                        public void run() {
                            for (int i = firstIdx; i < lastIdx; i++) {
                                elems[i] = values[i];
                            }
                        }
                    });
                }
                ConcurrencyUtils.waitForCompletion(futures);
            } else {
                for (int i = 0; i < dlength; i++) {
                    elems[i] = values[i];
                }
            }
            return this;
        } else {
            return super.assign(values);
        }
    }

    public DoubleMatrix2D assign(final DoubleMatrix2D y, final cern.colt.function.tdouble.DoubleDoubleFunction function) {
        checkShape(y);
        if (y instanceof WrapperDoubleMatrix2D) {
            IntArrayList rowList = new IntArrayList();
            IntArrayList columnList = new IntArrayList();
            DoubleArrayList valueList = new DoubleArrayList();
            y.getNonZeros(rowList, columnList, valueList);
            assign(y, function, rowList, columnList);
        } else {
            super.assign(y, function);
        }
        return this;
    }

    public Object elements() {
        return content.elements();
    }

    public synchronized double getQuick(int row, int column) {
        return content.getQuick(row, column);
    }

    public boolean equals(double value) {
        if (content instanceof DiagonalDoubleMatrix2D) {
            double epsilon = cern.colt.matrix.tdouble.algo.DoubleProperty.DEFAULT.tolerance();
            double[] elements = (double[]) content.elements();
            for (int r = 0; r < elements.length; r++) {
                double x = elements[r];
                double diff = Math.abs(value - x);
                if ((diff != diff) && ((value != value && x != x) || value == x))
                    diff = 0;
                if (!(diff <= epsilon)) {
                    return false;
                }
            }
            return true;
        } else {
            return super.equals(value);
        }
    }

    public boolean equals(Object obj) {
        if (content instanceof DiagonalDoubleMatrix2D && obj instanceof DiagonalDoubleMatrix2D) {
            double epsilon = cern.colt.matrix.tdouble.algo.DoubleProperty.DEFAULT.tolerance();
            if (this == obj)
                return true;
            if (!(this != null && obj != null))
                return false;
            DiagonalDoubleMatrix2D A = (DiagonalDoubleMatrix2D) content;
            DiagonalDoubleMatrix2D B = (DiagonalDoubleMatrix2D) obj;
            if (A.columns() != B.columns() || A.rows() != B.rows() || A.diagonalIndex() != B.diagonalIndex()
                    || A.diagonalLength() != B.diagonalLength())
                return false;
            double[] AElements = A.elements();
            double[] BElements = B.elements();
            for (int r = 0; r < AElements.length; r++) {
                double x = AElements[r];
                double value = BElements[r];
                double diff = Math.abs(value - x);
                if ((diff != diff) && ((value != value && x != x) || value == x))
                    diff = 0;
                if (!(diff <= epsilon)) {
                    return false;
                }
            }
            return true;
        } else {
            return super.equals(obj);
        }
    }

    public DoubleMatrix2D like(int rows, int columns) {
        return content.like(rows, columns);
    }

    public DoubleMatrix1D like1D(int size) {
        return content.like1D(size);
    }

    /**
     * Computes the 2D discrete cosine transform (DCT-II) of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void dct2(boolean scale) {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).dct2(scale);
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.dct2(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the discrete cosine transform (DCT-II) of each column of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void dctColumns(final boolean scale) {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).dctColumns(scale);
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.dctColumns(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the discrete cosine transform (DCT-II) of each row of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void dctRows(final boolean scale) {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).dctRows(scale);
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.dctRows(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 2D discrete sine transform (DST-II) of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void dst2(boolean scale) {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).dst2(scale);
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.dst2(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the discrete sine transform (DST-II) of each column of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void dstColumns(final boolean scale) {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).dstColumns(scale);
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.dstColumns(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the discrete sine transform (DST-II) of each row of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void dstRows(final boolean scale) {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).dstRows(scale);
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.dstRows(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 2D discrete Hartley transform (DHT) of this matrix.
     */
    public void dht2() {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).dht2();
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.dht2();
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the discrete Hertley transform (DHT) of each column of this
     * matrix.
     */
    public void dhtColumns() {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).dhtColumns();
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.dhtColumns();
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the discrete Hertley transform (DHT) of each row of this matrix.
     */
    public void dhtRows() {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).dhtRows();
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.dhtRows();
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 2D discrete Fourier transform (DFT) of this matrix. The
     * physical layout of the output data is as follows:
     * 
     * <pre>
     * this[k1][2*k2] = Re[k1][k2] = Re[rows-k1][columns-k2], 
     * this[k1][2*k2+1] = Im[k1][k2] = -Im[rows-k1][columns-k2], 
     *       0&lt;k1&lt;rows, 0&lt;k2&lt;columns/2, 
     * this[0][2*k2] = Re[0][k2] = Re[0][columns-k2], 
     * this[0][2*k2+1] = Im[0][k2] = -Im[0][columns-k2], 
     *       0&lt;k2&lt;columns/2, 
     * this[k1][0] = Re[k1][0] = Re[rows-k1][0], 
     * this[k1][1] = Im[k1][0] = -Im[rows-k1][0], 
     * this[rows-k1][1] = Re[k1][columns/2] = Re[rows-k1][columns/2], 
     * this[rows-k1][0] = -Im[k1][columns/2] = Im[rows-k1][columns/2], 
     *       0&lt;k1&lt;rows/2, 
     * this[0][0] = Re[0][0], 
     * this[0][1] = Re[0][columns/2], 
     * this[rows/2][0] = Re[rows/2][0], 
     * this[rows/2][1] = Re[rows/2][columns/2]
     * </pre>
     * 
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * forward transform, use <code>getFft2</code>. To get back the original
     * data, use <code>ifft2</code>.
     * 
     * @throws IllegalArgumentException
     *             if the row size or the column size of this matrix is not a
     *             power of 2 number.
     * 
     */
    public void fft2() {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).fft2();
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.fft2();
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Returns new complex matrix which is the 2D discrete Fourier transform
     * (DFT) of this matrix.
     * 
     * @return the 2D discrete Fourier transform (DFT) of this matrix.
     * 
     */
    public DenseLargeDComplexMatrix2D getFft2() {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                return ((DenseLargeDoubleMatrix2D) content).getFft2();
            } else {
                return ((DenseLargeDoubleMatrix2D) copy()).getFft2();
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Returns new complex matrix which is the 2D inverse of the discrete
     * Fourier transform (IDFT) of this matrix.
     * 
     * @return the 2D inverse of the discrete Fourier transform (IDFT) of this
     *         matrix.
     */
    public DenseLargeDComplexMatrix2D getIfft2(boolean scale) {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                return ((DenseLargeDoubleMatrix2D) content).getIfft2(scale);
            } else {
                return ((DenseLargeDoubleMatrix2D) copy()).getIfft2(scale);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Returns new complex matrix which is the discrete Fourier transform (DFT)
     * of each column of this matrix.
     * 
     * @return the discrete Fourier transform (DFT) of each column of this
     *         matrix.
     */
    public DenseLargeDComplexMatrix2D getFftColumns() {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                return ((DenseLargeDoubleMatrix2D) content).getFftColumns();
            } else {
                return ((DenseLargeDoubleMatrix2D) copy()).getFftColumns();
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Returns new complex matrix which is the discrete Fourier transform (DFT)
     * of each row of this matrix.
     * 
     * @return the discrete Fourier transform (DFT) of each row of this matrix.
     */
    public DenseLargeDComplexMatrix2D getFftRows() {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                return ((DenseLargeDoubleMatrix2D) content).getFftRows();
            } else {
                return ((DenseLargeDoubleMatrix2D) copy()).getFftRows();
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Returns new complex matrix which is the inverse of the discrete Fourier
     * transform (IDFT) of each column of this matrix.
     * 
     * @return the inverse of the discrete Fourier transform (IDFT) of each
     *         column of this matrix.
     */
    public DenseLargeDComplexMatrix2D getIfftColumns(final boolean scale) {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                return ((DenseLargeDoubleMatrix2D) content).getIfftColumns(scale);
            } else {
                return ((DenseLargeDoubleMatrix2D) copy()).getIfftColumns(scale);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Returns new complex matrix which is the inverse of the discrete Fourier
     * transform (IDFT) of each row of this matrix.
     * 
     * @return the inverse of the discrete Fourier transform (IDFT) of each row
     *         of this matrix.
     */
    public DenseLargeDComplexMatrix2D getIfftRows(final boolean scale) {
        if (this.isNoView == true) {
            return ((DenseLargeDoubleMatrix2D) content).getIfftRows(scale);
        } else {
            return ((DenseLargeDoubleMatrix2D) copy()).getIfftRows(scale);
        }

    }

    /**
     * Computes the 2D inverse of the discrete cosine transform (DCT-III) of
     * this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void idct2(boolean scale) {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).idct2(scale);
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.idct2(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the inverse of the discrete cosine transform (DCT-III) of each
     * column of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void idctColumns(final boolean scale) {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).idctColumns(scale);
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.idctColumns(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the inverse of the discrete cosine transform (DCT-III) of each
     * row of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void idctRows(final boolean scale) {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).idctRows(scale);
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.idctRows(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 2D inverse of the discrete size transform (DST-III) of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void idst2(boolean scale) {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).idst2(scale);
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.idst2(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the inverse of the discrete sine transform (DST-III) of each
     * column of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void idstColumns(final boolean scale) {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).idstColumns(scale);
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.idstColumns(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the inverse of the discrete sine transform (DST-III) of each row
     * of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void idstRows(final boolean scale) {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).idstRows(scale);
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.idstRows(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 2D inverse of the discrete Hartley transform (DHT) of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void idht2(boolean scale) {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).idht2(scale);
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.idht2(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the inverse of the discrete Hartley transform (DHT) of each
     * column of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void idhtColumns(final boolean scale) {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).idhtColumns(scale);
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.idhtColumns(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the inverse of the discrete Hartley transform (DHT) of each row
     * of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void idhtRows(final boolean scale) {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).idhtRows(scale);
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.idhtRows(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 2D inverse of the discrete Fourier transform (IDFT) of this
     * matrix. The physical layout of the input data has to be as follows:
     * 
     * <pre>
     * this[k1][2*k2] = Re[k1][k2] = Re[rows-k1][columns-k2], 
     * this[k1][2*k2+1] = Im[k1][k2] = -Im[rows-k1][columns-k2], 
     *       0&lt;k1&lt;rows, 0&lt;k2&lt;columns/2, 
     * this[0][2*k2] = Re[0][k2] = Re[0][columns-k2], 
     * this[0][2*k2+1] = Im[0][k2] = -Im[0][columns-k2], 
     *       0&lt;k2&lt;columns/2, 
     * this[k1][0] = Re[k1][0] = Re[rows-k1][0], 
     * this[k1][1] = Im[k1][0] = -Im[rows-k1][0], 
     * this[rows-k1][1] = Re[k1][columns/2] = Re[rows-k1][columns/2], 
     * this[rows-k1][0] = -Im[k1][columns/2] = Im[rows-k1][columns/2], 
     *       0&lt;k1&lt;rows/2, 
     * this[0][0] = Re[0][0], 
     * this[0][1] = Re[0][columns/2], 
     * this[rows/2][0] = Re[rows/2][0], 
     * this[rows/2][1] = Re[rows/2][columns/2]
     * </pre>
     * 
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * inverse transform, use <code>getIfft2</code>.
     * 
     * @throws IllegalArgumentException
     *             if the row size or the column size of this matrix is not a
     *             power of 2 number.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void ifft2(boolean scale) {
        if (content instanceof DenseLargeDoubleMatrix2D) {
            if (this.isNoView == true) {
                ((DenseLargeDoubleMatrix2D) content).ifft2(scale);
            } else {
                DenseLargeDoubleMatrix2D copy = (DenseLargeDoubleMatrix2D) copy();
                copy.ifft2(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    public synchronized void setQuick(int row, int column, double value) {
        content.setQuick(row, column, value);
    }

    public DoubleMatrix1D vectorize() {
        final DenseDoubleMatrix1D v = new DenseDoubleMatrix1D((int) size());
        int nthreads = ConcurrencyUtils.getNumberOfThreads();
        if ((nthreads > 1) && (size() >= ConcurrencyUtils.getThreadsBeginN_2D())) {
            nthreads = Math.min(nthreads, columns);
            Future<?>[] futures = new Future[nthreads];
            int k = columns / nthreads;
            for (int j = 0; j < nthreads; j++) {
                final int firstCol = j * k;
                final int lastCol = (j == nthreads - 1) ? columns : firstCol + k;
                final int firstidx = j * k * rows;
                futures[j] = ConcurrencyUtils.submit(new Runnable() {
                    public void run() {
                        int idx = firstidx;
                        for (int c = firstCol; c < lastCol; c++) {
                            for (int r = 0; r < rows; r++) {
                                v.setQuick(idx++, getQuick(r, c));
                            }
                        }
                    }
                });
            }
            ConcurrencyUtils.waitForCompletion(futures);
        } else {
            int idx = 0;
            for (int c = 0; c < columns; c++) {
                for (int r = 0; r < rows; r++) {
                    v.setQuick(idx++, getQuick(r, c));
                }
            }
        }
        return v;
    }

    public DoubleMatrix1D viewColumn(int column) {
        return viewDice().viewRow(column);
    }

    public DoubleMatrix2D viewColumnFlip() {
        if (columns == 0)
            return this;
        WrapperDoubleMatrix2D view = new WrapperDoubleMatrix2D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized double getQuick(int row, int column) {
                return content.getQuick(row, columns - 1 - column);
            }

            public synchronized void setQuick(int row, int column, double value) {
                content.setQuick(row, columns - 1 - column, value);
            }

            public synchronized double get(int row, int column) {
                return content.get(row, columns - 1 - column);
            }

            public synchronized void set(int row, int column, double value) {
                content.set(row, columns - 1 - column, value);
            }
        };
        view.isNoView = false;

        return view;
    }

    public DoubleMatrix2D viewDice() {
        WrapperDoubleMatrix2D view = new WrapperDoubleMatrix2D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized double getQuick(int row, int column) {
                return content.getQuick(column, row);
            }

            public synchronized void setQuick(int row, int column, double value) {
                content.setQuick(column, row, value);
            }

            public synchronized double get(int row, int column) {
                return content.get(column, row);
            }

            public synchronized void set(int row, int column, double value) {
                content.set(column, row, value);
            }
        };
        view.rows = columns;
        view.columns = rows;
        view.isNoView = false;

        return view;
    }

    public DoubleMatrix2D viewPart(final int row, final int column, int height, int width) {
        checkBox(row, column, height, width);
        WrapperDoubleMatrix2D view = new WrapperDoubleMatrix2D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized double getQuick(int i, int j) {
                return content.getQuick(row + i, column + j);
            }

            public synchronized void setQuick(int i, int j, double value) {
                content.setQuick(row + i, column + j, value);
            }

            public synchronized double get(int i, int j) {
                return content.get(row + i, column + j);
            }

            public synchronized void set(int i, int j, double value) {
                content.set(row + i, column + j, value);
            }
        };
        view.rows = height;
        view.columns = width;
        view.isNoView = false;

        return view;
    }

    public DoubleMatrix1D viewRow(int row) {
        checkRow(row);
        return new DelegateDoubleMatrix1D(this, row);
    }

    public DoubleMatrix2D viewRowFlip() {
        if (rows == 0)
            return this;
        WrapperDoubleMatrix2D view = new WrapperDoubleMatrix2D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized double getQuick(int row, int column) {
                return content.getQuick(rows - 1 - row, column);
            }

            public synchronized void setQuick(int row, int column, double value) {
                content.setQuick(rows - 1 - row, column, value);
            }

            public synchronized double get(int row, int column) {
                return content.get(rows - 1 - row, column);
            }

            public synchronized void set(int row, int column, double value) {
                content.set(rows - 1 - row, column, value);
            }
        };
        view.isNoView = false;
        return view;
    }

    public DoubleMatrix2D viewSelection(int[] rowIndexes, int[] columnIndexes) {
        // check for "all"
        if (rowIndexes == null) {
            rowIndexes = new int[rows];
            for (int i = rows; --i >= 0;)
                rowIndexes[i] = i;
        }
        if (columnIndexes == null) {
            columnIndexes = new int[columns];
            for (int i = columns; --i >= 0;)
                columnIndexes[i] = i;
        }

        checkRowIndexes(rowIndexes);
        checkColumnIndexes(columnIndexes);
        final int[] rix = rowIndexes;
        final int[] cix = columnIndexes;

        WrapperDoubleMatrix2D view = new WrapperDoubleMatrix2D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized double getQuick(int i, int j) {
                return content.getQuick(rix[i], cix[j]);
            }

            public synchronized void setQuick(int i, int j, double value) {
                content.setQuick(rix[i], cix[j], value);
            }

            public synchronized double get(int i, int j) {
                return content.get(rix[i], cix[j]);
            }

            public synchronized void set(int i, int j, double value) {
                content.set(rix[i], cix[j], value);
            }
        };
        view.rows = rowIndexes.length;
        view.columns = columnIndexes.length;
        view.isNoView = false;

        return view;
    }

    public DoubleMatrix2D viewStrides(final int _rowStride, final int _columnStride) {
        if (_rowStride <= 0 || _columnStride <= 0)
            throw new IndexOutOfBoundsException("illegal stride");
        WrapperDoubleMatrix2D view = new WrapperDoubleMatrix2D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized double getQuick(int row, int column) {
                return content.getQuick(_rowStride * row, _columnStride * column);
            }

            public synchronized void setQuick(int row, int column, double value) {
                content.setQuick(_rowStride * row, _columnStride * column, value);
            }

            public synchronized double get(int row, int column) {
                return content.get(_rowStride * row, _columnStride * column);
            }

            public synchronized void set(int row, int column, double value) {
                content.set(_rowStride * row, _columnStride * column, value);
            }
        };
        if (rows != 0)
            view.rows = (rows - 1) / _rowStride + 1;
        if (columns != 0)
            view.columns = (columns - 1) / _columnStride + 1;
        view.isNoView = false;

        return view;
    }

    protected DoubleMatrix2D getContent() {
        return content;
    }

    protected DoubleMatrix1D like1D(int size, int offset, int stride) {
        throw new InternalError(); // should never get called
    }

    protected DoubleMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        throw new InternalError(); // should never be called
    }
}
