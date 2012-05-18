/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tfloat.impl;

import cern.colt.matrix.tfcomplex.impl.DenseLargeFComplexMatrix3D;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.FloatMatrix3D;

/**
 * 3-d matrix holding <tt>float</tt> elements; either a view wrapping another
 * matrix or a matrix whose views are wrappers.
 * 
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class WrapperFloatMatrix3D extends FloatMatrix3D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;
    /*
     * The elements of the matrix.
     */
    protected FloatMatrix3D content;

    public WrapperFloatMatrix3D(FloatMatrix3D newContent) {
        if (newContent != null)
            try {
                setUp(newContent.slices(), newContent.rows(), newContent.columns());
            } catch (IllegalArgumentException exc) { // we can hold slices*rows*columns>Integer.MAX_VALUE cells !
                if (!"matrix too large".equals(exc.getMessage()))
                    throw exc;
            }
        this.content = newContent;
    }

    public Object elements() {
        return content.elements();
    }

    /**
     * Computes the 3D discrete cosine transform (DCT-II) of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void dct3(boolean scale) {
        if (content instanceof DenseLargeFloatMatrix3D) {
            if (this.isNoView == true) {
                ((DenseLargeFloatMatrix3D) content).dct3(scale);
            } else {
                DenseLargeFloatMatrix3D copy = (DenseLargeFloatMatrix3D) copy();
                copy.dct3(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 2D discrete cosine transform (DCT-II) of each slice of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void dct2Slices(final boolean scale) {
        if (content instanceof DenseLargeFloatMatrix3D) {
            if (this.isNoView == true) {
                ((DenseLargeFloatMatrix3D) content).dct2Slices(scale);
            } else {
                DenseLargeFloatMatrix3D copy = (DenseLargeFloatMatrix3D) copy();
                copy.dct2Slices(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 3D discrete sine transform (DST-II) of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void dst3(boolean scale) {
        if (content instanceof DenseLargeFloatMatrix3D) {
            if (this.isNoView == true) {
                ((DenseLargeFloatMatrix3D) content).dst3(scale);
            } else {
                DenseLargeFloatMatrix3D copy = (DenseLargeFloatMatrix3D) copy();
                copy.dst3(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 2D discrete sine transform (DST-II) of each slice of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     */
    public void dst2Slices(final boolean scale) {
        if (content instanceof DenseLargeFloatMatrix3D) {
            if (this.isNoView == true) {
                ((DenseLargeFloatMatrix3D) content).dst2Slices(scale);
            } else {
                DenseLargeFloatMatrix3D copy = (DenseLargeFloatMatrix3D) copy();
                copy.dst2Slices(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 3D discrete Hartley transform (DHT) of this matrix.
     */
    public void dht3() {
        if (content instanceof DenseLargeFloatMatrix3D) {
            if (this.isNoView == true) {
                ((DenseLargeFloatMatrix3D) content).dht3();
            } else {
                DenseLargeFloatMatrix3D copy = (DenseLargeFloatMatrix3D) copy();
                copy.dht3();
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 2D discrete Hertley transform (DHT) of each column of this
     * matrix.
     */
    public void dht2Slices() {
        if (content instanceof DenseLargeFloatMatrix3D) {
            if (this.isNoView == true) {
                ((DenseLargeFloatMatrix3D) content).dht2Slices();
            } else {
                DenseLargeFloatMatrix3D copy = (DenseLargeFloatMatrix3D) copy();
                copy.dht2Slices();
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 3D discrete Fourier transform (DFT) of this matrix. The
     * physical layout of the output data is as follows:
     * 
     * <pre>
     * this[k1][k2][2*k3] = Re[k1][k2][k3]
     *                 = Re[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     * this[k1][k2][2*k3+1] = Im[k1][k2][k3]
     *                   = -Im[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     *     0&lt;=k1&lt;n1, 0&lt;=k2&lt;n2, 0&lt;k3&lt;n3/2, 
     * this[k1][k2][0] = Re[k1][k2][0]
     *              = Re[(n1-k1)%n1][n2-k2][0], 
     * this[k1][k2][1] = Im[k1][k2][0]
     *              = -Im[(n1-k1)%n1][n2-k2][0], 
     * this[k1][n2-k2][1] = Re[(n1-k1)%n1][k2][n3/2]
     *                 = Re[k1][n2-k2][n3/2], 
     * this[k1][n2-k2][0] = -Im[(n1-k1)%n1][k2][n3/2]
     *                 = Im[k1][n2-k2][n3/2], 
     *     0&lt;=k1&lt;n1, 0&lt;k2&lt;n2/2, 
     * this[k1][0][0] = Re[k1][0][0]
     *             = Re[n1-k1][0][0], 
     * this[k1][0][1] = Im[k1][0][0]
     *             = -Im[n1-k1][0][0], 
     * this[k1][n2/2][0] = Re[k1][n2/2][0]
     *                = Re[n1-k1][n2/2][0], 
     * this[k1][n2/2][1] = Im[k1][n2/2][0]
     *                = -Im[n1-k1][n2/2][0], 
     * this[n1-k1][0][1] = Re[k1][0][n3/2]
     *                = Re[n1-k1][0][n3/2], 
     * this[n1-k1][0][0] = -Im[k1][0][n3/2]
     *                = Im[n1-k1][0][n3/2], 
     * this[n1-k1][n2/2][1] = Re[k1][n2/2][n3/2]
     *                   = Re[n1-k1][n2/2][n3/2], 
     * this[n1-k1][n2/2][0] = -Im[k1][n2/2][n3/2]
     *                   = Im[n1-k1][n2/2][n3/2], 
     *     0&lt;k1&lt;n1/2, 
     * this[0][0][0] = Re[0][0][0], 
     * this[0][0][1] = Re[0][0][n3/2], 
     * this[0][n2/2][0] = Re[0][n2/2][0], 
     * this[0][n2/2][1] = Re[0][n2/2][n3/2], 
     * this[n1/2][0][0] = Re[n1/2][0][0], 
     * this[n1/2][0][1] = Re[n1/2][0][n3/2], 
     * this[n1/2][n2/2][0] = Re[n1/2][n2/2][0], 
     * this[n1/2][n2/2][1] = Re[n1/2][n2/2][n3/2]
     * </pre>
     * 
     * 
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * forward transform, use <code>getFft3</code>. To get back the original
     * data, use <code>ifft3</code>.
     * 
     * @throws IllegalArgumentException
     *             if the slice size or the row size or the column size of this
     *             matrix is not a power of 2 number.
     */
    public void fft3() {
        if (content instanceof DenseLargeFloatMatrix3D) {
            if (this.isNoView == true) {
                ((DenseLargeFloatMatrix3D) content).fft3();
            } else {
                DenseLargeFloatMatrix3D copy = (DenseLargeFloatMatrix3D) copy();
                copy.fft3();
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Returns new complex matrix which is the 3D discrete Fourier transform
     * (DFT) of this matrix.
     * 
     * @return the 3D discrete Fourier transform (DFT) of this matrix.
     * 
     */
    public DenseLargeFComplexMatrix3D getFft3() {
        if (content instanceof DenseLargeFloatMatrix3D) {
            if (this.isNoView == true) {
                return ((DenseLargeFloatMatrix3D) content).getFft3();
            } else {
                return ((DenseLargeFloatMatrix3D) copy()).getFft3();
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Returns new complex matrix which is the 3D inverse of the discrete
     * Fourier transform (IDFT) of this matrix.
     * 
     * @return the 3D inverse of the discrete Fourier transform (IDFT) of this
     *         matrix.
     */
    public DenseLargeFComplexMatrix3D getIfft3(boolean scale) {
        if (content instanceof DenseLargeFloatMatrix3D) {
            if (this.isNoView == true) {
                return ((DenseLargeFloatMatrix3D) content).getIfft3(scale);
            } else {
                return ((DenseLargeFloatMatrix3D) copy()).getIfft3(scale);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Returns new complex matrix which is the 2D discrete Fourier transform
     * (DFT) of each slice of this matrix.
     * 
     * @return the 2D discrete Fourier transform (DFT) of each slice of this
     *         matrix.
     */
    public DenseLargeFComplexMatrix3D getFft2Slices() {
        if (content instanceof DenseLargeFloatMatrix3D) {
            if (this.isNoView == true) {
                return ((DenseLargeFloatMatrix3D) content).getFft2Slices();
            } else {
                return ((DenseLargeFloatMatrix3D) copy()).getFft2Slices();
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Returns new complex matrix which is the 2D inverse of the discrete
     * Fourier transform (IDFT) of each slice of this matrix.
     * 
     * @return the 2D inverse of the discrete Fourier transform (IDFT) of each
     *         slice of this matrix.
     */
    public DenseLargeFComplexMatrix3D getIfft2Slices(final boolean scale) {
        if (content instanceof DenseLargeFloatMatrix3D) {
            if (this.isNoView == true) {
                return ((DenseLargeFloatMatrix3D) content).getIfft2Slices(scale);
            } else {
                return ((DenseLargeFloatMatrix3D) copy()).getIfft2Slices(scale);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 3D inverse of the discrete cosine transform (DCT-III) of
     * this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void idct3(boolean scale) {
        if (content instanceof DenseLargeFloatMatrix3D) {
            if (this.isNoView == true) {
                ((DenseLargeFloatMatrix3D) content).idct3(scale);
            } else {
                DenseLargeFloatMatrix3D copy = (DenseLargeFloatMatrix3D) copy();
                copy.idct3(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 2D inverse of the discrete cosine transform (DCT-III) of
     * each slice of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void idct2Slices(final boolean scale) {
        if (content instanceof DenseLargeFloatMatrix3D) {
            if (this.isNoView == true) {
                ((DenseLargeFloatMatrix3D) content).idct2Slices(scale);
            } else {
                DenseLargeFloatMatrix3D copy = (DenseLargeFloatMatrix3D) copy();
                copy.idct2Slices(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 3D inverse of the discrete size transform (DST-III) of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void idst3(boolean scale) {
        if (content instanceof DenseLargeFloatMatrix3D) {
            if (this.isNoView == true) {
                ((DenseLargeFloatMatrix3D) content).idst3(scale);
            } else {
                DenseLargeFloatMatrix3D copy = (DenseLargeFloatMatrix3D) copy();
                copy.idst3(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 2D inverse of the discrete sine transform (DST-III) of each
     * slice of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void idst2Slices(final boolean scale) {
        if (content instanceof DenseLargeFloatMatrix3D) {
            if (this.isNoView == true) {
                ((DenseLargeFloatMatrix3D) content).idst2Slices(scale);
            } else {
                DenseLargeFloatMatrix3D copy = (DenseLargeFloatMatrix3D) copy();
                copy.idst2Slices(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 3D inverse of the discrete Hartley transform (DHT) of this
     * matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void idht3(boolean scale) {
        if (content instanceof DenseLargeFloatMatrix3D) {
            if (this.isNoView == true) {
                ((DenseLargeFloatMatrix3D) content).idht3(scale);
            } else {
                DenseLargeFloatMatrix3D copy = (DenseLargeFloatMatrix3D) copy();
                copy.idht3(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 2D inverse of the discrete Hartley transform (DHT) of each
     * slice of this matrix.
     * 
     * @param scale
     *            if true then scaling is performed
     */
    public void idht2Slices(final boolean scale) {
        if (content instanceof DenseLargeFloatMatrix3D) {
            if (this.isNoView == true) {
                ((DenseLargeFloatMatrix3D) content).idht2Slices(scale);
            } else {
                DenseLargeFloatMatrix3D copy = (DenseLargeFloatMatrix3D) copy();
                copy.idht2Slices(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    /**
     * Computes the 3D inverse of the discrete Fourier transform (IDFT) of this
     * matrix. The physical layout of the input data has to be as follows:
     * 
     * <pre>
     * this[k1][k2][2*k3] = Re[k1][k2][k3]
     *                 = Re[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     * this[k1][k2][2*k3+1] = Im[k1][k2][k3]
     *                   = -Im[(n1-k1)%n1][(n2-k2)%n2][n3-k3], 
     *     0&lt;=k1&lt;n1, 0&lt;=k2&lt;n2, 0&lt;k3&lt;n3/2, 
     * this[k1][k2][0] = Re[k1][k2][0]
     *              = Re[(n1-k1)%n1][n2-k2][0], 
     * this[k1][k2][1] = Im[k1][k2][0]
     *              = -Im[(n1-k1)%n1][n2-k2][0], 
     * this[k1][n2-k2][1] = Re[(n1-k1)%n1][k2][n3/2]
     *                 = Re[k1][n2-k2][n3/2], 
     * this[k1][n2-k2][0] = -Im[(n1-k1)%n1][k2][n3/2]
     *                 = Im[k1][n2-k2][n3/2], 
     *     0&lt;=k1&lt;n1, 0&lt;k2&lt;n2/2, 
     * this[k1][0][0] = Re[k1][0][0]
     *             = Re[n1-k1][0][0], 
     * this[k1][0][1] = Im[k1][0][0]
     *             = -Im[n1-k1][0][0], 
     * this[k1][n2/2][0] = Re[k1][n2/2][0]
     *                = Re[n1-k1][n2/2][0], 
     * this[k1][n2/2][1] = Im[k1][n2/2][0]
     *                = -Im[n1-k1][n2/2][0], 
     * this[n1-k1][0][1] = Re[k1][0][n3/2]
     *                = Re[n1-k1][0][n3/2], 
     * this[n1-k1][0][0] = -Im[k1][0][n3/2]
     *                = Im[n1-k1][0][n3/2], 
     * this[n1-k1][n2/2][1] = Re[k1][n2/2][n3/2]
     *                   = Re[n1-k1][n2/2][n3/2], 
     * this[n1-k1][n2/2][0] = -Im[k1][n2/2][n3/2]
     *                   = Im[n1-k1][n2/2][n3/2], 
     *     0&lt;k1&lt;n1/2, 
     * this[0][0][0] = Re[0][0][0], 
     * this[0][0][1] = Re[0][0][n3/2], 
     * this[0][n2/2][0] = Re[0][n2/2][0], 
     * this[0][n2/2][1] = Re[0][n2/2][n3/2], 
     * this[n1/2][0][0] = Re[n1/2][0][0], 
     * this[n1/2][0][1] = Re[n1/2][0][n3/2], 
     * this[n1/2][n2/2][0] = Re[n1/2][n2/2][0], 
     * this[n1/2][n2/2][1] = Re[n1/2][n2/2][n3/2]
     * </pre>
     * 
     * This method computes only half of the elements of the real transform. The
     * other half satisfies the symmetry condition. If you want the full real
     * inverse transform, use <code>getIfft3</code>.
     * 
     * @param scale
     *            if true then scaling is performed
     * 
     * @throws IllegalArgumentException
     *             if the slice size or the row size or the column size of this
     *             matrix is not a power of 2 number.
     */
    public void ifft3(boolean scale) {
        if (content instanceof DenseLargeFloatMatrix3D) {
            if (this.isNoView == true) {
                ((DenseLargeFloatMatrix3D) content).ifft3(scale);
            } else {
                DenseLargeFloatMatrix3D copy = (DenseLargeFloatMatrix3D) copy();
                copy.ifft3(scale);
                assign(copy);
            }
        } else {
            throw new IllegalArgumentException("This method is not supported");
        }
    }

    public synchronized float getQuick(int slice, int row, int column) {
        return content.getQuick(slice, row, column);
    }

    public FloatMatrix3D like(int slices, int rows, int columns) {
        return content.like(slices, rows, columns);
    }

    public synchronized void setQuick(int slice, int row, int column, float value) {
        content.setQuick(slice, row, column, value);
    }

    public FloatMatrix1D vectorize() {
        FloatMatrix1D v = new DenseFloatMatrix1D((int) size());
        int length = rows * columns;
        for (int s = 0; s < slices; s++) {
            v.viewPart(s * length, length).assign(viewSlice(s).vectorize());
        }
        return v;
    }

    public FloatMatrix2D viewColumn(int column) {
        checkColumn(column);
        return new DelegateFloatMatrix2D(this, 2, column);
    }

    public FloatMatrix3D viewColumnFlip() {
        if (columns == 0)
            return this;
        WrapperFloatMatrix3D view = new WrapperFloatMatrix3D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized float getQuick(int slice, int row, int column) {
                return content.getQuick(slice, row, columns - 1 - column);
            }

            public synchronized void setQuick(int slice, int row, int column, float value) {
                content.setQuick(slice, row, columns - 1 - column, value);
            }

            public synchronized float get(int slice, int row, int column) {
                return content.get(slice, row, columns - 1 - column);
            }

            public synchronized void set(int slice, int row, int column, float value) {
                content.set(slice, row, columns - 1 - column, value);
            }
        };
        view.isNoView = false;
        return view;
    }

    public FloatMatrix2D viewSlice(int slice) {
        checkSlice(slice);
        return new DelegateFloatMatrix2D(this, 0, slice);
    }

    public FloatMatrix3D viewSliceFlip() {
        if (slices == 0)
            return this;
        WrapperFloatMatrix3D view = new WrapperFloatMatrix3D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized float getQuick(int slice, int row, int column) {
                return content.getQuick(slices - 1 - slice, row, column);
            }

            public synchronized void setQuick(int slice, int row, int column, float value) {
                content.setQuick(slices - 1 - slice, row, column, value);
            }

            public synchronized float get(int slice, int row, int column) {
                return content.get(slices - 1 - slice, row, column);
            }

            public synchronized void set(int slice, int row, int column, float value) {
                content.set(slices - 1 - slice, row, column, value);
            }
        };
        view.isNoView = false;
        return view;
    }

    public FloatMatrix3D viewDice(int axis0, int axis1, int axis2) {
        int d = 3;
        if (axis0 < 0 || axis0 >= d || axis1 < 0 || axis1 >= d || axis2 < 0 || axis2 >= d || axis0 == axis1
                || axis0 == axis2 || axis1 == axis2) {
            throw new IllegalArgumentException("Illegal Axes: " + axis0 + ", " + axis1 + ", " + axis2);
        }
        WrapperFloatMatrix3D view = null;
        if (axis0 == 0 && axis1 == 1 && axis2 == 2) {
            view = new WrapperFloatMatrix3D(this);
        } else if (axis0 == 1 && axis1 == 0 && axis2 == 2) {
            view = new WrapperFloatMatrix3D(this) {
                /**
                 * 
                 */
                private static final long serialVersionUID = 1L;

                public synchronized float getQuick(int slice, int row, int column) {
                    return content.getQuick(row, slice, column);
                }

                public synchronized void setQuick(int slice, int row, int column, float value) {
                    content.setQuick(row, slice, column, value);
                }

                public synchronized float get(int slice, int row, int column) {
                    return content.get(row, slice, column);
                }

                public synchronized void set(int slice, int row, int column, float value) {
                    content.set(row, slice, column, value);
                }
            };
        } else if (axis0 == 1 && axis1 == 2 && axis2 == 0) {
            view = new WrapperFloatMatrix3D(this) {
                /**
                 * 
                 */
                private static final long serialVersionUID = 1L;

                public synchronized float getQuick(int slice, int row, int column) {
                    return content.getQuick(row, column, slice);
                }

                public synchronized void setQuick(int slice, int row, int column, float value) {
                    content.setQuick(row, column, slice, value);
                }

                public synchronized float get(int slice, int row, int column) {
                    return content.get(row, column, slice);
                }

                public synchronized void set(int slice, int row, int column, float value) {
                    content.set(row, column, slice, value);
                }
            };
        } else if (axis0 == 2 && axis1 == 1 && axis2 == 0) {
            view = new WrapperFloatMatrix3D(this) {
                /**
                 * 
                 */
                private static final long serialVersionUID = 1L;

                public synchronized float getQuick(int slice, int row, int column) {
                    return content.getQuick(column, row, slice);
                }

                public synchronized void setQuick(int slice, int row, int column, float value) {
                    content.setQuick(column, row, slice, value);
                }

                public synchronized float get(int slice, int row, int column) {
                    return content.get(column, row, slice);
                }

                public synchronized void set(int slice, int row, int column, float value) {
                    content.set(column, row, slice, value);
                }
            };
        } else if (axis0 == 2 && axis1 == 0 && axis2 == 1) {
            view = new WrapperFloatMatrix3D(this) {
                /**
                 * 
                 */
                private static final long serialVersionUID = 1L;

                public synchronized float getQuick(int slice, int row, int column) {
                    return content.getQuick(column, slice, row);
                }

                public synchronized void setQuick(int slice, int row, int column, float value) {
                    content.setQuick(column, slice, row, value);
                }

                public synchronized float get(int slice, int row, int column) {
                    return content.get(column, slice, row);
                }

                public synchronized void set(int slice, int row, int column, float value) {
                    content.set(column, slice, row, value);
                }
            };
        }
        int[] shape = shape();
        view.slices = shape[axis0];
        view.rows = shape[axis1];
        view.columns = shape[axis2];
        view.isNoView = false;
        return view;
    }

    public FloatMatrix3D viewPart(final int slice, final int row, final int column, int depth, int height, int width) {
        checkBox(slice, row, column, depth, height, width);
        WrapperFloatMatrix3D view = new WrapperFloatMatrix3D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized float getQuick(int i, int j, int k) {
                return content.getQuick(slice + i, row + j, column + k);
            }

            public synchronized void setQuick(int i, int j, int k, float value) {
                content.setQuick(slice + i, row + j, column + k, value);
            }

            public synchronized float get(int i, int j, int k) {
                return content.get(slice + i, row + j, column + k);
            }

            public synchronized void set(int i, int j, int k, float value) {
                content.set(slice + i, row + j, column + k, value);
            }
        };
        view.slices = depth;
        view.rows = height;
        view.columns = width;
        view.isNoView = false;
        return view;
    }

    public FloatMatrix2D viewRow(int row) {
        checkRow(row);
        return new DelegateFloatMatrix2D(this, 1, row);
    }

    public FloatMatrix3D viewRowFlip() {
        if (rows == 0)
            return this;
        WrapperFloatMatrix3D view = new WrapperFloatMatrix3D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized float getQuick(int slice, int row, int column) {
                return content.getQuick(slice, rows - 1 - row, column);
            }

            public synchronized void setQuick(int slice, int row, int column, float value) {
                content.setQuick(slice, rows - 1 - row, column, value);
            }

            public synchronized float get(int slice, int row, int column) {
                return content.get(slice, rows - 1 - row, column);
            }

            public synchronized void set(int slice, int row, int column, float value) {
                content.set(slice, rows - 1 - row, column, value);
            }
        };
        view.isNoView = false;
        return view;
    }

    public FloatMatrix3D viewSelection(int[] sliceIndexes, int[] rowIndexes, int[] columnIndexes) {
        // check for "all"
        if (sliceIndexes == null) {
            sliceIndexes = new int[slices];
            for (int i = slices; --i >= 0;)
                sliceIndexes[i] = i;
        }
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

        checkSliceIndexes(sliceIndexes);
        checkRowIndexes(rowIndexes);
        checkColumnIndexes(columnIndexes);
        final int[] six = sliceIndexes;
        final int[] rix = rowIndexes;
        final int[] cix = columnIndexes;

        WrapperFloatMatrix3D view = new WrapperFloatMatrix3D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized float getQuick(int i, int j, int k) {
                return content.getQuick(six[i], rix[j], cix[k]);
            }

            public synchronized void setQuick(int i, int j, int k, float value) {
                content.setQuick(six[i], rix[j], cix[k], value);
            }

            public synchronized float get(int i, int j, int k) {
                return content.get(six[i], rix[j], cix[k]);
            }

            public synchronized void set(int i, int j, int k, float value) {
                content.set(six[i], rix[j], cix[k], value);
            }
        };
        view.slices = sliceIndexes.length;
        view.rows = rowIndexes.length;
        view.columns = columnIndexes.length;
        view.isNoView = false;
        return view;
    }

    public FloatMatrix3D viewStrides(final int _sliceStride, final int _rowStride, final int _columnStride) {
        if (_sliceStride <= 0 || _rowStride <= 0 || _columnStride <= 0)
            throw new IndexOutOfBoundsException("illegal stride");
        WrapperFloatMatrix3D view = new WrapperFloatMatrix3D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized float getQuick(int slice, int row, int column) {
                return content.getQuick(_sliceStride * slice, _rowStride * row, _columnStride * column);
            }

            public synchronized void setQuick(int slice, int row, int column, float value) {
                content.setQuick(_sliceStride * slice, _rowStride * row, _columnStride * column, value);
            }

            public synchronized float get(int slice, int row, int column) {
                return content.get(_sliceStride * slice, _rowStride * row, _columnStride * column);
            }

            public synchronized void set(int slice, int row, int column, float value) {
                content.set(_sliceStride * slice, _rowStride * row, _columnStride * column, value);
            }
        };
        if (slices != 0)
            view.slices = (slices - 1) / _sliceStride + 1;
        if (rows != 0)
            view.rows = (rows - 1) / _rowStride + 1;
        if (columns != 0)
            view.columns = (columns - 1) / _columnStride + 1;
        view.isNoView = false;
        return view;
    }

    protected FloatMatrix3D getContent() {
        return content;
    }

    public FloatMatrix2D like2D(int rows, int columns) {
        throw new InternalError(); // should never get called
    }

    protected FloatMatrix2D like2D(int rows, int columns, int rowZero, int columnZero, int rowStride, int columnStride) {
        throw new InternalError(); // should never get called
    }

    protected FloatMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets) {
        throw new InternalError(); // should never get called
    }
}
