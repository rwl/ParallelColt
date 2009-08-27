/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tlong.impl;

import cern.colt.matrix.tlong.LongMatrix1D;
import cern.colt.matrix.tlong.LongMatrix2D;
import cern.colt.matrix.tlong.LongMatrix3D;

/**
 * 3-d matrix holding <tt>long</tt> elements; either a view wrapping another
 * matrix or a matrix whose views are wrappers.
 * 
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class WrapperLongMatrix3D extends LongMatrix3D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;
    /*
     * The elements of the matrix.
     */
    protected LongMatrix3D content;

    public WrapperLongMatrix3D(LongMatrix3D newContent) {
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

    public synchronized long getQuick(int slice, int row, int column) {
        return content.getQuick(slice, row, column);
    }

    public LongMatrix3D like(int slices, int rows, int columns) {
        return content.like(slices, rows, columns);
    }

    public synchronized void setQuick(int slice, int row, int column, long value) {
        content.setQuick(slice, row, column, value);
    }

    public LongMatrix1D vectorize() {
        LongMatrix1D v = new DenseLongMatrix1D((int) size());
        int length = rows * columns;
        for (int s = 0; s < slices; s++) {
            v.viewPart(s * length, length).assign(viewSlice(s).vectorize());
        }
        return v;
    }

    public LongMatrix2D viewColumn(int column) {
        checkColumn(column);
        return new DelegateLongMatrix2D(this, 2, column);
    }

    public LongMatrix3D viewColumnFlip() {
        if (columns == 0)
            return this;
        WrapperLongMatrix3D view = new WrapperLongMatrix3D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized long getQuick(int slice, int row, int column) {
                return content.getQuick(slice, row, columns - 1 - column);
            }

            public synchronized void setQuick(int slice, int row, int column, long value) {
                content.setQuick(slice, row, columns - 1 - column, value);
            }

            public synchronized long get(int slice, int row, int column) {
                return content.get(slice, row, columns - 1 - column);
            }

            public synchronized void set(int slice, int row, int column, long value) {
                content.set(slice, row, columns - 1 - column, value);
            }
        };
        view.isNoView = false;
        return view;
    }

    public LongMatrix2D viewSlice(int slice) {
        checkSlice(slice);
        return new DelegateLongMatrix2D(this, 0, slice);
    }

    public LongMatrix3D viewSliceFlip() {
        if (slices == 0)
            return this;
        WrapperLongMatrix3D view = new WrapperLongMatrix3D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized long getQuick(int slice, int row, int column) {
                return content.getQuick(slices - 1 - slice, row, column);
            }

            public synchronized void setQuick(int slice, int row, int column, long value) {
                content.setQuick(slices - 1 - slice, row, column, value);
            }

            public synchronized long get(int slice, int row, int column) {
                return content.get(slices - 1 - slice, row, column);
            }

            public synchronized void set(int slice, int row, int column, long value) {
                content.set(slices - 1 - slice, row, column, value);
            }
        };
        view.isNoView = false;
        return view;
    }

    public LongMatrix3D viewDice(int axis0, int axis1, int axis2) {
        int d = 3;
        if (axis0 < 0 || axis0 >= d || axis1 < 0 || axis1 >= d || axis2 < 0 || axis2 >= d || axis0 == axis1
                || axis0 == axis2 || axis1 == axis2) {
            throw new IllegalArgumentException("Illegal Axes: " + axis0 + ", " + axis1 + ", " + axis2);
        }
        WrapperLongMatrix3D view = null;
        if (axis0 == 0 && axis1 == 1 && axis2 == 2) {
            view = new WrapperLongMatrix3D(this);
        } else if (axis0 == 1 && axis1 == 0 && axis2 == 2) {
            view = new WrapperLongMatrix3D(this) {
                /**
                 * 
                 */
                private static final long serialVersionUID = 1L;

                public synchronized long getQuick(int slice, int row, int column) {
                    return content.getQuick(row, slice, column);
                }

                public synchronized void setQuick(int slice, int row, int column, long value) {
                    content.setQuick(row, slice, column, value);
                }

                public synchronized long get(int slice, int row, int column) {
                    return content.get(row, slice, column);
                }

                public synchronized void set(int slice, int row, int column, long value) {
                    content.set(row, slice, column, value);
                }
            };
        } else if (axis0 == 1 && axis1 == 2 && axis2 == 0) {
            view = new WrapperLongMatrix3D(this) {
                /**
                 * 
                 */
                private static final long serialVersionUID = 1L;

                public synchronized long getQuick(int slice, int row, int column) {
                    return content.getQuick(row, column, slice);
                }

                public synchronized void setQuick(int slice, int row, int column, long value) {
                    content.setQuick(row, column, slice, value);
                }

                public synchronized long get(int slice, int row, int column) {
                    return content.get(row, column, slice);
                }

                public synchronized void set(int slice, int row, int column, long value) {
                    content.set(row, column, slice, value);
                }
            };
        } else if (axis0 == 2 && axis1 == 1 && axis2 == 0) {
            view = new WrapperLongMatrix3D(this) {
                /**
                 * 
                 */
                private static final long serialVersionUID = 1L;

                public synchronized long getQuick(int slice, int row, int column) {
                    return content.getQuick(column, row, slice);
                }

                public synchronized void setQuick(int slice, int row, int column, long value) {
                    content.setQuick(column, row, slice, value);
                }

                public synchronized long get(int slice, int row, int column) {
                    return content.get(column, row, slice);
                }

                public synchronized void set(int slice, int row, int column, long value) {
                    content.set(column, row, slice, value);
                }
            };
        } else if (axis0 == 2 && axis1 == 0 && axis2 == 1) {
            view = new WrapperLongMatrix3D(this) {
                /**
                 * 
                 */
                private static final long serialVersionUID = 1L;

                public synchronized long getQuick(int slice, int row, int column) {
                    return content.getQuick(column, slice, row);
                }

                public synchronized void setQuick(int slice, int row, int column, long value) {
                    content.setQuick(column, slice, row, value);
                }

                public synchronized long get(int slice, int row, int column) {
                    return content.get(column, slice, row);
                }

                public synchronized void set(int slice, int row, int column, long value) {
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

    public LongMatrix3D viewPart(final int slice, final int row, final int column, int depth, int height, int width) {
        checkBox(slice, row, column, depth, height, width);
        WrapperLongMatrix3D view = new WrapperLongMatrix3D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized long getQuick(int i, int j, int k) {
                return content.getQuick(slice + i, row + j, column + k);
            }

            public synchronized void setQuick(int i, int j, int k, long value) {
                content.setQuick(slice + i, row + j, column + k, value);
            }

            public synchronized long get(int i, int j, int k) {
                return content.get(slice + i, row + j, column + k);
            }

            public synchronized void set(int i, int j, int k, long value) {
                content.set(slice + i, row + j, column + k, value);
            }
        };
        view.slices = depth;
        view.rows = height;
        view.columns = width;
        view.isNoView = false;
        return view;
    }

    public LongMatrix2D viewRow(int row) {
        checkRow(row);
        return new DelegateLongMatrix2D(this, 1, row);
    }

    public LongMatrix3D viewRowFlip() {
        if (rows == 0)
            return this;
        WrapperLongMatrix3D view = new WrapperLongMatrix3D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized long getQuick(int slice, int row, int column) {
                return content.getQuick(slice, rows - 1 - row, column);
            }

            public synchronized void setQuick(int slice, int row, int column, long value) {
                content.setQuick(slice, rows - 1 - row, column, value);
            }

            public synchronized long get(int slice, int row, int column) {
                return content.get(slice, rows - 1 - row, column);
            }

            public synchronized void set(int slice, int row, int column, long value) {
                content.set(slice, rows - 1 - row, column, value);
            }
        };
        view.isNoView = false;
        return view;
    }

    public LongMatrix3D viewSelection(int[] sliceIndexes, int[] rowIndexes, int[] columnIndexes) {
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

        WrapperLongMatrix3D view = new WrapperLongMatrix3D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized long getQuick(int i, int j, int k) {
                return content.getQuick(six[i], rix[j], cix[k]);
            }

            public synchronized void setQuick(int i, int j, int k, long value) {
                content.setQuick(six[i], rix[j], cix[k], value);
            }

            public synchronized long get(int i, int j, int k) {
                return content.get(six[i], rix[j], cix[k]);
            }

            public synchronized void set(int i, int j, int k, long value) {
                content.set(six[i], rix[j], cix[k], value);
            }
        };
        view.slices = sliceIndexes.length;
        view.rows = rowIndexes.length;
        view.columns = columnIndexes.length;
        view.isNoView = false;
        return view;
    }

    public LongMatrix3D viewStrides(final int _sliceStride, final int _rowStride, final int _columnStride) {
        if (_sliceStride <= 0 || _rowStride <= 0 || _columnStride <= 0)
            throw new IndexOutOfBoundsException("illegal stride");
        WrapperLongMatrix3D view = new WrapperLongMatrix3D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized long getQuick(int slice, int row, int column) {
                return content.getQuick(_sliceStride * slice, _rowStride * row, _columnStride * column);
            }

            public synchronized void setQuick(int slice, int row, int column, long value) {
                content.setQuick(_sliceStride * slice, _rowStride * row, _columnStride * column, value);
            }

            public synchronized long get(int slice, int row, int column) {
                return content.get(_sliceStride * slice, _rowStride * row, _columnStride * column);
            }

            public synchronized void set(int slice, int row, int column, long value) {
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

    protected LongMatrix3D getContent() {
        return content;
    }

    public LongMatrix2D like2D(int rows, int columns) {
        throw new InternalError(); // should never get called
    }

    protected LongMatrix2D like2D(int rows, int columns, int rowZero, int columnZero, int rowStride, int columnStride) {
        throw new InternalError(); // should never get called
    }

    protected LongMatrix3D viewSelectionLike(int[] sliceOffsets, int[] rowOffsets, int[] columnOffsets) {
        throw new InternalError(); // should never get called
    }
}
