/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.colt.matrix.tint.impl;

import java.util.concurrent.Future;

import cern.colt.list.tint.IntArrayList;
import cern.colt.matrix.tint.IntMatrix1D;
import cern.colt.matrix.tint.IntMatrix2D;
import edu.emory.mathcs.utils.ConcurrencyUtils;

/**
 * 2-d matrix holding <tt>int</tt> elements; either a view wrapping another
 * matrix or a matrix whose views are wrappers.
 * 
 * @author wolfgang.hoschek@cern.ch
 * @version 1.0, 04/14/2000
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class WrapperIntMatrix2D extends IntMatrix2D {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;
    /*
     * The elements of the matrix.
     */
    protected IntMatrix2D content;

    public WrapperIntMatrix2D(IntMatrix2D newContent) {
        if (newContent != null)
            try {
                setUp(newContent.rows(), newContent.columns());
            } catch (IllegalArgumentException exc) { // we can hold rows*columns>Integer.MAX_VALUE cells !
                if (!"matrix too large".equals(exc.getMessage()))
                    throw exc;
            }
        this.content = newContent;
    }

    public IntMatrix2D assign(final IntMatrix2D y, final cern.colt.function.tint.IntIntFunction function) {
        checkShape(y);
        if (y instanceof WrapperIntMatrix2D) {
            IntArrayList rowList = new IntArrayList();
            IntArrayList columnList = new IntArrayList();
            IntArrayList valueList = new IntArrayList();
            y.getNonZeros(rowList, columnList, valueList);
            assign(y, function, rowList, columnList);
        } else {
            super.assign(y, function);
        }
        return this;
    }

    public IntMatrix2D assign(final int[] values) {
        if (content instanceof DiagonalIntMatrix2D) {
            int dlength = ((DiagonalIntMatrix2D) content).dlength;
            final int[] elems = ((DiagonalIntMatrix2D) content).elements;
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

    public Object elements() {
        return content.elements();
    }

    public synchronized int getQuick(int row, int column) {
        return content.getQuick(row, column);
    }

    public boolean equals(int value) {
        if (content instanceof DiagonalIntMatrix2D) {
            int[] elements = (int[]) content.elements();
            for (int r = 0; r < elements.length; r++) {
                int x = elements[r];
                int diff = value - x;
                if (diff != 0) {
                    return false;
                }
            }
            return true;
        } else {
            return super.equals(value);
        }
    }

    public boolean equals(Object obj) {
        if (content instanceof DiagonalIntMatrix2D && obj instanceof DiagonalIntMatrix2D) {
            if (this == obj)
                return true;
            if (!(this != null && obj != null))
                return false;
            DiagonalIntMatrix2D A = (DiagonalIntMatrix2D) content;
            DiagonalIntMatrix2D B = (DiagonalIntMatrix2D) obj;
            if (A.columns() != B.columns() || A.rows() != B.rows() || A.diagonalIndex() != B.diagonalIndex()
                    || A.diagonalLength() != B.diagonalLength())
                return false;
            int[] AElements = A.elements();
            int[] BElements = B.elements();
            for (int r = 0; r < AElements.length; r++) {
                int x = AElements[r];
                int value = BElements[r];
                int diff = value - x;
                if (diff != 0) {
                    return false;
                }
            }
            return true;
        } else {
            return super.equals(obj);
        }
    }

    public IntMatrix2D like(int rows, int columns) {
        return content.like(rows, columns);
    }

    public IntMatrix1D like1D(int size) {
        return content.like1D(size);
    }

    public synchronized void setQuick(int row, int column, int value) {
        content.setQuick(row, column, value);
    }

    public IntMatrix1D vectorize() {
        final DenseIntMatrix1D v = new DenseIntMatrix1D((int) size());
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

    public IntMatrix1D viewColumn(int column) {
        return viewDice().viewRow(column);
    }

    public IntMatrix2D viewColumnFlip() {
        if (columns == 0)
            return this;
        WrapperIntMatrix2D view = new WrapperIntMatrix2D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized int getQuick(int row, int column) {
                return content.getQuick(row, columns - 1 - column);
            }

            public synchronized void setQuick(int row, int column, int value) {
                content.setQuick(row, columns - 1 - column, value);
            }

            public synchronized int get(int row, int column) {
                return content.get(row, columns - 1 - column);
            }

            public synchronized void set(int row, int column, int value) {
                content.set(row, columns - 1 - column, value);
            }
        };
        view.isNoView = false;

        return view;
    }

    public IntMatrix2D viewDice() {
        WrapperIntMatrix2D view = new WrapperIntMatrix2D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized int getQuick(int row, int column) {
                return content.getQuick(column, row);
            }

            public synchronized void setQuick(int row, int column, int value) {
                content.setQuick(column, row, value);
            }

            public synchronized int get(int row, int column) {
                return content.get(column, row);
            }

            public synchronized void set(int row, int column, int value) {
                content.set(column, row, value);
            }
        };
        view.rows = columns;
        view.columns = rows;
        view.isNoView = false;

        return view;
    }

    public IntMatrix2D viewPart(final int row, final int column, int height, int width) {
        checkBox(row, column, height, width);
        WrapperIntMatrix2D view = new WrapperIntMatrix2D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized int getQuick(int i, int j) {
                return content.getQuick(row + i, column + j);
            }

            public synchronized void setQuick(int i, int j, int value) {
                content.setQuick(row + i, column + j, value);
            }

            public synchronized int get(int i, int j) {
                return content.get(row + i, column + j);
            }

            public synchronized void set(int i, int j, int value) {
                content.set(row + i, column + j, value);
            }
        };
        view.rows = height;
        view.columns = width;
        view.isNoView = false;

        return view;
    }

    public IntMatrix1D viewRow(int row) {
        checkRow(row);
        return new DelegateIntMatrix1D(this, row);
    }

    public IntMatrix2D viewRowFlip() {
        if (rows == 0)
            return this;
        WrapperIntMatrix2D view = new WrapperIntMatrix2D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized int getQuick(int row, int column) {
                return content.getQuick(rows - 1 - row, column);
            }

            public synchronized void setQuick(int row, int column, int value) {
                content.setQuick(rows - 1 - row, column, value);
            }

            public synchronized int get(int row, int column) {
                return content.get(rows - 1 - row, column);
            }

            public synchronized void set(int row, int column, int value) {
                content.set(rows - 1 - row, column, value);
            }
        };
        view.isNoView = false;
        return view;
    }

    public IntMatrix2D viewSelection(int[] rowIndexes, int[] columnIndexes) {
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

        WrapperIntMatrix2D view = new WrapperIntMatrix2D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized int getQuick(int i, int j) {
                return content.getQuick(rix[i], cix[j]);
            }

            public synchronized void setQuick(int i, int j, int value) {
                content.setQuick(rix[i], cix[j], value);
            }

            public synchronized int get(int i, int j) {
                return content.get(rix[i], cix[j]);
            }

            public synchronized void set(int i, int j, int value) {
                content.set(rix[i], cix[j], value);
            }
        };
        view.rows = rowIndexes.length;
        view.columns = columnIndexes.length;
        view.isNoView = false;

        return view;
    }

    public IntMatrix2D viewStrides(final int _rowStride, final int _columnStride) {
        if (_rowStride <= 0 || _columnStride <= 0)
            throw new IndexOutOfBoundsException("illegal stride");
        WrapperIntMatrix2D view = new WrapperIntMatrix2D(this) {
            /**
             * 
             */
            private static final long serialVersionUID = 1L;

            public synchronized int getQuick(int row, int column) {
                return content.getQuick(_rowStride * row, _columnStride * column);
            }

            public synchronized void setQuick(int row, int column, int value) {
                content.setQuick(_rowStride * row, _columnStride * column, value);
            }

            public synchronized int get(int row, int column) {
                return content.get(_rowStride * row, _columnStride * column);
            }

            public synchronized void set(int row, int column, int value) {
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

    protected IntMatrix2D getContent() {
        return content;
    }

    protected IntMatrix1D like1D(int size, int offset, int stride) {
        throw new InternalError(); // should never get called
    }

    protected IntMatrix2D viewSelectionLike(int[] rowOffsets, int[] columnOffsets) {
        throw new InternalError(); // should never be called
    }
}
