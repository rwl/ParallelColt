/*
 * Copyright (C) 2010-2012 Richard Lincoln
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 */

package edu.emory.mathcs.utils.pc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import cern.colt.matrix.AbstractMatrix;
import cern.colt.matrix.io.MatrixInfo;
import cern.colt.matrix.io.MatrixSize;
import cern.colt.matrix.io.MatrixVectorReader;
import cern.colt.matrix.tdcomplex.DComplexFactory1D;
import cern.colt.matrix.tdcomplex.DComplexFactory2D;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix1D;
import cern.colt.matrix.tdcomplex.impl.DenseDComplexMatrix2D;
import cern.colt.matrix.tdcomplex.impl.SparseDComplexMatrix1D;
import cern.colt.matrix.tdcomplex.impl.SparseDComplexMatrix2D;
import cern.colt.matrix.tdouble.DoubleFactory1D;
import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;

/**
 * Utility class for reading data in MatrixMarket format.
 *
 * @author Richard Lincoln
 */
public class MatrixMarketUtils {

	private static int i, j;
	private static int[] row, col;
	private static double[] data, dataR, dataI;

	private static FileReader fileReader;
	private static MatrixVectorReader reader;
	private static MatrixInfo info;
	private static MatrixSize size;

	private static AbstractMatrix m;

	/**
	 *
	 * @param uri
	 * @return
	 */
	public static AbstractMatrix readMatrix(String fileName) {

		try {
			fileReader = new FileReader(fileName);
			reader = new MatrixVectorReader(fileReader);

			info = reader.readMatrixInfo();
			size = reader.readMatrixSize(info);

			data  = new double[size.numEntries()];
			dataR = new double[size.numEntries()];
			dataI = new double[size.numEntries()];

			row = new int[size.numEntries()];
			col = new int[size.numEntries()];

			if (info.isArray()) {
				if (info.isComplex()) {
					try {
						reader.readArray(dataR, dataI);
					} catch (IOException e) {
						e.printStackTrace();
					}
					if (info.isDense()) {
						m = DComplexFactory1D.dense.make(size.numEntries());
						for (i = 0; i < size.numEntries(); i++)
							((DenseDComplexMatrix1D) m).setQuick(i, dataR[i], dataI[i]);
					} else if (info.isSparse()) {
						m = DComplexFactory1D.sparse.make(size.numEntries());
						for (i = 0; i < size.numEntries(); i++)
							((SparseDComplexMatrix1D) m).setQuick(i, dataR[i], dataI[i]);
					} else {
						throw new UnsupportedOperationException();
					}
				} else {
					reader.readArray(data);

					if (size.numRows() == 1 || size.numColumns() == 1) {
						if (info.isDense()) {
							m = DoubleFactory1D.dense.make(size.numEntries());
							for (i = 0; i < size.numEntries(); i++)
								((DenseDoubleMatrix1D) m).setQuick(i, data[i]);
						} else if (info.isSparse()) {
							m = DoubleFactory1D.sparse.make(size.numEntries());
							for (i = 0; i < size.numEntries(); i++)
								((SparseDoubleMatrix1D) m).setQuick(i, data[i]);
						} else {
							throw new UnsupportedOperationException();
						}
					} else {
						if (info.isDense()) {
							m = DoubleFactory2D.dense.make(size.numRows(), size.numColumns());
							for (i = 0; i < size.numColumns(); i++) {
								for (j = 0; j < size.numRows(); j++) {
									((DenseDoubleMatrix2D) m).setQuick(j, i, data[i * size.numRows() + j]);
								}
							}
						} else if (info.isSparse()) {
							m = DoubleFactory2D.sparse.make(size.numRows(), size.numColumns());
							for (i = 0; i < size.numColumns(); i++) {
								for (j = 0; j < size.numRows(); j++) {
									((SparseDoubleMatrix2D) m).setQuick(j, i, data[i * size.numRows() + j]);
								}
							}
						} else {
							throw new UnsupportedOperationException();
						}
					}
				}
			} else if (info.isCoordinate()) {
				if (info.isComplex()) {
					reader.readCoordinate(row, col, dataR, dataI);

					if (info.isDense()) {
						m = DComplexFactory2D.dense.make(size.numRows(), size.numColumns());
						for (i = 0; i < size.numEntries(); i++) {
							((DenseDComplexMatrix2D) m).setQuick(row[i], col[i], dataR[i], dataI[i]);
							if (info.isSymmetric())
								((DenseDComplexMatrix2D) m).setQuick(col[i], row[i], dataR[i], dataI[i]);
						}
					} else if (info.isSparse()) {
						m = DComplexFactory2D.sparse.make(size.numRows(), size.numColumns());
						for (i = 0; i < size.numEntries(); i++) {
							((SparseDComplexMatrix2D) m).setQuick(row[i], col[i], dataR[i], dataI[i]);
							if (info.isSymmetric())
								((SparseDComplexMatrix2D) m).setQuick(col[i], row[i], dataR[i], dataI[i]);
						}
					} else {
						throw new UnsupportedOperationException();
					}
				} else {
					reader.readCoordinate(row, col, data);

					if (info.isDense()) {
						m = DoubleFactory2D.dense.make(size.numRows(), size.numColumns());
						for (i = 0; i < size.numEntries(); i++) {
							((DenseDoubleMatrix2D) m).setQuick(row[i], col[i], data[i]);
							if (info.isSymmetric())
								((DenseDoubleMatrix2D) m).setQuick(col[i], row[i], data[i]);
						}
					} else if (info.isSparse()) {
						m = DoubleFactory2D.sparse.make(size.numRows(), size.numColumns());
						for (i = 0; i < size.numEntries(); i++) {
							((SparseDoubleMatrix2D) m).setQuick(row[i], col[i], data[i]);
							if (info.isSymmetric()) {
								((SparseDoubleMatrix2D) m).setQuick(col[i], row[i], data[i]);
							}
						}
					} else {
						throw new UnsupportedOperationException();
					}
				}
			} else {
				throw new UnsupportedOperationException();
			}

			fileReader.close();
			reader.close();
		} catch (FileNotFoundException e) {
			// TODO Handle exception
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return m;
	}

	public static AbstractMatrix readMatrix(File file) {
		return readMatrix(file.getAbsolutePath());
	}

}
