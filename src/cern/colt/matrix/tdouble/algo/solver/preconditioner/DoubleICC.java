/*
 * Copyright (C) 2003-2006 Bjørn-Ove Heimsund
 * 
 * This file is part of MTJ.
 * 
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation; either version 2.1 of the License, or (at your
 * option) any later version.
 * 
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
 * for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

package cern.colt.matrix.tdouble.algo.solver.preconditioner;

import java.util.Arrays;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DoubleProperty;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.SparseRCDoubleMatrix2D;

/**
 * Incomplete Cholesky preconditioner without fill-in using a compressed row
 * matrix as internal storage
 */
public class DoubleICC implements DoublePreconditioner {

    /**
     * Factorisation matrix
     */
    private SparseRCDoubleMatrix2D R;

    /**
     * Temporary vector for solving the factorised system
     */
    private final DoubleMatrix1D y;

    private int[] diagind;

    private final int n;

    /**
     * Sets up the ICC preconditioner
     * 
     * @param n
     *            Problem size (number of rows)
     */
    public DoubleICC(int n) {
        this.n = n;
        y = new DenseDoubleMatrix1D(n);
    }

    public DoubleMatrix1D apply(DoubleMatrix1D b, DoubleMatrix1D x) {
        if (x == null) {
            x = b.like();
        }

        // R'y = b, y = R'\b
        upperTransSolve(b, y);

        // Rx = R'\b = y
        return upperSolve(y, x);
    }

    public DoubleMatrix1D transApply(DoubleMatrix1D b, DoubleMatrix1D x) {
        if (x == null) {
            x = b.like();
        }

        return apply(b, x);
    }

    public void setMatrix(DoubleMatrix2D A) {
        DoubleProperty.DEFAULT.isSquare(A);
        if (A.rows() != n) {
            throw new IllegalArgumentException("A.rows() != n");
        }
        R = new SparseRCDoubleMatrix2D(n, n);
        R.assign(A);
        if (!R.hasColumnIndexesSorted()) {
            R.sortColumnIndexes();
        }
        factor();
    }

    private void factor() {
        int n = R.rows();

        // Internal CRS matrix storage
        int[] colind = R.getColumnIndexes();
        int[] rowptr = R.getRowPointers();
        double[] data = R.getValues();

        // Temporary storage of a dense row
        double[] Rk = new double[n];

        // Find the indexes to the diagonal entries
        diagind = findDiagonalIndexes(n, colind, rowptr);

        // Go down along the main diagonal
        for (int k = 0; k < n; ++k) {

            // Expand current row to dense storage
            Arrays.fill(Rk, 0);
            for (int i = rowptr[k]; i < rowptr[k + 1]; ++i)
                Rk[colind[i]] = data[i];

            for (int i = 0; i < k; ++i) {

                // Get the current diagonal entry
                double Rii = data[diagind[i]];

                if (Rii == 0)
                    throw new RuntimeException("Zero pivot encountered on row " + (i + 1) + " during ICC process");

                // Elimination factor
                double Rki = Rk[i] / Rii;

                if (Rki == 0)
                    continue;

                // Traverse the sparse row i, reducing on row k
                for (int j = diagind[i] + 1; j < rowptr[i + 1]; ++j)
                    Rk[colind[j]] -= Rki * data[j];
            }

            // Store the row back into the factorisation matrix
            if (Rk[k] == 0)
                throw new RuntimeException("Zero diagonal entry encountered on row " + (k + 1) + " during ICC process");
            double sqRkk = Math.sqrt(Rk[k]);

            for (int i = diagind[k]; i < rowptr[k + 1]; ++i)
                data[i] = Rk[colind[i]] / sqRkk;
        }
    }

    private int[] findDiagonalIndexes(int m, int[] colind, int[] rowptr) {
        int[] diagind = new int[m];

        for (int k = 0; k < m; ++k) {
            diagind[k] = cern.colt.Sorting.binarySearchFromTo(colind, k, rowptr[k], rowptr[k + 1] - 1);

            if (diagind[k] < 0)
                throw new RuntimeException("Missing diagonal entry on row " + (k + 1));
        }

        return diagind;
    }

    private DoubleMatrix1D upperSolve(DoubleMatrix1D b, DoubleMatrix1D x) {
        double[] bd = ((DenseDoubleMatrix1D) b).elements();
        double[] xd = ((DenseDoubleMatrix1D) x).elements();
        int[] colind = R.getColumnIndexes();
        int[] rowptr = R.getRowPointers();
        double[] data = R.getValues();
        int rows = R.rows();

        for (int i = rows - 1; i >= 0; --i) {

            // xi = (bi - sum[j>i] Uij * xj) / Uii
            double sum = 0;
            for (int j = diagind[i] + 1; j < rowptr[i + 1]; ++j)
                sum += data[j] * xd[colind[j]];

            xd[i] = (bd[i] - sum) / data[diagind[i]];
        }

        return x;
    }

    private DoubleMatrix1D upperTransSolve(DoubleMatrix1D b, DoubleMatrix1D x) {
        x.assign(b);

        double[] xd = ((DenseDoubleMatrix1D) x).elements();
        int[] colind = R.getColumnIndexes();
        int[] rowptr = R.getRowPointers();
        double[] data = R.getValues();
        int rows = R.rows();

        for (int i = 0; i < rows; ++i) {

            // Solve for the current entry
            xd[i] /= data[diagind[i]];

            // Move this known solution over to the right hand side for the
            // remaining equations
            for (int j = diagind[i] + 1; j < rowptr[i + 1]; ++j)
                xd[colind[j]] -= data[j] * xd[i];
        }

        return x;
    }
}
