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

package cern.colt.matrix.tfloat.algo.solver.preconditioner;

import java.util.Arrays;

import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.algo.FloatProperty;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix1D;
import cern.colt.matrix.tfloat.impl.SparseRCFloatMatrix2D;

/**
 * Incomplete Cholesky preconditioner without fill-in using a compressed row
 * matrix as internal storage
 */
public class FloatICC implements FloatPreconditioner {

    /**
     * Factorisation matrix
     */
    private SparseRCFloatMatrix2D R;

    /**
     * Temporary vector for solving the factorised system
     */
    private final FloatMatrix1D y;

    private int[] diagind;

    private final int n;

    /**
     * Sets up the ICC preconditioner
     * 
     * @param n
     *            Problem size (number of rows)
     */
    public FloatICC(int n) {
        this.n = n;
        y = new DenseFloatMatrix1D(n);
    }

    public FloatMatrix1D apply(FloatMatrix1D b, FloatMatrix1D x) {
        if (x == null) {
            x = b.like();
        }

        // R'y = b, y = R'\b
        upperTransSolve(b, y);

        // Rx = R'\b = y
        return upperSolve(y, x);
    }

    public FloatMatrix1D transApply(FloatMatrix1D b, FloatMatrix1D x) {
        if (x == null) {
            x = b.like();
        }

        return apply(b, x);
    }

    public void setMatrix(FloatMatrix2D A) {
        FloatProperty.DEFAULT.isSquare(A);
        if (A.rows() != n) {
            throw new IllegalArgumentException("A.rows() != n");
        }
        R = new SparseRCFloatMatrix2D(n, n);
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
        float[] data = R.getValues();

        // Temporary storage of a dense row
        float[] Rk = new float[n];

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
                float Rii = data[diagind[i]];

                if (Rii == 0)
                    throw new RuntimeException("Zero pivot encountered on row " + (i + 1) + " during ICC process");

                // Elimination factor
                float Rki = Rk[i] / Rii;

                if (Rki == 0)
                    continue;

                // Traverse the sparse row i, reducing on row k
                for (int j = diagind[i] + 1; j < rowptr[i + 1]; ++j)
                    Rk[colind[j]] -= Rki * data[j];
            }

            // Store the row back into the factorisation matrix
            if (Rk[k] == 0)
                throw new RuntimeException("Zero diagonal entry encountered on row " + (k + 1) + " during ICC process");
            float sqRkk = (float) Math.sqrt(Rk[k]);

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

    private FloatMatrix1D upperSolve(FloatMatrix1D b, FloatMatrix1D x) {
        float[] bd = ((DenseFloatMatrix1D) b).elements();
        float[] xd = ((DenseFloatMatrix1D) x).elements();
        int[] colind = R.getColumnIndexes();
        int[] rowptr = R.getRowPointers();
        float[] data = R.getValues();
        int rows = R.rows();

        for (int i = rows - 1; i >= 0; --i) {

            // xi = (bi - sum[j>i] Uij * xj) / Uii
            float sum = 0;
            for (int j = diagind[i] + 1; j < rowptr[i + 1]; ++j)
                sum += data[j] * xd[colind[j]];

            xd[i] = (bd[i] - sum) / data[diagind[i]];
        }

        return x;
    }

    private FloatMatrix1D upperTransSolve(FloatMatrix1D b, FloatMatrix1D x) {
        x.assign(b);

        float[] xd = ((DenseFloatMatrix1D) x).elements();
        int[] colind = R.getColumnIndexes();
        int[] rowptr = R.getRowPointers();
        float[] data = R.getValues();
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
