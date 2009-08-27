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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import cern.colt.matrix.Norm;
import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.algo.DenseFloatAlgebra;
import cern.colt.matrix.tfloat.algo.FloatProperty;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix1D;
import cern.colt.matrix.tfloat.impl.SparseFloatMatrix1D;
import cern.colt.matrix.tfloat.impl.SparseRCMFloatMatrix2D;

/**
 * ILU preconditioner with fill-in. Uses the dual threshold approach of Saad.
 */
public class FloatILUT implements FloatPreconditioner {

    /**
     * Factorisation matrix
     */
    private SparseRCMFloatMatrix2D LU;

    /**
     * Temporary vector for solving the factorised system
     */
    private final FloatMatrix1D y;

    /**
     * Drop-tolerance
     */
    private final float tau;

    /**
     * Stores entries in the lower and upper part of the matrix. Used by the
     * dropping rule to determine the largest entries in the two parts of the
     * matrix
     */
    private final List<IntFloatEntry> lower, upper;

    /**
     * Number of additional entries to keep in the lower and upper part of the
     * factored matrix. The entries of the original matrix are always kept,
     * unless they numerically too small
     */
    private final int p;

    private final int n;

    /**
     * Sets up the preconditioner for the problem size
     * 
     * @param n
     *            Problem size (number of rows)
     * @param tau
     *            Drop tolerance
     * @param p
     *            Number of entries to keep on each row in of the factored
     *            matrix. This is in addition to the entries of the original
     *            matrix
     */
    public FloatILUT(int n, float tau, int p) {
        this.n = n;
        this.tau = tau;
        this.p = p;

        lower = new ArrayList<IntFloatEntry>(n);
        upper = new ArrayList<IntFloatEntry>(n);
        y = new DenseFloatMatrix1D(n);
    }

    /**
     * Sets up the preconditioner for the given problem size. Uses a
     * drop-tolerance of 10<sup>-6</sup>, and keeps 25 entries on each row,
     * including the main diagonal and any previous entries in the matrix
     * structure
     * 
     * @param n
     *            Problem size (number of rows)
     */
    public FloatILUT(int n) {
        this(n, 1e-6f, 25);
    }

    public FloatMatrix1D apply(FloatMatrix1D b, FloatMatrix1D x) {
        if (x == null) {
            x = b.like();
        }

        // Ly = b, y = L\b
        unitLowerSolve(b, y);

        // Ux = L\b = y
        return upperSolve(y, x);
    }

    public FloatMatrix1D transApply(FloatMatrix1D b, FloatMatrix1D x) {
        if (x == null) {
            x = b.like();
        }

        // U'y = b, y = U'\b
        upperTransSolve(b, y);
        // L'x = U'\b = y
        return unitLowerTransSolve(y, x);
    }

    public void setMatrix(FloatMatrix2D A) {
        FloatProperty.DEFAULT.isSquare(A);
        if (A.rows() != n) {
            throw new IllegalArgumentException("A.rows() != n");
        }
        LU = new SparseRCMFloatMatrix2D(n, n);
        LU.assign(A);
        LU.trimToSize();

        factor();
    }

    private void factor() {
        int n = LU.rows();

        for (int i = 1; i < n; ++i) {

            // Get row i
            SparseFloatMatrix1D rowi = LU.viewRow(i);

            // Drop tolerance on current row
            float taui = DenseFloatAlgebra.DEFAULT.norm(rowi, Norm.Two) * tau;

            for (int k = 0; k < i; ++k) {

                // Get row k
                SparseFloatMatrix1D rowk = LU.viewRow(k);

                if (rowk.getQuick(k) == 0)
                    throw new RuntimeException("Zero diagonal entry on row " + (k + 1) + " during ILU process");

                float LUik = rowi.getQuick(k) / rowk.getQuick(k);

                // Check for small elimination entry
                if (Math.abs(LUik) <= taui)
                    continue;

                // Traverse the sparse row k, reducing row i
                int rowUsed = (int) rowk.size();
                for (int j = k + 1; j < rowUsed; ++j)
                    rowi.setQuick(j, rowi.getQuick(j) - LUik * rowk.getQuick(j));

                // The above has overwritten LUik, so remedy that
                rowi.setQuick(k, LUik);
            }

            // Store back into the LU matrix, dropping as needed
            gather(rowi, taui, i);
        }
        //        System.out.println(LU.toString());
    }

    /**
     * Copies the dense array back into the sparse vector, applying a numerical
     * dropping rule and keeping only a given number of entries
     */
    private void gather(SparseFloatMatrix1D v, float taui, int d) {
        // Number of entries in the lower and upper part of the original matrix
        int nl = 0, nu = 0;
        long[] indexes = v.elements().keys().elements();
        for (int i = 0; i < indexes.length; i++) {
            if (indexes[i] < d)
                nl++;
            else if (indexes[i] > d)
                nu++;
        }
        float[] z = v.toArray();
        v.assign(0);

        // Entries in the L part of the vector
        lower.clear();
        for (int i = 0; i < d; ++i)
            if (Math.abs(z[i]) > taui)
                lower.add(new IntFloatEntry(i, z[i]));

        // Entries in the U part of the vector
        upper.clear();
        for (int i = d + 1; i < z.length; ++i)
            if (Math.abs(z[i]) > taui)
                upper.add(new IntFloatEntry(i, z[i]));

        // Sort in descending order
        Collections.sort(lower);
        Collections.sort(upper);

        // Always keep the diagonal
        v.setQuick(d, z[d]);

        // Keep at most nl+p lower entries
        for (int i = 0; i < Math.min(nl + p, lower.size()); ++i) {
            IntFloatEntry e = lower.get(i);
            v.setQuick(e.index, e.value);
        }

        // Keep at most nu+p upper entries
        for (int i = 0; i < Math.min(nu + p, upper.size()); ++i) {
            IntFloatEntry e = upper.get(i);
            v.setQuick(e.index, e.value);
        }
    }

    /**
     * Stores an integer/value pair, sorted by descending order according to the
     * value
     */
    private static class IntFloatEntry implements Comparable<IntFloatEntry> {

        public int index;

        public float value;

        public IntFloatEntry(int index, float value) {
            this.index = index;
            this.value = value;
        }

        public int compareTo(IntFloatEntry o) {
            // Descending order, so keep the largest entries first
            if (Math.abs(value) < Math.abs(o.value))
                return 1;
            else if (Math.abs(value) == Math.abs(o.value))
                return 0;
            else
                return -1;
        }

        public String toString() {
            return "(" + index + "=" + value + ")";
        }
    }

    private FloatMatrix1D unitLowerSolve(FloatMatrix1D b, FloatMatrix1D x) {
        float[] bd = ((DenseFloatMatrix1D) b).elements();
        float[] xd = ((DenseFloatMatrix1D) x).elements();
        int rows = LU.rows();
        for (int i = 0; i < rows; ++i) {

            // Get row i
            SparseFloatMatrix1D row = LU.viewRow(i);

            // xi = bi - sum[j<i] Lij * xj
            float sum = 0;
            for (int j = 0; j < i; ++j)
                sum += row.getQuick(j) * xd[j];

            xd[i] = bd[i] - sum;
        }

        return x;
    }

    private FloatMatrix1D unitLowerTransSolve(FloatMatrix1D b, FloatMatrix1D x) {

        x.assign(b);

        float[] xd = ((DenseFloatMatrix1D) x).elements();
        int rows = LU.rows();

        for (int i = rows - 1; i >= 0; --i) {

            // Get row i
            SparseFloatMatrix1D row = LU.viewRow(i);

            // At this stage, x[i] is known, so move it over to the right
            // hand side for the remaining equations
            for (int j = 0; j < i; ++j)
                xd[j] -= row.getQuick(j) * xd[i];

        }
        return x;
    }

    private FloatMatrix1D upperSolve(FloatMatrix1D b, FloatMatrix1D x) {

        float[] bd = ((DenseFloatMatrix1D) b).elements();
        float[] xd = ((DenseFloatMatrix1D) x).elements();
        int rows = LU.rows();
        for (int i = rows - 1; i >= 0; --i) {

            // Get row i
            SparseFloatMatrix1D row = LU.viewRow(i);
            int used = (int) row.size();

            // xi = (bi - sum[j>i] Uij * xj) / Uii
            float sum = 0;
            for (int j = i + 1; j < used; ++j)
                sum += row.getQuick(j) * xd[j];

            xd[i] = (bd[i] - sum) / row.getQuick(i);
        }

        return x;
    }

    private FloatMatrix1D upperTransSolve(FloatMatrix1D b, FloatMatrix1D x) {
        x.assign(b);

        float[] xd = ((DenseFloatMatrix1D) x).elements();
        int rows = LU.rows();
        for (int i = 0; i < rows; ++i) {

            // Get row i
            SparseFloatMatrix1D row = LU.viewRow(i);
            int used = (int) row.size();

            // Solve for the current entry
            xd[i] /= row.getQuick(i);

            // Move this known solution over to the right hand side for the
            // remaining equations
            for (int j = i + 1; j < used; ++j)
                xd[j] -= row.getQuick(j) * xd[i];
        }

        return x;
    }

}
