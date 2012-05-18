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

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.SparseRCDoubleMatrix2D;

/**
 * SSOR preconditioner. Uses symmetrical sucessive overrelaxation as a
 * preconditioner. Meant for symmetrical, positive definite matrices. For best
 * performance, omega must be carefully chosen (between 0 and 2).
 */
public class DoubleSSOR implements DoublePreconditioner {

    /**
     * Overrelaxation parameter for the forward sweep
     */
    private double omegaF;

    /**
     * Overrelaxation parameter for the backwards sweep
     */
    private double omegaR;

    /**
     * Holds a copy of the matrix A in the compressed row format
     */
    private SparseRCDoubleMatrix2D F;

    /**
     * indexes to the diagonal entries of the matrix
     */
    private final int[] diagind;

    /**
     * Temporary vector for holding the half-step state
     */
    private final double[] xx;

    /**
     * True if the reverse (backward) sweep is to be done. Without this, the
     * method is SOR instead of SSOR
     */
    private final boolean reverse;

    private final int n;

    /**
     * Constructor for SSOR
     * 
     * @param n
     *            Problem size (number of rows)
     * @param reverse
     *            True to perform a reverse sweep as well as the forward sweep.
     *            If false, this preconditioner becomes the SOR method instead
     * @param omegaF
     *            Overrelaxation parameter for the forward sweep. Between 0 and
     *            2.
     * @param omegaR
     *            Overrelaxation parameter for the backwards sweep. Between 0
     *            and 2.
     */
    public DoubleSSOR(int n, boolean reverse, double omegaF, double omegaR) {
        this.n = n;
        this.reverse = reverse;
        setOmega(omegaF, omegaR);
        diagind = new int[n];
        xx = new double[n];
    }

    /**
     * Constructor for SSOR. Uses <code>omega=1</code> with a backwards sweep
     * 
     * @param n
     *            Problem size (number of rows)
     */
    public DoubleSSOR(int n) {
        this(n, true, 1, 1);
    }

    /**
     * Sets the overrelaxation parameters
     * 
     * @param omegaF
     *            Overrelaxation parameter for the forward sweep. Between 0 and
     *            2.
     * @param omegaR
     *            Overrelaxation parameter for the backwards sweep. Between 0
     *            and 2.
     */
    public void setOmega(double omegaF, double omegaR) {
        if (omegaF < 0 || omegaF > 2)
            throw new IllegalArgumentException("omegaF must be between 0 and 2");
        if (omegaR < 0 || omegaR > 2)
            throw new IllegalArgumentException("omegaR must be between 0 and 2");

        this.omegaF = omegaF;
        this.omegaR = omegaR;
    }

    public void setMatrix(DoubleMatrix2D A) {
        if (A.rows() != n) {
            throw new IllegalArgumentException("A.rows() != n");
        }
        F = new SparseRCDoubleMatrix2D(n, n);
        F.assign(A);
        if (!F.hasColumnIndexesSorted()) {
            F.sortColumnIndexes();
        }

        int[] rowptr = F.getRowPointers();
        int[] colind = F.getColumnIndexes();

        // Find the indexes to the diagonal entries
        for (int k = 0; k < n; ++k) {
            diagind[k] = cern.colt.Sorting.binarySearchFromTo(colind, k, rowptr[k], rowptr[k + 1] - 1);
            if (diagind[k] < 0)
                throw new RuntimeException("Missing diagonal on row " + (k + 1));
        }
    }

    public DoubleMatrix1D apply(DoubleMatrix1D b, DoubleMatrix1D x) {
        if (x == null) {
            x = b.like();
        }

        if (!(b instanceof DenseDoubleMatrix1D) || !(x instanceof DenseDoubleMatrix1D))
            throw new IllegalArgumentException("b and x must be a DenseDoubleMatrix1D");

        int[] rowptr = F.getRowPointers();
        int[] colind = F.getColumnIndexes();
        double[] data = F.getValues();

        double[] bd = ((DenseDoubleMatrix1D) b).elements();
        //        double[] xd = ((DenseDoubleMatrix1D) x).elements();
        double[] xd = new double[(int) x.size()];

        int n = F.rows();
        System.arraycopy(xd, 0, xx, 0, n);

        // Forward sweep (xd oldest, xx halfiterate)
        for (int i = 0; i < n; ++i) {

            double sigma = 0;
            for (int j = rowptr[i]; j < diagind[i]; ++j)
                sigma += data[j] * xx[colind[j]];

            for (int j = diagind[i] + 1; j < rowptr[i + 1]; ++j)
                sigma += data[j] * xd[colind[j]];

            sigma = (bd[i] - sigma) / data[diagind[i]];

            xx[i] = xd[i] + omegaF * (sigma - xd[i]);
        }

        // Stop here if the reverse sweep was not requested
        if (!reverse) {
            System.arraycopy(xx, 0, xd, 0, n);
            x.assign(xd);
            return x;
        }

        // Backward sweep (xx oldest, xd halfiterate)
        for (int i = n - 1; i >= 0; --i) {

            double sigma = 0;
            for (int j = rowptr[i]; j < diagind[i]; ++j)
                sigma += data[j] * xx[colind[j]];

            for (int j = diagind[i] + 1; j < rowptr[i + 1]; ++j)
                sigma += data[j] * xd[colind[j]];

            sigma = (bd[i] - sigma) / data[diagind[i]];

            xd[i] = xx[i] + omegaR * (sigma - xx[i]);
        }
        x.assign(xd);
        return x;
    }

    public DoubleMatrix1D transApply(DoubleMatrix1D b, DoubleMatrix1D x) {
        if (x == null) {
            x = b.like();
        }

        // Assume a symmetric matrix
        return apply(b, x);
    }

}
