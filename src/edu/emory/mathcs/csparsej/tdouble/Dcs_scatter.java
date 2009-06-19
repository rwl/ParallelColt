/* ***** BEGIN LICENSE BLOCK *****
 * 
 * CSparse: a Concise Sparse matrix package.
 * Copyright (c) 2006, Timothy A. Davis.
 * http://www.cise.ufl.edu/research/sparse/CSparse
 *
 * -------------------------------------------------------------------------
 * 
 * CSparseJ is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * CSparseJ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public
 * License along with this Module; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA
 *
 * ***** END LICENSE BLOCK ***** */

package edu.emory.mathcs.csparsej.tdouble;

import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcs;

/**
 * Scatter a sparse vector.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Dcs_scatter {

    /**
     * Scatters and sums a sparse vector A(:,j) into a dense vector, x = x +
     * beta * A(:,j).
     * 
     * @param A
     *            the sparse vector is A(:,j)
     * @param j
     *            the column of A to use
     * @param beta
     *            scalar multiplied by A(:,j)
     * @param w
     *            size m, node i is marked if w[i] = mark
     * @param x
     *            size m, ignored if null
     * @param mark
     *            mark value of w
     * @param C
     *            pattern of x accumulated in C.i
     * @param nz
     *            pattern of x placed in C starting at C.i[nz]
     * @return new value of nz, -1 on error
     */
    public static int cs_scatter(Dcs A, int j, double beta, int[] w, double[] x, int mark, Dcs C, int nz) {
        int i, p;
        int Ap[], Ai[], Ci[];
        double[] Ax;
        if (!Dcs_util.CS_CSC(A) || w == null || !Dcs_util.CS_CSC(C))
            return (-1); /* check inputs */
        Ap = A.p;
        Ai = A.i;
        Ax = A.x;
        Ci = C.i;
        for (p = Ap[j]; p < Ap[j + 1]; p++) {
            i = Ai[p]; /* A(i,j) is nonzero */
            if (w[i] < mark) {
                w[i] = mark; /* i is new entry in column j */
                Ci[nz++] = i; /* add i to pattern of C(:,j) */
                if (x != null)
                    x[i] = beta * Ax[p]; /* x(i) = beta*A(i,j) */
            } else if (x != null)
                x[i] += beta * Ax[p]; /* i exists in C(:,j) already */
        }
        return nz;
    }
}
