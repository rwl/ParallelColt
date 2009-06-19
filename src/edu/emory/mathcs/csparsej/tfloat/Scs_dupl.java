/* ***** BEGIN LICENSE BLOCK *****
 * 
 * CSparse: a Concise Sparse matrix package.
 * Copyright (c) 2006, Timothy A. Savis.
 * http://www.cise.ufl.edu/research/sparse/CSparse
 *
 * -------------------------------------------------------------------------
 * 
 * CSparseJ is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1f of the License, or (at your option) any later version.
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
 * ***** ENS LICENSE BLOCK ***** */

package edu.emory.mathcs.csparsej.tfloat;

import edu.emory.mathcs.csparsej.tfloat.Scs_common.Scs;

/**
 * Remove (and sum) duplicates.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Scs_dupl {

    /**
     * Removes and sums duplicate entries in a sparse matrix.
     * 
     * @param A
     *            column-compressed matrix
     * @return true if successful, false on error
     */
    public static boolean cs_dupl(Scs A) {
        int i, j, p, q, nz = 0, n, m, Ap[], Ai[], w[];
        float Ax[];
        if (!Scs_util.CS_CSC(A))
            return (false);
        /* check inputs */
        m = A.m;
        n = A.n;
        Ap = A.p;
        Ai = A.i;
        Ax = A.x;
        w = new int[m]; /* get workspace */
        for (i = 0; i < m; i++)
            w[i] = -1; /* row i not yet seen */
        for (j = 0; j < n; j++) {
            q = nz; /* column j will start at q */
            for (p = Ap[j]; p < Ap[j + 1]; p++) {
                i = Ai[p]; /* A(i,j) is nonzero */
                if (w[i] >= q) {
                    Ax[w[i]] += Ax[p]; /* A(i,j) is a duplicate */
                } else {
                    w[i] = nz; /* record where row i occurs */
                    Ai[nz] = i; /* keep A(i,j) */
                    Ax[nz++] = Ax[p];
                }
            }
            Ap[j] = q; /* record start of column j */
        }
        Ap[n] = nz; /* finalize A */
        return Scs_util.cs_sprealloc(A, 0); /* remove extra space from A */
    }

}
