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
 * Permute a sparse matrix.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Scs_permute {
    /**
     * Permutes a sparse matrix, C = PAQ.
     * 
     * @param A
     *            m-by-n, column-compressed matrix
     * @param pinv
     *            a permutation vector of length m
     * @param q
     *            a permutation vector of length n
     * @param values
     *            allocate pattern only if false, values and pattern otherwise
     * @return C = PAQ, null on error
     */
    public static Scs cs_permute(Scs A, int[] pinv, int[] q, boolean values) {
        int t, j, k, nz = 0, m, n, Ap[], Ai[], Cp[], Ci[];
        float Cx[], Ax[];
        Scs C;
        if (!Scs_util.CS_CSC(A))
            return (null); /* check inputs */
        m = A.m;
        n = A.n;
        Ap = A.p;
        Ai = A.i;
        Ax = A.x;
        C = Scs_util.cs_spalloc(m, n, Ap[n], values && Ax != null, false); /* alloc result */
        Cp = C.p;
        Ci = C.i;
        Cx = C.x;
        for (k = 0; k < n; k++) {
            Cp[k] = nz; /* column k of C is column q[k] of A */
            j = q != null ? (q[k]) : k;
            for (t = Ap[j]; t < Ap[j + 1]; t++) {
                if (Cx != null)
                    Cx[nz] = Ax[t]; /* row i of A is row pinv[i] of C */
                Ci[nz++] = pinv != null ? (pinv[Ai[t]]) : Ai[t];
            }
        }
        Cp[n] = nz; /* finalize the last column of C */
        return C;
    }

}
