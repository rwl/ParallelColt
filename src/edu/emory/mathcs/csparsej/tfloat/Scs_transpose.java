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
 * Transpose a sparse matrix.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Scs_transpose {

    /**
     * Computes the transpose of a sparse matrix, C =A';
     * 
     * @param A
     *            column-compressed matrix
     * @param values
     *            pattern only if false, both pattern and values otherwise
     * @return C=A', null on error
     */
    public static Scs cs_transpose(Scs A, boolean values) {
        int p, q, j, Cp[], Ci[], n, m, Ap[], Ai[], w[];
        float Cx[], Ax[];
        Scs C;
        if (!Scs_util.CS_CSC(A))
            return (null); /* check inputs */
        m = A.m;
        n = A.n;
        Ap = A.p;
        Ai = A.i;
        Ax = A.x;
        C = Scs_util.cs_spalloc(n, m, Ap[n], values && (Ax != null), false); /* allocate result */
        w = new int[m]; /* get workspace */
        Cp = C.p;
        Ci = C.i;
        Cx = C.x;
        for (p = 0; p < Ap[n]; p++)
            w[Ai[p]]++; /* row counts */
        Scs_cumsum.cs_cumsum(Cp, w, m); /* row pointers */
        for (j = 0; j < n; j++) {
            for (p = Ap[j]; p < Ap[j + 1]; p++) {
                Ci[q = w[Ai[p]]++] = j; /* place A(i,j) as entry C(j,i) */
                if (Cx != null)
                    Cx[q] = Ax[p];
            }
        }
        return C;
    }

}
