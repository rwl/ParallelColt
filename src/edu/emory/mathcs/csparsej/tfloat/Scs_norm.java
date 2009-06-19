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
 * Sparse matrix 1-norm.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Scs_norm {

    /**
     * Computes the 1-norm of a sparse matrix = max (sum (abs (A))), largest
     * column sum.
     * 
     * @param A
     *            column-compressed matrix
     * @return the 1-norm if successful, -1 on error
     */
    public static float cs_norm(Scs A) {
        int p, j, n, Ap[];
        float Ax[], norm = 0, s;
        if (!Scs_util.CS_CSC(A) || A.x == null)
            return (-1); /* check inputs */
        n = A.n;
        Ap = A.p;
        Ax = A.x;
        for (j = 0; j < n; j++) {
            for (s = 0, p = Ap[j]; p < Ap[j + 1]; p++)
                s += Math.abs(Ax[p]);
            norm = Math.max(norm, s);
        }
        return (norm);
    }
}
