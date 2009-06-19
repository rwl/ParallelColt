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
 * Solve a lower triangular system Lx=b.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Scs_lsolve {
    /**
     * Solves a lower triangular system Lx=b where x and b are dense. x=b on
     * input, solution on output.
     * 
     * @param L
     *            column-compressed, lower triangular matrix
     * @param x
     *            size n, right hand side on input, solution on output
     * @return true if successful, false on error
     */
    public static boolean cs_lsolve(Scs L, float[] x) {
        int p, j, n, Lp[], Li[];
        float Lx[];
        if (!Scs_util.CS_CSC(L) || x == null)
            return (false); /* check inputs */
        n = L.n;
        Lp = L.p;
        Li = L.i;
        Lx = L.x;
        for (j = 0; j < n; j++) {
            x[j] /= Lx[Lp[j]];
            for (p = Lp[j] + 1; p < Lp[j + 1]; p++) {
                x[Li[p]] -= Lx[p] * x[j];
            }
        }
        return true;
    }

}
