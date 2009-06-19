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
 * Solve an upper triangular system Ux=b.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Dcs_usolve {
    /**
     * Solves an upper triangular system Ux=b, where x and b are dense vectors.
     * The diagonal of U must be the last entry of each column.
     * 
     * @param U
     *            upper triangular matrix in column-compressed form
     * @param x
     *            size n, right hand side on input, solution on output
     * @return true if successful, false on error
     */
    public static boolean cs_usolve(Dcs U, double[] x) {
        int p, j, n, Up[], Ui[];
        double Ux[];
        if (!Dcs_util.CS_CSC(U) || x == null)
            return (false); /* check inputs */
        n = U.n;
        Up = U.p;
        Ui = U.i;
        Ux = U.x;
        for (j = n - 1; j >= 0; j--) {
            x[j] /= Ux[Up[j + 1] - 1];
            for (p = Up[j]; p < Up[j + 1] - 1; p++) {
                x[Ui[p]] -= Ux[p] * x[j];
            }
        }
        return (true);
    }

}
