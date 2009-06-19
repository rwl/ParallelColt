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
import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcsn;
import edu.emory.mathcs.csparsej.tdouble.Dcs_common.Dcss;

/**
 * Solves Ax=b where A is symmetric positive definite.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Dcs_cholsol {

    /**
     * Solves Ax=b where A is symmetric positive definite; b is overwritten with
     * solution.
     * 
     * @param order
     *            ordering method to use (0 or 1)
     * @param A
     *            column-compressed matrix, symmetric positive definite, only
     *            upper triangular part is used
     * @param b
     *            right hand side, b is overwritten with solution
     * @return true if successful, false on error
     */
    public static boolean cs_cholsol(int order, Dcs A, double[] b) {
        double x[];
        Dcss S;
        Dcsn N;
        int n;
        boolean ok;
        if (!Dcs_util.CS_CSC(A) || b == null)
            return (false); /* check inputs */
        n = A.n;
        S = Dcs_schol.cs_schol(order, A); /* ordering and symbolic analysis */
        N = Dcs_chol.cs_chol(A, S); /* numeric Cholesky factorization */
        x = new double[n]; /* get workspace */
        ok = (S != null && N != null);
        if (ok) {
            Dcs_ipvec.cs_ipvec(S.pinv, b, x, n); /* x = P*b */
            Dcs_lsolve.cs_lsolve(N.L, x); /* x = L\x */
            Dcs_ltsolve.cs_ltsolve(N.L, x); /* x = L'\x */
            Dcs_pvec.cs_pvec(S.pinv, x, b, n); /* b = P'*x */
        }
        return (ok);
    }

}
