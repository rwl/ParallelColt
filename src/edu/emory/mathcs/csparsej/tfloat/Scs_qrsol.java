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
import edu.emory.mathcs.csparsej.tfloat.Scs_common.Scsn;
import edu.emory.mathcs.csparsej.tfloat.Scs_common.Scss;

/**
 * Solve a least-squares or underdetermined problem.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Scs_qrsol {

    /**
     * Solve a least-squares problem (min ||Ax-b||_2, where A is m-by-n with m
     * >= n) or underdetermined system (Ax=b, where m < n)
     * 
     * @param order
     *            ordering method to use (0 to 3)
     * @param A
     *            column-compressed matrix
     * @param b
     *            size max(m,n), b (size m) on input, x(size n) on output
     * @return true if successful, false on error
     */
    public static boolean cs_qrsol(int order, Scs A, float[] b) {
        float x[];
        Scss S;
        Scsn N;
        Scs AT = null;
        int k, m, n;
        boolean ok;
        if (!Scs_util.CS_CSC(A) || b == null)
            return (false); /* check inputs */
        n = A.n;
        m = A.m;
        if (m >= n) {
            S = Scs_sqr.cs_sqr(order, A, true); /* ordering and symbolic analysis */
            N = Scs_qr.cs_qr(A, S); /* numeric QR factorization */
            x = new float[S != null ? S.m2 : 1]; /* get workspace */
            ok = (S != null && N != null);
            if (ok) {
                Scs_ipvec.cs_ipvec(S.pinv, b, x, m); /* x(0:m-1) = b(p(0:m-1) */
                for (k = 0; k < n; k++) /* apply Householder refl. to x */
                {
                    Scs_happly.cs_happly(N.L, k, N.B[k], x);
                }
                Scs_usolve.cs_usolve(N.U, x); /* x = R\x */
                Scs_ipvec.cs_ipvec(S.q, x, b, n); /* b(q(0:n-1)) = x(0:n-1) */
            }
        } else {
            AT = Scs_transpose.cs_transpose(A, true); /* Ax=b is underdetermined */
            S = Scs_sqr.cs_sqr(order, AT, true); /* ordering and symbolic analysis */
            N = Scs_qr.cs_qr(AT, S); /* numeric QR factorization of A' */
            x = new float[S != null ? S.m2 : 1]; /* get workspace */
            ok = (AT != null && S != null && N != null);
            if (ok) {
                Scs_pvec.cs_pvec(S.q, b, x, m); /* x(q(0:m-1)) = b(0:m-1) */
                Scs_utsolve.cs_utsolve(N.U, x); /* x = R'\x */
                for (k = m - 1; k >= 0; k--) /* apply Householder refl. to x */
                {
                    Scs_happly.cs_happly(N.L, k, N.B[k], x);
                }
                Scs_pvec.cs_pvec(S.pinv, x, b, n); /* b(0:n-1) = x(p(0:n-1)) */
            }
        }
        return (ok);
    }
}
