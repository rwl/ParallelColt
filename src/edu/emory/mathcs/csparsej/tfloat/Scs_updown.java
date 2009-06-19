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
 * Sparse rank-1 Cholesky update/downate.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Scs_updown {

    /**
     * Sparse Cholesky rank-1 update/downdate, L*L' + sigma*w*w' (sigma = +1 or
     * -1)
     * 
     * @param L
     *            factorization to update/downdate
     * @param sigma
     *            +1 for update, -1 for downdate
     * @param C
     *            the vector c
     * @param parent
     *            the elimination tree of L
     * @return true if successful, false on error
     */
    public static boolean cs_updown(Scs L, int sigma, Scs C, int[] parent) {
        int n, p, f, j, Lp[], Li[], Cp[], Ci[];
        float Lx[], Cx[], alpha, beta = 1, delta, gamma, w1, w2, w[], beta2 = 1;
        if (!Scs_util.CS_CSC(L) || !Scs_util.CS_CSC(C) || parent == null)
            return (false); /* check inputs */
        Lp = L.p;
        Li = L.i;
        Lx = L.x;
        n = L.n;
        Cp = C.p;
        Ci = C.i;
        Cx = C.x;
        if ((p = Cp[0]) >= Cp[1])
            return (true); /* return if C empty */
        w = new float[n]; /* get workspace */
        f = Ci[p];
        for (; p < Cp[1]; p++)
            f = Math.min(f, Ci[p]); /* f = min (find (C)) */
        for (j = f; j != -1; j = parent[j])
            w[j] = 0; /* clear workspace w */
        for (p = Cp[0]; p < Cp[1]; p++)
            w[Ci[p]] = Cx[p]; /* w = C */
        for (j = f; j != -1; j = parent[j]) /* walk path f up to root */
        {
            p = Lp[j];
            alpha = w[j] / Lx[p]; /* alpha = w(j) / L(j,j) */
            beta2 = beta * beta + sigma * alpha * alpha;
            if (beta2 <= 0)
                break; /* not positive definite */
            beta2 = (float) Math.sqrt(beta2);
            delta = (sigma > 0) ? (beta / beta2) : (beta2 / beta);
            gamma = sigma * alpha / (beta2 * beta);
            Lx[p] = delta * Lx[p] + ((sigma > 0) ? (gamma * w[j]) : 0);
            beta = beta2;
            for (p++; p < Lp[j + 1]; p++) {
                w1 = w[Li[p]];
                w[Li[p]] = w2 = w1 - alpha * Lx[p];
                Lx[p] = delta * Lx[p] + gamma * ((sigma > 0) ? w1 : w2);
            }
        }
        return (beta2 > 0);
    }

}
