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

/**
 * Permutes a vector, x=P*b.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Dcs_pvec {

    /**
     * Permutes a vector, x=P*b, for dense vectors x and b.
     * 
     * @param p
     *            permutation vector, p=null denotes identity
     * @param b
     *            input vector
     * @param x
     *            output vector, x=P*b
     * @param n
     *            length of p, b and x
     * @return true if successful, false otherwise
     */
    public static boolean cs_pvec(int[] p, double[] b, double[] x, int n) {
        int k;
        if (x == null || b == null)
            return (false); /* check inputs */
        for (k = 0; k < n; k++)
            x[k] = b[p != null ? p[k] : k];
        return (true);
    }

}
