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

import java.util.Random;

/**
 * Random permutation.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Dcs_randperm {

    /**
     * Returns a random permutation vector, the identity perm, or p = n-1:-1:0.
     * seed = -1 means p = n-1:-1:0. seed = 0 means p = identity. otherwise p =
     * random permutation.
     * 
     * @param n
     *            length of p
     * @param seed
     *            0: natural, -1: reverse, random p oterwise
     * @return p, null on error or for natural order
     */
    public static int[] cs_randperm(int n, int seed) {
        int p[], k, j, t;
        if (seed == 0)
            return (null); /* return p = NULL (identity) */
        p = new int[n]; /* allocate result */
        for (k = 0; k < n; k++)
            p[k] = n - k - 1;
        if (seed == -1)
            return (p); /* return reverse permutation */
        Random r = new Random(seed); /* get new random number seed */
        for (k = 0; k < n; k++) {
            j = k + r.nextInt(n - k); /* j = rand int in range k to n-1 */
            t = p[j]; /* swap p[k] and p[j] */
            p[j] = p[k];
            p[k] = t;
        }
        return (p);
    }

}
