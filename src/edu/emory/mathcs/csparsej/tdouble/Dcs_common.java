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
 * Common data structures.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Dcs_common {

    public static final int CS_VER = 1; /* CSparseJ Version 1.0.0 */
    public static final int CS_SUBVER = 0;
    public static final int CS_SUBSUB = 0;
    public static final String CS_DATE = "June 13, 2009"; /* CSparseJ release date */
    public static final String CS_COPYRIGHT = "Copyright (c) Timothy A. Davis, 2006-2009";

    /**
     * 
     * Matrix in compressed-column or triplet form.
     * 
     */
    public static class Dcs {

        /**
         * maximum number of entries
         */
        public int nzmax;

        /**
         * number of rows
         */
        public int m;

        /**
         * number of columns
         */
        public int n;

        /**
         * column pointers (size n+1) or col indices (size nzmax)
         */
        public int[] p;

        /**
         * row indices, size nzmax
         */
        public int[] i;

        /**
         * numerical values, size nzmax
         */
        public double[] x;

        /**
         * # of entries in triplet matrix, -1 for compressed-col
         */
        public int nz;

        public Dcs() {

        }

    };

    /**
     * 
     * Output of symbolic Cholesky, LU, or QR analysis.
     * 
     */
    public static class Dcss {
        /**
         * inverse row perm. for QR, fill red. perm for Chol
         */
        public int[] pinv;

        /**
         * fill-reducing column permutation for LU and QR
         */
        public int[] q;

        /**
         * elimination tree for Cholesky and QR
         */
        public int[] parent;

        /**
         * column pointers for Cholesky, row counts for QR
         */
        public int[] cp;

        /**
         * leftmost[i] = min(find(A(i,:))), for QR
         */
        public int[] leftmost;

        /**
         * # of rows for QR, after adding fictitious rows
         */
        public int m2;

        /**
         * # entries in L for LU or Cholesky; in V for QR
         */
        public int lnz;

        /**
         * # entries in U for LU; in R for QR
         */
        public int unz;

        public Dcss() {
        }
    };

    /**
     * 
     * Output of numeric Cholesky, LU, or QR factorization
     * 
     */
    public static class Dcsn {
        /**
         * L for LU and Cholesky, V for QR
         */
        public Dcs L;

        /**
         * U for LU, R for QR, not used for Cholesky
         */
        public Dcs U;

        /**
         * partial pivoting for LU
         */
        public int[] pinv;

        /**
         * beta [0..n-1] for QR
         */
        public double[] B;

        public Dcsn() {
        }

    };

    /**
     * 
     * Output of Dulmage-Mendelsohn decomposition.
     * 
     */
    public static class Dcsd {

        /**
         * size m, row permutation
         */
        public int[] p;

        /**
         * size n, column permutation
         */
        public int[] q;

        /**
         * size nb+1, block k is rows r[k] to r[k+1]-1 in A(p,q)
         */
        public int[] r;

        /**
         * size nb+1, block k is cols s[k] to s[k+1]-1 in A(p,q)
         */
        public int[] s;

        /**
         * # of blocks in fine dmperm decomposition
         */
        public int nb;

        /**
         * coarse row decomposition
         */
        public int[] rr;

        /**
         * coarse column decomposition
         */
        public int[] cc;
    };
}
