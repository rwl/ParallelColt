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
import edu.emory.mathcs.csparsej.tfloat.Scs_common.Scsd;

/**
 * Various utilities.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Scs_util {

    /**
     * Allocate a sparse matrix (triplet form or compressed-column form).
     * 
     * @param m
     *            number of rows
     * @param n
     *            number of columns
     * @param nzmax
     *            maximum number of entries
     * @param values
     *            allocate pattern only if false, values and pattern otherwise
     * @param triplet
     *            compressed-column if false, triplet form otherwise
     * @return sparse matrix
     */
    public static Scs cs_spalloc(int m, int n, int nzmax, boolean values, boolean triplet) {
        Scs A = new Scs(); /* allocate the Scs struct */
        A.m = m; /* define dimensions and nzmax */
        A.n = n;
        A.nzmax = nzmax = Math.max(nzmax, 1);
        A.nz = triplet ? 0 : -1; /* allocate triplet or comp.col */
        A.p = triplet ? new int[nzmax] : new int[n + 1];
        A.i = new int[nzmax];
        A.x = values ? new float[nzmax] : null;
        return A;
    }

    /**
     * Change the max # of entries a sparse matrix can hold.
     * 
     * @param A
     *            column-compressed matrix
     * @param nzmax
     *            new maximum number of entries
     * @return true if successful, false on error
     */
    public static boolean cs_sprealloc(Scs A, int nzmax) {
        if (A == null)
            return (false);
        if (nzmax <= 0)
            nzmax = (Scs_util.CS_CSC(A)) ? (A.p[A.n]) : A.nz;
        int[] Ainew = new int[nzmax];
        int length = Math.min(nzmax, A.i.length);
        System.arraycopy(A.i, 0, Ainew, 0, length);
        A.i = Ainew;
        if (Scs_util.CS_TRIPLET(A)) {
            int[] Apnew = new int[nzmax];
            length = Math.min(nzmax, A.p.length);
            System.arraycopy(A.p, 0, Apnew, 0, length);
            A.p = Apnew;
        }
        if (A.x != null) {
            float[] Axnew = new float[nzmax];
            length = Math.min(nzmax, A.x.length);
            System.arraycopy(A.x, 0, Axnew, 0, length);
            A.x = Axnew;
        }
        A.nzmax = nzmax;
        return (true);
    }

    /**
     * Allocate a Scsd object (a Sulmage-Mendelsohn decomposition).
     * 
     * @param m
     *            number of rows of the matrix A to be analyzed
     * @param n
     *            number of columns of the matrix A to be analyzed
     * @return Sulmage-Mendelsohn decomposition
     */
    public static Scsd cs_dalloc(int m, int n) {
        Scsd S;
        S = new Scsd();
        S.p = new int[m];
        S.r = new int[m + 6];
        S.q = new int[n];
        S.s = new int[n + 6];
        S.cc = new int[5];
        S.rr = new int[5];
        return S;
    }

    protected static int CS_FLIP(int i) {
        return (-(i) - 2);
    }

    protected static int CS_UNFLIP(int i) {
        return (((i) < 0) ? CS_FLIP(i) : (i));
    }

    protected static boolean CS_MARKES(int[] w, int j) {
        return (w[j] < 0);
    }

    protected static void CS_MARK(int[] w, int j) {
        w[j] = CS_FLIP(w[j]);
    }

    /**
     * Returns true if A is in column-compressed form, false otherwise.
     * 
     * @param A
     *            sparse matrix
     * @return true if A is in column-compressed form, false otherwise
     */
    public static boolean CS_CSC(Scs A) {
        return (A != null && (A.nz == -1));
    }

    /**
     * Returns true if A is in triplet form, false otherwise.
     * 
     * @param A
     *            sparse matrix
     * @return true if A is in triplet form, false otherwise
     */
    public static boolean CS_TRIPLET(Scs A) {
        return (A != null && (A.nz >= 0));
    }
}
