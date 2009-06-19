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

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import edu.emory.mathcs.csparsej.tfloat.Scs_common.Scs;

/**
 * Load a sparse matrix from a file.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class Scs_load {

    /**
     * Loads a triplet matrix T from a file. Each line of the file contains
     * three values: a row index i, a column index j, and a numerical value aij.
     * The file is zero-based.
     * 
     * @param fileName
     *            file name
     * @return T if successful, null on error
     */
    public static Scs cs_load(String fileName) {
        int i, j;
        float x;
        Scs T;
        BufferedReader in;
        try {
            in = new BufferedReader(new FileReader(fileName));
        } catch (FileNotFoundException e1) {
            return (null);
        }
        T = Scs_util.cs_spalloc(0, 0, 1, true, true); /* allocate result */
        String line;
        try {
            while ((line = in.readLine()) != null) {
                String[] tokens = line.trim().split("\\s+");
                if (tokens.length != 3) {
                    return null;
                }
                i = Integer.parseInt(tokens[0]);
                j = Integer.parseInt(tokens[1]);
                x = Float.parseFloat(tokens[2]);
                if (!Scs_entry.cs_entry(T, i, j, x))
                    return (null);
            }
        } catch (IOException e) {
            return (null);
        }
        return (T);
    }
}
