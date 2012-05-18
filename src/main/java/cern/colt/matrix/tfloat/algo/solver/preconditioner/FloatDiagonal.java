/*
 * Copyright (C) 2003-2006 Bjørn-Ove Heimsund
 * 
 * This file is part of MTJ.
 * 
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation; either version 2.1 of the License, or (at your
 * option) any later version.
 * 
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
 * for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

package cern.colt.matrix.tfloat.algo.solver.preconditioner;

import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix1D;

/**
 * Diagonal preconditioner. Uses the inverse of the diagonal as preconditioner
 */
public class FloatDiagonal implements FloatPreconditioner {

    /**
     * This contains the inverse of the diagonal
     */
    private float[] invdiag;

    /**
     * Constructor for DiagonalPreconditioner
     * 
     * @param n
     *            Problem size (number of rows)
     */
    public FloatDiagonal(int n) {
        invdiag = new float[n];
    }

    public FloatMatrix1D apply(FloatMatrix1D b, FloatMatrix1D x) {
        if (x == null) {
            x = b.like();
        }

        if (!(x instanceof DenseFloatMatrix1D) || !(b instanceof DenseFloatMatrix1D))
            throw new IllegalArgumentException("a nad b must be dense vectors");

        float[] xd = ((DenseFloatMatrix1D) x).elements();
        float[] bd = ((DenseFloatMatrix1D) b).elements();

        for (int i = 0; i < invdiag.length; ++i)
            xd[i] = bd[i] * invdiag[i];

        return x;
    }

    public FloatMatrix1D transApply(FloatMatrix1D b, FloatMatrix1D x) {
        if (x == null) {
            x = b.like();
        }

        return apply(b, x);
    }

    public void setMatrix(FloatMatrix2D A) {
        if (A.rows() != invdiag.length)
            throw new IllegalArgumentException("Matrix size differs from preconditioner size");

        for (int i = 0; i < invdiag.length; ++i) {
            invdiag[i] = A.getQuick(i, i);
            if (invdiag[i] == 0) // Avoid zero-division
                throw new RuntimeException("Zero diagonal on row " + (i + 1));
            else
                invdiag[i] = 1 / invdiag[i];
        }
    }

}
