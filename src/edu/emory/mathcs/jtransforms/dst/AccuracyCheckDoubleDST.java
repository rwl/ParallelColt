/* ***** BEGIN LICENSE BLOCK *****
 * Version: MPL 1.1/GPL 2.0/LGPL 2.1
 *
 * The contents of this file are subject to the Mozilla Public License Version
 * 1.1 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 * http://www.mozilla.org/MPL/
 *
 * Software distributed under the License is distributed on an "AS IS" basis,
 * WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
 * for the specific language governing rights and limitations under the
 * License.
 *
 * The Original Code is JTransforms.
 *
 * The Initial Developer of the Original Code is
 * Piotr Wendykier, Emory University.
 * Portions created by the Initial Developer are Copyright (C) 2007
 * the Initial Developer. All Rights Reserved.
 *
 * Alternatively, the contents of this file may be used under the terms of
 * either the GNU General Public License Version 2 or later (the "GPL"), or
 * the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
 * in which case the provisions of the GPL or the LGPL are applicable instead
 * of those above. If you wish to allow use of your version of this file only
 * under the terms of either the GPL or the LGPL, and not to allow others to
 * use your version of this file under the terms of the MPL, indicate your
 * decision by deleting the provisions above and replace them with the notice
 * and other provisions required by the GPL or the LGPL. If you do not delete
 * the provisions above, a recipient may use your version of this file under
 * the terms of any one of the MPL, the GPL or the LGPL.
 *
 * ***** END LICENSE BLOCK ***** */

package edu.emory.mathcs.jtransforms.dst;

import edu.emory.mathcs.utils.IOUtils;

/**
 * Accuracy check of double precision DST's
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class AccuracyCheckDoubleDST {

    private AccuracyCheckDoubleDST() {

    }

    public static void checkAccuracyDST_1D(int init_exp, int iters) {
        System.out.println("Checking accuracy of 1D DST...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            DoubleDST_1D dst = new DoubleDST_1D(size);
            double e, err = 0;
            double[] a = new double[size];
            IOUtils.fillMatrix_1D(size, a);
            double[] b = new double[size];
            IOUtils.fillMatrix_1D(size, b);
            dst.forward(a, true);
            dst.inverse(a, true);
            for (int j = 0; j < size; j++) {
                e = Math.abs(b[j] - a[j]);
                err = Math.max(err, e);
            }
            if (err > 1e-10) {
                System.err.println("\tsize = 2^" + exp + ";\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + ";\terror = " + err);
            }
            a = null;
            b = null;
            dst = null;
            System.gc();
        }
    }

    public static void checkAccuracyDST_2D(int init_exp, int iters) {
        System.out.println("Checking accuracy of 2D DST (double[] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            DoubleDST_2D dst2 = new DoubleDST_2D(size, size);
            double e, err = 0.0;
            double[] a = new double[size * size];
            IOUtils.fillMatrix_2D(size, size, a);
            double[] b = new double[size * size];
            IOUtils.fillMatrix_2D(size, size, b);
            dst2.forward(a, true);
            dst2.inverse(a, true);
            for (int j = 0; j < size * size; j++) {
                e = Math.abs(b[j] - a[j]);
                err = Math.max(err, e);
            }
            if (err > 1e-10) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            }
            a = null;
            b = null;
            dst2 = null;
            System.gc();
        }
        System.out.println("Checking accuracy of 2D DST (double[][] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            DoubleDST_2D dst2 = new DoubleDST_2D(size, size);
            double e, err = 0.0;
            double[][] a = new double[size][size];
            IOUtils.fillMatrix_2D(size, size, a);
            double[][] b = new double[size][size];
            IOUtils.fillMatrix_2D(size, size, b);
            dst2.forward(a, true);
            dst2.inverse(a, true);
            for (int r = 0; r < size; r++) {
                for (int c = 0; c < size; c++) {
                    e = Math.abs(b[r][c] - a[r][c]);
                    err = Math.max(err, e);
                }
            }
            if (err > 1e-10) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + ";\terror = " + err);
            }
            a = null;
            b = null;
            dst2 = null;
            System.gc();
        }

    }

    public static void checkAccuracyDST_3D(int init_exp, int iters) {
        System.out.println("Checking accuracy of 3D DST (double[] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            DoubleDST_3D dst3 = new DoubleDST_3D(size, size, size);
            double e, err = 0.0;
            double[] a = new double[size * size * size];
            IOUtils.fillMatrix_3D(size, size, size, a);
            double[] b = new double[size * size * size];
            IOUtils.fillMatrix_3D(size, size, size, b);
            dst3.forward(a, true);
            dst3.inverse(a, true);
            for (int j = 0; j < size * size * size; j++) {
                e = Math.abs(b[j] - a[j]);
                err = Math.max(err, e);
            }
            if (err > 1e-10) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            }
            a = null;
            b = null;
            dst3 = null;
            System.gc();
        }

        System.out.println("Checking accuracy of 3D DST (double[][][] input)...");
        for (int i = 0; i < iters; i++) {
            int exp = init_exp + i;
            int size = (int) Math.pow(2, exp);
            DoubleDST_3D dst3 = new DoubleDST_3D(size, size, size);
            double e, err = 0.0;
            double[][][] a = new double[size][size][size];
            IOUtils.fillMatrix_3D(size, size, size, a);
            double[][][] b = new double[size][size][size];
            IOUtils.fillMatrix_3D(size, size, size, b);
            dst3.forward(a, true);
            dst3.inverse(a, true);
            for (int s = 0; s < size; s++) {
                for (int r = 0; r < size; r++) {
                    for (int c = 0; c < size; c++) {
                        e = Math.abs(b[s][r][c] - a[s][r][c]);
                        err = Math.max(err, e);
                    }
                }
            }
            if (err > 1e-10) {
                System.err.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            } else {
                System.out.println("\tsize = 2^" + exp + " x 2^" + exp + " x 2^" + exp + ";\t\terror = " + err);
            }
            a = null;
            b = null;
            dst3 = null;
            System.gc();
        }
    }

    public static void main(String[] args) {
        checkAccuracyDST_1D(0, 21);
        checkAccuracyDST_2D(1, 11);
        checkAccuracyDST_3D(1, 7);
        System.exit(0);
    }
}
