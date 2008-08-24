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
 * The Original Code is Parallel Colt.
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
package edu.emory.mathcs.utils;

import static org.junit.Assert.assertEquals;

import org.junit.Assert;

/**
 * Utility methods.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class AssertUtils {

    public static void assertArrayEquals(double[] a, double[] b, double tol) {
        if (a.length != b.length)
            Assert.fail("a.length != b.length");
        for (int i = 0; i < a.length; i++) {
            assertEquals(a[i], b[i], tol);
        }
    }

    public static void assertArrayEquals(double[][] a, double[][] b, double tol) {
        if ((a.length != b.length) || (a[0].length != b[0].length))
            Assert.fail("a.length != b.length");
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                assertEquals(a[i][j], b[i][j], tol);
            }
        }
    }

    public static void assertArrayEquals(double[][][] a, double[][][] b, double tol) {
        if ((a.length != b.length) || (a[0].length != b[0].length) || (a[0][0].length != b[0][0].length))
            Assert.fail("a.length != b.length");
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    assertEquals(a[i][j][k], b[i][j][k], tol);
                }
            }
        }
    }

    public static void assertArrayEquals(float[] a, float[] b, float tol) {
        if (a.length != b.length)
            Assert.fail("a.length != b.length");
        for (int i = 0; i < a.length; i++) {
            assertEquals(a[i], b[i], tol);
        }
    }

    public static void assertArrayEquals(float[][] a, float[][] b, float tol) {
        if ((a.length != b.length) || (a[0].length != b[0].length))
            Assert.fail("a.length != b.length");
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                assertEquals(a[i][j], b[i][j], tol);
            }
        }
    }

    public static void assertArrayEquals(float[][][] a, float[][][] b, float tol) {
        if ((a.length != b.length) || (a[0].length != b[0].length) || (a[0][0].length != b[0][0].length))
            Assert.fail("a.length != b.length");
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                for (int k = 0; k < a[0][0].length; k++) {
                    assertEquals(a[i][j][k], b[i][j][k], tol);
                }
            }
        }
    }

}
