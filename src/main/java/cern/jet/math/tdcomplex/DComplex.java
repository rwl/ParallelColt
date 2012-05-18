/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.jet.math.tdcomplex;

import cern.jet.math.tdouble.DoubleConstants;

/**
 * Complex arithmetic
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class DComplex extends DoubleConstants {

    public static final double abs(double[] x) {
        double absX = Math.abs(x[0]);
        double absY = Math.abs(x[1]);
        if (absX == 0.0 && absY == 0.0) {
            return 0.0;
        } else if (absX >= absY) {
            double d = x[1] / x[0];
            return absX * Math.sqrt(1.0 + d * d);
        } else {
            double d = x[0] / x[1];
            return absY * Math.sqrt(1.0 + d * d);
        }
    }

    public static final double abs(double re, double im) {
        double absX = Math.abs(re);
        double absY = Math.abs(im);
        if (absX == 0.0 && absY == 0.0) {
            return 0.0;
        } else if (absX >= absY) {
            double d = im / re;
            return absX * Math.sqrt(1.0 + d * d);
        } else {
            double d = re / im;
            return absY * Math.sqrt(1.0 + d * d);
        }
    }

    public static final double[] acos(double[] x) {
        double[] z = new double[2];

        double re, im;

        re = 1.0 - ((x[0] * x[0]) - (x[1] * x[1]));
        im = -((x[0] * x[1]) + (x[1] * x[0]));

        z[0] = re;
        z[1] = im;
        z = sqrt(z);

        re = -z[1];
        im = z[0];

        z[0] = x[0] + re;
        z[1] = x[1] + im;

        re = Math.log(abs(z));
        im = Math.atan2(z[1], z[0]);

        z[0] = im;
        z[1] = -re;
        return z;
    }

    public static final double arg(double[] x) {
        return Math.atan2(x[1], x[0]);
    }

    public static final double arg(double re, double im) {
        return Math.atan2(im, re);
    }

    public static final double[] asin(double[] x) {
        double[] z = new double[2];

        double re, im;

        re = 1.0 - ((x[0] * x[0]) - (x[1] * x[1]));
        im = -((x[0] * x[1]) + (x[1] * x[0]));

        z[0] = re;
        z[1] = im;
        z = sqrt(z);

        re = -z[1];
        im = z[0];

        z[0] = z[0] + re;
        z[1] = z[1] + im;

        re = Math.log(abs(z));
        im = Math.atan2(z[1], z[0]);

        z[0] = im;
        z[1] = -re;
        return z;
    }

    public static final double[] atan(double[] x) {
        double[] z = new double[2];

        double re, im;

        z[0] = -x[0];
        z[1] = 1.0 - x[1];

        re = x[0];
        im = 1.0 + x[1];

        z = div(z, re, im);

        re = Math.log(abs(z));
        im = Math.atan2(z[1], z[0]);

        z[0] = 0.5 * im;
        z[1] = -0.5 * re;

        return z;
    }

    public static final double[] conj(double[] x) {
        double[] z = new double[2];
        z[0] = x[0];
        z[1] = -x[1];
        return z;
    }

    public static final double[] cos(double[] x) {
        double[] z = new double[2];

        double re1, im1, re2, im2;
        double scalar;
        double iz_re, iz_im;

        iz_re = -x[1];
        iz_im = x[0];

        scalar = Math.exp(iz_re);
        re1 = scalar * Math.cos(iz_im);
        im1 = scalar * Math.sin(iz_im);

        scalar = Math.exp(-iz_re);
        re2 = scalar * Math.cos(-iz_im);
        im2 = scalar * Math.sin(-iz_im);

        re1 = re1 + re2;
        im1 = im1 + im2;

        z[0] = 0.5 * re1;
        z[1] = 0.5 * im1;

        return z;
    }

    public static final double[] div(double[] x, double re, double im) {
        double[] z = new double[2];
        double scalar;

        if (Math.abs(re) >= Math.abs(im)) {
            scalar = 1.0 / (re + im * (im / re));

            z[0] = scalar * (x[0] + x[1] * (im / re));
            z[1] = scalar * (x[1] - x[0] * (im / re));

        } else {
            scalar = 1.0 / (re * (re / im) + im);

            z[0] = scalar * (x[0] * (re / im) + x[1]);
            z[1] = scalar * (x[1] * (re / im) - x[0]);
        }

        return z;
    }

    public static final double[] div(double[] x, double[] y) {
        return div(x, y[0], y[1]);
    }

    public static final double equals(double[] x, double[] y, double tol) {
        if (abs(x[0] - y[0], x[1] - y[1]) <= Math.abs(tol)) {
            return 1;
        } else {
            return 0;
        }
    }

    public static final boolean isEqual(double[] x, double[] y, double tol) {
        if (abs(x[0] - y[0], x[1] - y[1]) <= Math.abs(tol)) {
            return true;
        } else {
            return false;
        }
    }

    public static final double[] exp(double[] x) {
        double[] z = new double[2];
        double scalar = Math.exp(x[0]);
        z[0] = scalar * Math.cos(x[1]);
        z[1] = scalar * Math.sin(x[1]);
        return z;
    }

    public static final double[] inv(double[] x) {
        double[] z = new double[2];
        if (x[1] != 0.0) {
            double scalar;
            if (Math.abs(x[0]) >= Math.abs(x[1])) {
                scalar = 1.0 / (x[0] + x[1] * (x[1] / x[0]));
                z[0] = scalar;
                z[1] = scalar * (-x[1] / x[0]);
            } else {
                scalar = 1.0 / (x[0] * (x[0] / x[1]) + x[1]);
                z[0] = scalar * (x[0] / x[1]);
                z[1] = -scalar;
            }
        } else {
            z[0] = 1 / x[0];
            z[1] = 0;
        }
        return z;
    }

    public static final double[] log(double[] x) {
        double[] z = new double[2];
        z[0] = Math.log(abs(x));
        z[1] = arg(x);
        return z;
    }

    public static final double[] minus(double[] x, double[] y) {
        double[] z = new double[2];
        z[0] = x[0] - y[0];
        z[1] = x[1] - y[1];
        return z;
    }

    public static final double[] minusAbs(double[] x, double[] y) {
        double[] z = new double[2];
        z[0] = Math.abs(x[0] - y[0]);
        z[1] = Math.abs(x[1] - y[1]);
        return z;
    }

    public static final double[] mult(double[] x, double y) {
        double[] z = new double[2];
        z[0] = x[0] * y;
        z[1] = x[1] * y;
        return z;
    }

    public static final double[] mult(double[] x, double[] y) {
        double[] z = new double[2];
        z[0] = x[0] * y[0] - x[1] * y[1];
        z[1] = x[1] * y[0] + x[0] * y[1];
        return z;
    }

    public static final double[] neg(double[] x) {
        double[] z = new double[2];
        z[0] = -x[0];
        z[1] = -x[1];
        return z;
    }

    public static final double[] plus(double[] x, double[] y) {
        double[] z = new double[2];
        z[0] = x[0] + y[0];
        z[1] = x[1] + y[1];
        return z;
    }

    public static final double[] pow(double[] x, double y) {
        double[] z = new double[2];
        double re = y * Math.log(abs(x));
        double im = y * arg(x);
        double scalar = Math.exp(re);
        z[0] = scalar * Math.cos(im);
        z[1] = scalar * Math.sin(im);
        return z;
    }

    public static final double[] pow(double x, double[] y) {
        double[] z = new double[2];
        double re = Math.log(Math.abs(x));
        double im = Math.atan2(0.0, x);

        double re2 = (re * y[0]) - (im * y[1]);
        double im2 = (re * y[1]) + (im * y[0]);

        double scalar = Math.exp(re2);

        z[0] = scalar * Math.cos(im2);
        z[1] = scalar * Math.sin(im2);
        return z;
    }

    public static final double[] pow(double[] x, double[] y) {
        double[] z = new double[2];
        double re = Math.log(abs(x));
        double im = arg(x);

        double re2 = (re * y[0]) - (im * y[1]);
        double im2 = (re * y[1]) + (im * y[0]);

        double scalar = Math.exp(re2);

        z[0] = scalar * Math.cos(im2);
        z[1] = scalar * Math.sin(im2);
        return z;
    }

    public static final double[] sin(double[] x) {
        double[] z = new double[2];
        double re1, im1, re2, im2;
        double scalar;
        double iz_re, iz_im;

        iz_re = -x[1];
        iz_im = x[0];

        scalar = Math.exp(iz_re);
        re1 = scalar * Math.cos(iz_im);
        im1 = scalar * Math.sin(iz_im);

        scalar = Math.exp(-iz_re);
        re2 = scalar * Math.cos(-iz_im);
        im2 = scalar * Math.sin(-iz_im);

        re1 = re1 - re2;
        im1 = im1 - im2;

        z[0] = 0.5 * im1;
        z[1] = -0.5 * re1;

        return z;
    }

    public static final double[] sqrt(double[] x) {
        double[] z = new double[2];
        double absx = abs(x);
        double tmp;
        if (absx > 0.0) {
            if (x[0] > 0.0) {
                tmp = Math.sqrt(0.5 * (absx + x[0]));
                z[0] = tmp;
                z[1] = 0.5 * (x[1] / tmp);
            } else {
                tmp = Math.sqrt(0.5 * (absx - x[0]));
                if (x[1] < 0.0) {
                    tmp = -tmp;
                }
                z[0] = 0.5 * (x[1] / tmp);
                z[1] = tmp;
            }
        } else {
            z[0] = 0.0;
            z[1] = 0.0;
        }
        return z;
    }

    public static final double[] square(double[] x) {
        return mult(x, x);
    }

    public static final double[] tan(double[] x) {
        double[] z = new double[2];
        double scalar;
        double iz_re, iz_im;
        double re1, im1, re2, im2, re3, im3;
        double cs_re, cs_im;

        iz_re = -x[1];
        iz_im = x[0];

        scalar = Math.exp(iz_re);
        re1 = scalar * Math.cos(iz_im);
        im1 = scalar * Math.sin(iz_im);

        scalar = Math.exp(-iz_re);
        re2 = scalar * Math.cos(-iz_im);
        im2 = scalar * Math.sin(-iz_im);

        re3 = re1 - re2;
        im3 = im1 - im2;

        z[0] = 0.5 * im3;
        z[1] = -0.5 * re3;

        re3 = re1 + re2;
        im3 = im1 + im2;

        cs_re = 0.5 * re3;
        cs_im = 0.5 * im3;

        z = div(z, cs_re, cs_im);

        return z;
    }

    /**
     * Makes this class non instantiable, but still let's others inherit from
     * it.
     */
    protected DComplex() {
    }

}
