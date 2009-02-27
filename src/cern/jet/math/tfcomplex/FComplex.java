/*
Copyright (C) 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
 */
package cern.jet.math.tfcomplex;

import cern.jet.math.tfloat.FloatConstants;

/**
 * Complex arithmetic
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 * 
 */
public class FComplex extends FloatConstants {

    public static final float abs(float[] x) {
        float absX = Math.abs(x[0]);
        float absY = Math.abs(x[1]);
        if (absX == 0.0 && absY == 0.0) {
            return 0.0f;
        } else if (absX >= absY) {
            float d = x[1] / x[0];
            return (float) (absX * Math.sqrt(1.0 + d * d));
        } else {
            float d = x[0] / x[1];
            return (float) (absY * Math.sqrt(1.0 + d * d));
        }
    }

    public static final float abs(float re, float im) {
        float absX = Math.abs(re);
        float absY = Math.abs(im);
        if (absX == 0.0 && absY == 0.0) {
            return 0.0f;
        } else if (absX >= absY) {
            float d = im / re;
            return (float) (absX * Math.sqrt(1.0 + d * d));
        } else {
            float d = re / im;
            return (float) (absY * Math.sqrt(1.0 + d * d));
        }
    }

    public static final float[] acos(float[] x) {
        float[] z = new float[2];

        float re, im;

        re = (float) (1.0 - ((x[0] * x[0]) - (x[1] * x[1])));
        im = -((x[0] * x[1]) + (x[1] * x[0]));

        z[0] = re;
        z[1] = im;
        z = sqrt(z);

        re = -z[1];
        im = z[0];

        z[0] = x[0] + re;
        z[1] = x[1] + im;

        re = (float) Math.log(abs(z));
        im = (float) Math.atan2(z[1], z[0]);

        z[0] = im;
        z[1] = -re;
        return z;
    }

    public static final float arg(float[] x) {
        return (float) Math.atan2(x[1], x[0]);
    }

    public static final float arg(float re, float im) {
        return (float) Math.atan2(im, re);
    }

    public static final float[] asin(float[] x) {
        float[] z = new float[2];

        float re, im;

        re = (float) (1.0 - ((x[0] * x[0]) - (x[1] * x[1])));
        im = -((x[0] * x[1]) + (x[1] * x[0]));

        z[0] = re;
        z[1] = im;
        z = sqrt(z);

        re = -z[1];
        im = z[0];

        z[0] = z[0] + re;
        z[1] = z[1] + im;

        re = (float) Math.log(abs(z));
        im = (float) Math.atan2(z[1], z[0]);

        z[0] = im;
        z[1] = -re;
        return z;
    }

    public static final float[] atan(float[] x) {
        float[] z = new float[2];

        float re, im;

        z[0] = -x[0];
        z[1] = 1.0f - x[1];

        re = x[0];
        im = 1.0f + x[1];

        z = div(z, re, im);

        re = (float) Math.log(abs(z));
        im = (float) Math.atan2(z[1], z[0]);

        z[0] = 0.5f * im;
        z[1] = -0.5f * re;

        return z;
    }

    public static final float[] conj(float[] x) {
        float[] z = new float[2];
        z[0] = x[0];
        z[1] = -x[1];
        return z;
    }

    public static final float[] cos(float[] x) {
        float[] z = new float[2];

        float re1, im1, re2, im2;
        float scalar;
        float iz_re, iz_im;

        iz_re = -x[1];
        iz_im = x[0];

        scalar = (float) Math.exp(iz_re);
        re1 = (float) (scalar * Math.cos(iz_im));
        im1 = (float) (scalar * Math.sin(iz_im));

        scalar = (float) Math.exp(-iz_re);
        re2 = (float) (scalar * Math.cos(-iz_im));
        im2 = (float) (scalar * Math.sin(-iz_im));

        re1 = re1 + re2;
        im1 = im1 + im2;

        z[0] = 0.5f * re1;
        z[1] = 0.5f * im1;

        return z;
    }

    public static final float[] div(float[] x, float re, float im) {
        float[] z = new float[2];
        float scalar;

        if (Math.abs(re) >= Math.abs(im)) {
            scalar = (float) (1.0 / (re + im * (im / re)));

            z[0] = scalar * (x[0] + x[1] * (im / re));
            z[1] = scalar * (x[1] - x[0] * (im / re));

        } else {
            scalar = (float) (1.0 / (re * (re / im) + im));

            z[0] = scalar * (x[0] * (re / im) + x[1]);
            z[1] = scalar * (x[1] * (re / im) - x[0]);
        }

        return z;
    }

    public static final float[] div(float[] x, float[] y) {
        return div(x, y[0], y[1]);
    }

    public static final float equals(float[] x, float[] y, float tol) {
        if (abs(x[0] - y[0], x[1] - y[1]) <= Math.abs(tol)) {
            return 1;
        } else {
            return 0;
        }
    }

    public static final boolean isEqual(float[] x, float[] y, float tol) {
        if (abs(x[0] - y[0], x[1] - y[1]) <= Math.abs(tol)) {
            return true;
        } else {
            return false;
        }
    }

    public static final float[] exp(float[] x) {
        float[] z = new float[2];
        float scalar = (float) Math.exp(x[0]);
        z[0] = (float) (scalar * Math.cos(x[1]));
        z[1] = (float) (scalar * Math.sin(x[1]));
        return z;
    }

    public static final float[] inv(float[] x) {
        float[] z = new float[2];
        if (x[1] != 0.0) {
            float scalar;
            if (Math.abs(x[0]) >= Math.abs(x[1])) {
                scalar = (float) (1.0 / (x[0] + x[1] * (x[1] / x[0])));
                z[0] = scalar;
                z[1] = scalar * (-x[1] / x[0]);
            } else {
                scalar = (float) (1.0 / (x[0] * (x[0] / x[1]) + x[1]));
                z[0] = scalar * (x[0] / x[1]);
                z[1] = -scalar;
            }
        } else {
            z[0] = 1 / x[0];
            z[1] = 0;
        }
        return z;
    }

    public static final float[] log(float[] x) {
        float[] z = new float[2];
        z[0] = (float) Math.log(abs(x));
        z[1] = arg(x);
        return z;
    }

    public static final float[] minus(float[] x, float[] y) {
        float[] z = new float[2];
        z[0] = x[0] - y[0];
        z[1] = x[1] - y[1];
        return z;
    }

    public static final float[] minusAbs(float[] x, float[] y) {
        float[] z = new float[2];
        z[0] = Math.abs(x[0] - y[0]);
        z[1] = Math.abs(x[1] - y[1]);
        return z;
    }

    public static final float[] mult(float[] x, float y) {
        float[] z = new float[2];
        z[0] = x[0] * y;
        z[1] = x[1] * y;
        return z;
    }

    public static final float[] mult(float[] x, float[] y) {
        float[] z = new float[2];
        z[0] = x[0] * y[0] - x[1] * y[1];
        z[1] = x[1] * y[0] + x[0] * y[1];
        return z;
    }

    public static final float[] neg(float[] x) {
        float[] z = new float[2];
        z[0] = -x[0];
        z[1] = -x[1];
        return z;
    }

    public static final float[] plus(float[] x, float[] y) {
        float[] z = new float[2];
        z[0] = x[0] + y[0];
        z[1] = x[1] + y[1];
        return z;
    }

    public static final float[] pow(float[] x, float y) {
        float[] z = new float[2];
        float re = (float) (y * Math.log(abs(x)));
        float im = y * arg(x);
        float scalar = (float) Math.exp(re);
        z[0] = (float) (scalar * Math.cos(im));
        z[1] = (float) (scalar * Math.sin(im));
        return z;
    }

    public static final float[] pow(float x, float[] y) {
        float[] z = new float[2];
        float re = (float) Math.log(Math.abs(x));
        float im = (float) Math.atan2(0.0, x);

        float re2 = (re * y[0]) - (im * y[1]);
        float im2 = (re * y[1]) + (im * y[0]);

        float scalar = (float) Math.exp(re2);

        z[0] = (float) (scalar * Math.cos(im2));
        z[1] = (float) (scalar * Math.sin(im2));
        return z;
    }

    public static final float[] pow(float[] x, float[] y) {
        float[] z = new float[2];
        float re = (float) Math.log(abs(x));
        float im = arg(x);

        float re2 = (re * y[0]) - (im * y[1]);
        float im2 = (re * y[1]) + (im * y[0]);

        float scalar = (float) Math.exp(re2);

        z[0] = (float) (scalar * Math.cos(im2));
        z[1] = (float) (scalar * Math.sin(im2));
        return z;
    }

    public static final float[] sin(float[] x) {
        float[] z = new float[2];
        float re1, im1, re2, im2;
        float scalar;
        float iz_re, iz_im;

        iz_re = -x[1];
        iz_im = x[0];

        scalar = (float) Math.exp(iz_re);
        re1 = (float) (scalar * Math.cos(iz_im));
        im1 = (float) (scalar * Math.sin(iz_im));

        scalar = (float) Math.exp(-iz_re);
        re2 = (float) (scalar * Math.cos(-iz_im));
        im2 = (float) (scalar * Math.sin(-iz_im));

        re1 = re1 - re2;
        im1 = im1 - im2;

        z[0] = 0.5f * im1;
        z[1] = -0.5f * re1;

        return z;
    }

    public static final float[] sqrt(float[] x) {
        float[] z = new float[2];
        float absx = abs(x);
        float tmp;
        if (absx > 0.0) {
            if (x[0] > 0.0) {
                tmp = (float) Math.sqrt(0.5 * (absx + x[0]));
                z[0] = tmp;
                z[1] = 0.5f * (x[1] / tmp);
            } else {
                tmp = (float) Math.sqrt(0.5 * (absx - x[0]));
                if (x[1] < 0.0) {
                    tmp = -tmp;
                }
                z[0] = 0.5f * (x[1] / tmp);
                z[1] = tmp;
            }
        } else {
            z[0] = 0.0f;
            z[1] = 0.0f;
        }
        return z;
    }

    public static final float[] square(float[] x) {
        return mult(x, x);
    }

    public static final float[] tan(float[] x) {
        float[] z = new float[2];
        float scalar;
        float iz_re, iz_im;
        float re1, im1, re2, im2, re3, im3;
        float cs_re, cs_im;

        iz_re = -x[1];
        iz_im = x[0];

        scalar = (float) Math.exp(iz_re);
        re1 = (float) (scalar * Math.cos(iz_im));
        im1 = (float) (scalar * Math.sin(iz_im));

        scalar = (float) Math.exp(-iz_re);
        re2 = (float) (scalar * Math.cos(-iz_im));
        im2 = (float) (scalar * Math.sin(-iz_im));

        re3 = re1 - re2;
        im3 = im1 - im2;

        z[0] = 0.5f * im3;
        z[1] = -0.5f * re3;

        re3 = re1 + re2;
        im3 = im1 + im2;

        cs_re = 0.5f * re3;
        cs_im = 0.5f * im3;

        z = div(z, cs_re, cs_im);

        return z;
    }

    /**
     * Makes this class non instantiable, but still let's others inherit from
     * it.
     */
    protected FComplex() {
    }

}
