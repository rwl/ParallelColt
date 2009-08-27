package cern.jet.math.tfcomplex;

import cern.colt.function.tfcomplex.FComplexFComplexFComplexFunction;
import cern.colt.function.tfcomplex.FComplexFComplexFunction;
import cern.colt.function.tfcomplex.FComplexFComplexRealProcedure;
import cern.colt.function.tfcomplex.FComplexFComplexRealRealFunction;
import cern.colt.function.tfcomplex.FComplexProcedure;
import cern.colt.function.tfcomplex.FComplexRealFComplexFunction;
import cern.colt.function.tfcomplex.FComplexRealFunction;
import cern.colt.function.tfcomplex.RealFComplexFComplexFunction;
import cern.colt.function.tfcomplex.RealFComplexFunction;

/**
 * Complex function objects to be passed to generic methods.
 * 
 * @author Piotr Wendykier (piotr.wendykier@gmail.com)
 */
public class FComplexFunctions {

    public static final FComplexFunctions functions = new FComplexFunctions();

    /***************************************************************************
     * <H3>Unary functions</H3>
     **************************************************************************/

    public static final FComplexRealFunction abs = new FComplexRealFunction() {
        public final float apply(float[] x) {
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
    };

    public static final FComplexFComplexFunction acos = new FComplexFComplexFunction() {
        public final float[] apply(float[] x) {
            float[] z = new float[2];

            float re, im;

            re = (float) (1.0 - ((x[0] * x[0]) - (x[1] * x[1])));
            im = -((x[0] * x[1]) + (x[1] * x[0]));

            z[0] = re;
            z[1] = im;
            z = FComplex.sqrt(z);

            re = -z[1];
            im = z[0];

            z[0] = x[0] + re;
            z[1] = x[1] + im;

            re = (float) Math.log(FComplex.abs(z));
            im = (float) Math.atan2(z[1], z[0]);

            z[0] = im;
            z[1] = -re;
            return z;
        }

        public final float[] apply(float re, float im) {
            float[] z = new float[2];

            float re2, im2;

            re2 = (float) (1.0 - ((re * re) - (im * im)));
            im2 = -((re * im) + (im * re));

            z[0] = re2;
            z[1] = im2;
            z = FComplex.sqrt(z);

            re2 = -z[1];
            im2 = z[0];

            z[0] = re + re2;
            z[1] = im + im2;

            re2 = (float) Math.log(FComplex.abs(z));
            im2 = (float) Math.atan2(z[1], z[0]);

            z[0] = im2;
            z[1] = -re2;
            return z;
        }
    };

    public static final FComplexRealFunction arg = new FComplexRealFunction() {
        public final float apply(float[] x) {
            return (float) Math.atan2(x[1], x[0]);
        }
    };

    public static final FComplexFComplexFunction asin = new FComplexFComplexFunction() {
        public final float[] apply(float[] x) {
            float[] z = new float[2];

            float re, im;

            re = (float) (1.0 - ((x[0] * x[0]) - (x[1] * x[1])));
            im = -((x[0] * x[1]) + (x[1] * x[0]));

            z[0] = re;
            z[1] = im;
            z = FComplex.sqrt(z);

            re = -z[1];
            im = z[0];

            z[0] = z[0] + re;
            z[1] = z[1] + im;

            re = (float) Math.log(FComplex.abs(z));
            im = (float) Math.atan2(z[1], z[0]);

            z[0] = im;
            z[1] = -re;
            return z;
        }

        public final float[] apply(float re, float im) {
            float[] z = new float[2];

            float re2, im2;

            re2 = (float) (1.0 - ((re * re) - (im * im)));
            im2 = -((re * im) + (im * re));

            z[0] = re2;
            z[1] = im2;
            z = FComplex.sqrt(z);

            re2 = -z[1];
            im2 = z[0];

            z[0] = z[0] + re2;
            z[1] = z[1] + im2;

            re2 = (float) Math.log(FComplex.abs(z));
            im2 = (float) Math.atan2(z[1], z[0]);

            z[0] = im2;
            z[1] = -re2;
            return z;
        }
    };

    public static final FComplexFComplexFunction atan = new FComplexFComplexFunction() {
        public final float[] apply(float[] x) {
            float[] z = new float[2];

            float re, im;

            z[0] = -x[0];
            z[1] = 1.0f - x[1];

            re = x[0];
            im = 1.0f + x[1];

            z = FComplex.div(z, re, im);

            re = (float) Math.log(FComplex.abs(z));
            im = (float) Math.atan2(z[1], z[0]);

            z[0] = 0.5f * im;
            z[1] = -0.5f * re;

            return z;
        }

        public final float[] apply(float re, float im) {
            float[] z = new float[2];

            float re2, im2;

            z[0] = -re;
            z[1] = 1.0f - im;

            re2 = re;
            im2 = 1.0f + im;

            z = FComplex.div(z, re2, im2);

            re2 = (float) Math.log(FComplex.abs(z));
            im2 = (float) Math.atan2(z[1], z[0]);

            z[0] = 0.5f * im2;
            z[1] = -0.5f * re2;

            return z;
        }
    };

    public static final FComplexFComplexFunction conj = new FComplexFComplexFunction() {
        public final float[] apply(float[] x) {
            float[] z = new float[2];
            z[0] = x[0];
            z[1] = -x[1];
            return z;
        }

        public final float[] apply(float re, float im) {
            float[] z = new float[2];
            z[0] = re;
            z[1] = -im;
            return z;
        }
    };

    public static final FComplexFComplexFunction cos = new FComplexFComplexFunction() {
        public final float[] apply(float[] x) {
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

        public final float[] apply(float re, float im) {
            float[] z = new float[2];

            float re1, im1, re2, im2;
            float scalar;
            float iz_re, iz_im;

            iz_re = -im;
            iz_im = re;

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
    };

    public static final FComplexFComplexFunction exp = new FComplexFComplexFunction() {
        public final float[] apply(float[] x) {
            float[] z = new float[2];
            float scalar = (float) Math.exp(x[0]);
            z[0] = (float) (scalar * Math.cos(x[1]));
            z[1] = (float) (scalar * Math.sin(x[1]));
            return z;
        }

        public final float[] apply(float re, float im) {
            float[] z = new float[2];
            float scalar = (float) Math.exp(re);
            z[0] = (float) (scalar * Math.cos(im));
            z[1] = (float) (scalar * Math.sin(im));
            return z;
        }
    };

    public static final FComplexFComplexFunction identity = new FComplexFComplexFunction() {
        public final float[] apply(float[] x) {
            return x;
        }

        public final float[] apply(float re, float im) {
            return new float[] { re, im };
        }
    };

    public static final FComplexFComplexFunction inv = new FComplexFComplexFunction() {
        public final float[] apply(float[] x) {
            float[] z = new float[2];
            if (x[1] != 0.0) {
                float tmp = (x[0] * x[0]) + (x[1] * x[1]);
                z[0] = x[0] / tmp;
                z[1] = -x[1] / tmp;
            } else {
                z[0] = 1 / x[0];
                z[1] = 0;
            }
            return z;
        }

        public final float[] apply(float re, float im) {
            float[] z = new float[2];
            if (im != 0.0) {
                float scalar;
                if (Math.abs(re) >= Math.abs(z[1])) {
                    scalar = (float) (1.0 / (re + im * (im / re)));
                    z[0] = scalar;
                    z[1] = scalar * (-im / re);
                } else {
                    scalar = (float) (1.0 / (re * (re / im) + im));
                    z[0] = scalar * (re / im);
                    z[1] = -scalar;
                }
            } else {
                z[0] = 1 / re;
                z[1] = 0;
            }
            return z;
        }
    };

    public static final FComplexFComplexFunction log = new FComplexFComplexFunction() {
        public final float[] apply(float[] x) {
            float[] z = new float[2];
            z[0] = (float) Math.log(FComplex.abs(x));
            z[1] = FComplex.arg(x);
            return z;
        }

        public final float[] apply(float re, float im) {
            float[] z = new float[2];
            z[0] = (float) Math.log(FComplex.abs(re, im));
            z[1] = FComplex.arg(re, im);
            return z;
        }
    };

    public static final FComplexFComplexFunction neg = new FComplexFComplexFunction() {
        public final float[] apply(float[] x) {
            return new float[] { -x[0], -x[1] };
        }

        public final float[] apply(float re, float im) {
            return new float[] { -re, -im };
        }
    };

    public static final FComplexFComplexFunction sin = new FComplexFComplexFunction() {
        public final float[] apply(float[] x) {
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

        public final float[] apply(float re, float im) {
            float[] z = new float[2];
            float re1, im1, re2, im2;
            float scalar;
            float iz_re, iz_im;

            iz_re = -im;
            iz_im = re;

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
    };

    public static final FComplexFComplexFunction sqrt = new FComplexFComplexFunction() {
        public final float[] apply(float[] x) {
            float[] z = new float[2];
            float absx = FComplex.abs(x);
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

        public final float[] apply(float re, float im) {
            float[] z = new float[2];
            float absx = FComplex.abs(re, im);
            float tmp;
            if (absx > 0.0) {
                if (re > 0.0) {
                    tmp = (float) Math.sqrt(0.5 * (absx + re));
                    z[0] = tmp;
                    z[1] = 0.5f * (im / tmp);
                } else {
                    tmp = (float) Math.sqrt(0.5 * (absx - re));
                    if (im < 0.0) {
                        tmp = -tmp;
                    }
                    z[0] = 0.5f * (im / tmp);
                    z[1] = tmp;
                }
            } else {
                z[0] = 0.0f;
                z[1] = 0.0f;
            }
            return z;
        }
    };

    public static final FComplexFComplexFunction square = new FComplexFComplexFunction() {
        public final float[] apply(float[] x) {
            float[] z = new float[2];
            z[0] = x[0] * x[0] - x[1] * x[1];
            z[1] = x[1] * x[0] + x[0] * x[1];
            return z;
        }

        public final float[] apply(float re, float im) {
            float[] z = new float[2];
            z[0] = re * re - im * im;
            z[1] = im * re + re * im;
            return z;
        }
    };

    public static final FComplexFComplexFunction tan = new FComplexFComplexFunction() {
        public final float[] apply(float[] x) {
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

            z = FComplex.div(z, cs_re, cs_im);

            return z;
        }

        public final float[] apply(float re, float im) {
            float[] z = new float[2];
            float scalar;
            float iz_re, iz_im;
            float re1, im1, re2, im2, re3, im3;
            float cs_re, cs_im;

            iz_re = -im;
            iz_im = re;

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

            z = FComplex.div(z, cs_re, cs_im);

            return z;
        }
    };

    /***************************************************************************
     * <H3>Binary functions</H3>
     **************************************************************************/

    public static final FComplexFComplexFComplexFunction div = new FComplexFComplexFComplexFunction() {
        public final float[] apply(float[] x, float[] y) {
            float re = y[0];
            float im = y[1];

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
    };

    public static final FComplexFComplexRealRealFunction equals = new FComplexFComplexRealRealFunction() {
        public final float apply(float[] x, float[] y, float tol) {
            if (FComplex.abs(x[0] - y[0], x[1] - y[1]) <= Math.abs(tol)) {
                return 1;
            } else {
                return 0;
            }
        }
    };

    public static final FComplexFComplexRealProcedure isEqual = new FComplexFComplexRealProcedure() {
        public final boolean apply(float[] x, float[] y, float tol) {
            if (FComplex.abs(x[0] - y[0], x[1] - y[1]) <= Math.abs(tol)) {
                return true;
            } else {
                return false;
            }
        }
    };

    public static final FComplexFComplexFComplexFunction minus = new FComplexFComplexFComplexFunction() {
        public final float[] apply(float[] x, float[] y) {
            float[] z = new float[2];
            z[0] = x[0] - y[0];
            z[1] = x[1] - y[1];
            return z;
        }
    };

    public static final FComplexFComplexFComplexFunction mult = new FComplexFComplexFComplexFunction() {
        public final float[] apply(float[] x, float[] y) {
            float[] z = new float[2];
            z[0] = x[0] * y[0] - x[1] * y[1];
            z[1] = x[1] * y[0] + x[0] * y[1];
            return z;
        }
    };

    public static final FComplexFComplexFComplexFunction multConjFirst = new FComplexFComplexFComplexFunction() {
        public final float[] apply(float[] x, float[] y) {
            float[] z = new float[2];
            z[0] = x[0] * y[0] + x[1] * y[1];
            z[1] = -x[1] * y[0] + x[0] * y[1];
            return z;
        }
    };

    public static final FComplexFComplexFComplexFunction multConjSecond = new FComplexFComplexFComplexFunction() {
        public final float[] apply(float[] x, float[] y) {
            float[] z = new float[2];
            z[0] = x[0] * y[0] + x[1] * y[1];
            z[1] = x[1] * y[0] - x[0] * y[1];
            return z;
        }
    };

    public static final FComplexFComplexFComplexFunction plus = new FComplexFComplexFComplexFunction() {
        public final float[] apply(float[] x, float[] y) {
            float[] z = new float[2];
            z[0] = x[0] + y[0];
            z[1] = x[1] + y[1];
            return z;
        }
    };

    public static final FComplexRealFComplexFunction pow1 = new FComplexRealFComplexFunction() {
        public final float[] apply(float[] x, float y) {
            float[] z = new float[2];
            float re = (float) (y * Math.log(FComplex.abs(x)));
            float im = y * FComplex.arg(x);
            float scalar = (float) Math.exp(re);
            z[0] = (float) (scalar * Math.cos(im));
            z[1] = (float) (scalar * Math.sin(im));
            return z;
        }
    };

    public static final RealFComplexFComplexFunction pow2 = new RealFComplexFComplexFunction() {
        public final float[] apply(float x, float[] y) {
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
    };

    public static final FComplexFComplexFComplexFunction pow3 = new FComplexFComplexFComplexFunction() {
        public final float[] apply(float[] x, float[] y) {
            float[] z = new float[2];
            float re = (float) Math.log(FComplex.abs(x));
            float im = FComplex.arg(x);

            float re2 = (re * y[0]) - (im * y[1]);
            float im2 = (re * y[1]) + (im * y[0]);

            float scalar = (float) Math.exp(re2);

            z[0] = (float) (scalar * Math.cos(im2));
            z[1] = (float) (scalar * Math.sin(im2));
            return z;
        }
    };

    public static FComplexFComplexFunction bindArg1(final FComplexFComplexFComplexFunction function, final float[] c) {
        return new FComplexFComplexFunction() {
            public final float[] apply(float[] var) {
                return function.apply(c, var);
            }

            public final float[] apply(float re, float im) {
                return function.apply(c, new float[] { re, im });
            }
        };
    }

    public static FComplexFComplexFunction bindArg2(final FComplexFComplexFComplexFunction function, final float[] c) {
        return new FComplexFComplexFunction() {
            public final float[] apply(float[] var) {
                return function.apply(var, c);
            }

            public final float[] apply(float re, float im) {
                return function.apply(new float[] { re, im }, c);
            }
        };
    }

    public static FComplexFComplexFComplexFunction chain(final FComplexFComplexFComplexFunction f,
            final FComplexFComplexFunction g, final FComplexFComplexFunction h) {
        return new FComplexFComplexFComplexFunction() {
            public final float[] apply(float[] x, float[] y) {
                return f.apply(g.apply(x), h.apply(y));
            }
        };
    }

    public static FComplexFComplexFComplexFunction chain(final FComplexFComplexFunction g,
            final FComplexFComplexFComplexFunction h) {
        return new FComplexFComplexFComplexFunction() {
            public final float[] apply(float[] x, float[] y) {
                return g.apply(h.apply(x, y));
            }
        };
    }

    public static FComplexFComplexFunction chain(final FComplexFComplexFunction g, final FComplexFComplexFunction h) {
        return new FComplexFComplexFunction() {
            public final float[] apply(float[] x) {
                return g.apply(h.apply(x));
            }

            public final float[] apply(float re, float im) {
                return g.apply(h.apply(new float[] { re, im }));
            }
        };
    }

    public static FComplexFComplexFunction constant(final float[] c) {
        return new FComplexFComplexFunction() {
            public final float[] apply(float[] x) {
                return c;
            }

            public final float[] apply(float re, float im) {
                return new float[] { re, im };
            }
        };
    }

    public static FComplexFComplexFunction div(final float[] b) {
        return mult(FComplex.inv(b));
    }

    public static FComplexFComplexFunction div(final float b) {
        float[] tmp = new float[] { b, 0 };
        return mult(FComplex.inv(tmp));
    }

    public static FComplexRealFunction equals(final float[] y) {
        return new FComplexRealFunction() {
            public final float apply(float[] x) {
                if (x[0] == y[0] && x[1] == y[1]) {
                    return 1;
                } else {
                    return 0;
                }
            }
        };
    }

    public static FComplexProcedure isEqual(final float[] y) {
        return new FComplexProcedure() {
            public final boolean apply(float[] x) {
                if (x[0] == y[0] && x[1] == y[1]) {
                    return true;
                } else {
                    return false;
                }
            }
        };
    }

    public static FComplexFComplexFunction minus(final float[] x) {
        float[] negb = new float[2];
        negb[0] = -x[0];
        negb[1] = -x[1];
        return plus(negb);
    }

    public static FComplexFComplexFComplexFunction minusMult(final float[] constant) {
        float[] negconstant = new float[2];
        negconstant[0] = -constant[0];
        negconstant[1] = -constant[1];
        return plusMultSecond(negconstant);
    }

    public static FComplexFComplexFunction mult(final float[] x) {
        return new FComplexMult(x);
    }

    public static FComplexFComplexFunction mult(final float x) {
        return new FComplexMult(new float[] { x, 0 });
    }

    public static FComplexFComplexFunction plus(final float[] y) {
        return new FComplexFComplexFunction() {
            public final float[] apply(float[] x) {
                float[] z = new float[2];
                z[0] = x[0] + y[0];
                z[1] = x[1] + y[1];
                return z;
            }

            public final float[] apply(float re, float im) {
                float[] z = new float[2];
                z[0] = re + y[0];
                z[1] = im + y[1];
                return z;
            }
        };
    }

    public static FComplexFComplexFComplexFunction plusMultSecond(float[] constant) {
        return new FComplexPlusMultSecond(constant);
    }

    public static FComplexFComplexFComplexFunction plusMultFirst(float[] constant) {
        return new FComplexPlusMultFirst(constant);
    }

    public static FComplexFComplexFunction pow1(final float y) {
        return new FComplexFComplexFunction() {
            public final float[] apply(float[] x) {
                float[] z = new float[2];
                float re = (float) (y * Math.log(FComplex.abs(x)));
                float im = y * FComplex.arg(x);
                float scalar = (float) Math.exp(re);
                z[0] = (float) (scalar * Math.cos(im));
                z[1] = (float) (scalar * Math.sin(im));
                return z;
            }

            public final float[] apply(float re, float im) {
                float[] z = new float[2];
                float re2 = (float) (y * Math.log(FComplex.abs(re, im)));
                float im2 = y * FComplex.arg(re, im);
                float scalar = (float) Math.exp(re2);
                z[0] = (float) (scalar * Math.cos(im2));
                z[1] = (float) (scalar * Math.sin(im2));
                return z;
            }
        };
    }

    public static RealFComplexFunction pow2(final float[] y) {
        return new RealFComplexFunction() {
            public final float[] apply(float x) {
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
        };
    }

    public static FComplexFComplexFunction pow3(final float[] y) {
        return new FComplexFComplexFunction() {
            public final float[] apply(float[] x) {
                float[] z = new float[2];
                float re = (float) Math.log(FComplex.abs(x));
                float im = FComplex.arg(x);

                float re2 = (re * y[0]) - (im * y[1]);
                float im2 = (re * y[1]) + (im * y[0]);

                float scalar = (float) Math.exp(re2);

                z[0] = (float) (scalar * Math.cos(im2));
                z[1] = (float) (scalar * Math.sin(im2));
                return z;
            }

            public final float[] apply(float re, float im) {
                float[] z = new float[2];
                float re1 = (float) Math.log(FComplex.abs(re, im));
                float im1 = FComplex.arg(re, im);

                float re2 = (re1 * y[0]) - (im1 * y[1]);
                float im2 = (re1 * y[1]) + (im1 * y[0]);

                float scalar = (float) Math.exp(re2);

                z[0] = (float) (scalar * Math.cos(im2));
                z[1] = (float) (scalar * Math.sin(im2));
                return z;
            }
        };
    }

    public static FComplexFComplexFunction random() {
        return new RandomComplexFunction();
    }

    private static class RandomComplexFunction implements FComplexFComplexFunction {

        public float[] apply(float[] argument) {
            return new float[] { (float) Math.random(), (float) Math.random() };
        }

        public float[] apply(float re, float im) {
            return new float[] { (float) Math.random(), (float) Math.random() };
        }

    }

    public static FComplexFComplexFComplexFunction swapArgs(final FComplexFComplexFComplexFunction function) {
        return new FComplexFComplexFComplexFunction() {
            public final float[] apply(float[] x, float[] y) {
                return function.apply(y, x);
            }
        };
    }
}
