package cern.colt.matrix.tfloat.algo.solver.preconditioner;

import cern.colt.matrix.tfloat.FloatMatrix1D;
import cern.colt.matrix.tfloat.FloatMatrix2D;

public class FloatIdentity implements FloatPreconditioner {

    public FloatMatrix1D apply(FloatMatrix1D b, FloatMatrix1D x) {
        if (x == null) {
            x = b.like();
        }

        return x.assign(b);
    }

    public FloatMatrix1D transApply(FloatMatrix1D b, FloatMatrix1D x) {
        if (x == null) {
            x = b.like();
        }

        return x.assign(b);
    }

    public void setMatrix(FloatMatrix2D A) {
        // nothing to do
    }

}
