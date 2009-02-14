package cern.colt.matrix;

public enum Norm {

    /**
     * Maximum absolute row sum
     */
    One,
    
    /**
     * Euclidean norm
     */
    Two,

    /**
     * The root of sum of the sum of squares
     */
    Frobenius,

    /**
     * Maximum column sum
     */
    Infinity,

}