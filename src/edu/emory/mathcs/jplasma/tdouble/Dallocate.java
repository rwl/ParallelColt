/* ***** BEGIN LICENSE BLOCK *****
 * -- Innovative Computing Laboratory
 * -- Electrical Engineering and Computer Science Department
 * -- University of Tennessee
 * -- (C) Copyright 2008
 *
 * Redistribution  and  use  in  source and binary forms, with or without
 * modification,  are  permitted  provided  that the following conditions
 * are met:
 *
 * * Redistributions  of  source  code  must  retain  the above copyright
 *   notice,  this  list  of  conditions  and  the  following  disclaimer.
 * * Redistributions  in  binary  form must reproduce the above copyright
 *   notice,  this list of conditions and the following disclaimer in the
 *   documentation  and/or other materials provided with the distribution.
 * * Neither  the  name of the University of Tennessee, Knoxville nor the
 *   names of its contributors may be used to endorse or promote products
 *   derived from this software without specific prior written permission.
 *
 * THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * ***** END LICENSE BLOCK ***** */

package edu.emory.mathcs.jplasma.tdouble;

class Dallocate {

    private Dallocate() {

    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Allocate auxiliary structures WORK[cores_max][NB_max^2] and TAU[cores_max][NB_max]
     */
    protected static int plasma_alloc_aux_work_tau() {
        int size_elems;
        int core;

        /* Allocate cache line aligned workspace of size NB_max^2 for each core */
        size_elems = Dcommon.plasma_cntrl.NB_max * Dcommon.plasma_cntrl.NB_max;
        int cores_max = Dcommon.plasma_cntrl.cores_max;
        double[][] WORK = Dcommon.plasma_aux.WORK;
        for (core = 0; core < cores_max; core++) {
            WORK[core] = new double[size_elems];
        }

        /* Allocate cache line aligned workspace of size NB_max for each core */
        size_elems = Dcommon.plasma_cntrl.NB_max;

        double[][] TAU = Dcommon.plasma_aux.TAU;
        for (core = 0; core < cores_max; core++) {
            TAU[core] = new double[size_elems];
        }
        return Dplasma.PLASMA_SUCCESS;
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Free auxiliary structures WORK[cores_max][NB_max^2] and TAU[cores_max][NB_max]
     */
    protected static int plasma_free_aux_work_tau() {
        Dcommon.plasma_aux.WORK = null;
        Dcommon.plasma_aux.TAU = null;
        return Dplasma.PLASMA_SUCCESS;
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Allocate auxiliary structure progress
     */
    protected static int plasma_alloc_aux_progress(int size_elems) {
        int[] mem_block;

        mem_block = new int[size_elems];
        Dcommon.plasma_aux.progress = mem_block;
        Dcommon.plasma_cntrl.progress_size_elems = size_elems;
        return Dplasma.PLASMA_SUCCESS;
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Free auxiliary structure progress
     */
    protected static int plasma_free_aux_progress() {
        Dcommon.plasma_aux.progress = null;
        Dcommon.plasma_cntrl.progress_size_elems = 0;
        return Dplasma.PLASMA_SUCCESS;
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Allocate storage for Block Data Layout
     */
    protected static int plasma_alloc_aux_bdl(int size_elems) {
        double[] mem_block = new double[size_elems];

        /* Allocate in standard pages */
        Dcommon.plasma_aux.bdl_mem = mem_block;
        Dcommon.plasma_cntrl.bdl_size_elems = size_elems;
        return Dplasma.PLASMA_SUCCESS;
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Free storage of Block Data Layout
     */
    protected static int plasma_free_aux_bdl() {
        Dcommon.plasma_aux.bdl_mem = null;
        Dcommon.plasma_cntrl.bdl_size_elems = 0;
        return Dplasma.PLASMA_SUCCESS;
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Allocate user's storage for T
     */
    protected static double[] plasma_Allocate_T(int M, int N) {
        int status;
        int NB, MT, NT;
        double[] T;

        /* Check if initialized */
        if (!Dcommon.plasma_cntrl.initialized) {
            Dauxiliary.plasma_warning("plasma_allocate_T", "PLASMA not initialized");
            return null;
        }

        /* Tune NB & IB depending on M, N & NRHS; Set NBNBSIZE */
        status = Dauxiliary.plasma_tune(Dglobal.PLASMA_TUNE_DGESV, M, N, 0);
        if (status != Dplasma.PLASMA_SUCCESS) {
            Dauxiliary.plasma_error("plasma_allocate_T", "plasma_tune() failed");
            return null;
        }

        /* Set MT & NT */
        NB = Dcommon.plasma_cntrl.NB;
        NT = (N % NB == 0) ? (N / NB) : (N / NB + 1);
        MT = (M % NB == 0) ? (M / NB) : (M / NB + 1);

        T = new double[MT * NT * Dcommon.plasma_cntrl.IBNBSIZE];
        return T;
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Allocate user's storage for L
     */
    protected static double[] plasma_Allocate_L(int M, int N) {
        int status;
        int NB, MT, NT;
        double[] L;

        /* Check if initialized */
        if (!Dcommon.plasma_cntrl.initialized) {
            Dauxiliary.plasma_warning("plasma_allocate_L", "PLASMA not initialized");
            return null;
        }

        /* Tune NB & IB depending on M, N & NRHS; Set NBNBSIZE */
        status = Dauxiliary.plasma_tune(Dglobal.PLASMA_TUNE_DGESV, M, N, 0);
        if (status != Dplasma.PLASMA_SUCCESS) {
            Dauxiliary.plasma_error("plasma_allocate_L", "plasma_tune() failed");
            return null;
        }

        /* Set MT & NT */
        NB = Dcommon.plasma_cntrl.NB;
        NT = (N % NB == 0) ? (N / NB) : (N / NB + 1);
        MT = (M % NB == 0) ? (M / NB) : (M / NB + 1);

        L = new double[MT * NT * Dcommon.plasma_cntrl.IBNBSIZE];
        return L;
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Allocate user's storage for IPIV
     */
    protected static int[] plasma_Allocate_IPIV(int M, int N) {
        int status;
        int NB, MT, NT;
        int[] IPIV;

        /* Check if initialized */
        if (!Dcommon.plasma_cntrl.initialized) {
            Dauxiliary.plasma_warning("plasma_allocate_IPIV", "PLASMA not initialized");
            return null;
        }

        /* Tune NB & IB depending on M, N & NRHS; Set NBNBSIZE */
        status = Dauxiliary.plasma_tune(Dglobal.PLASMA_TUNE_DGESV, M, N, 0);
        if (status != Dplasma.PLASMA_SUCCESS) {
            Dauxiliary.plasma_error("plasma_allocate_IPIV", "plasma_tune() failed");
            return null;
        }

        /* Set MT & NT */
        NB = Dcommon.plasma_cntrl.NB;
        NT = (N % NB == 0) ? (N / NB) : (N / NB + 1);
        MT = (M % NB == 0) ? (M / NB) : (M / NB + 1);

        IPIV = new int[(MT * NB) * NT];
        return IPIV;
    }
}
