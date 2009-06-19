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

import java.util.concurrent.locks.Lock;

import edu.emory.mathcs.jplasma.Barrier;

class Dgesv {

    private Dgesv() {

    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  Find the solution of a real system of linear equations using tile LU factorization
     */
    protected static int plasma_DGESV(int N, int NRHS, double[] A, int A_offset, int LDA, double[] L, int L_offset,
            int[] IPIV, int IPIV_offset, double[] B, int B_offset, int LDB) {
        int NB, NT, NTRHS;
        int status;
        double[] Abdl;
        double[] Bbdl;
        double[] Lbdl;
        double[] bdl_mem;
        int size_elems;

        /* Check if initialized */
        if (!Dcommon.plasma_cntrl.initialized) {
            Dauxiliary.plasma_warning("plasma_DGESV", "PLASMA not initialized");
            return Dplasma.PLASMA_ERR_NOT_INITIALIZED;
        }

        /* Check input arguments */
        if (N < 0) {
            Dauxiliary.plasma_error("plasma_DGESV", "illegal value of N");
            return Dplasma.PLASMA_ERR_ILLEGAL_VALUE;
        }
        if (NRHS < 0) {
            Dauxiliary.plasma_error("plasma_DGESV", "illegal value of NRHS");
            return Dplasma.PLASMA_ERR_ILLEGAL_VALUE;
        }
        if (LDA < Math.max(1, N)) {
            Dauxiliary.plasma_error("plasma_DGESV", "illegal value of LDA");
            return Dplasma.PLASMA_ERR_ILLEGAL_VALUE;
        }
        if (LDB < Math.max(1, N)) {
            Dauxiliary.plasma_error("plasma_DGESV", "illegal value of LDB");
            return Dplasma.PLASMA_ERR_ILLEGAL_VALUE;
        }
        /* Quick return */
        if (Math.min(N, NRHS) == 0)
            return Dplasma.PLASMA_SUCCESS;

        /* Tune NB & IB depending on M, N & NRHS; Set NBNBSIZE */
        status = Dauxiliary.plasma_tune(Dglobal.PLASMA_TUNE_DGESV, N, N, NRHS);
        if (status != Dplasma.PLASMA_SUCCESS) {
            Dauxiliary.plasma_error("plasma_DGESV", "plasma_tune() failed");
            return status;
        }

        /* Set NT & NTRHS */
        NB = Dcommon.plasma_cntrl.NB;
        NT = (N % NB == 0) ? (N / NB) : (N / NB + 1);
        NTRHS = (NRHS % NB == 0) ? (NRHS / NB) : (NRHS / NB + 1);

        /* If progress table too small, reallocate */
        size_elems = Math.max(NT * NT, NT * NTRHS);
        if (Dcommon.plasma_cntrl.progress_size_elems < size_elems) {
            status = Dallocate.plasma_free_aux_progress();
            if (status != Dplasma.PLASMA_SUCCESS) {
                Dauxiliary.plasma_error("plasma_DGESV", "plasma_free_aux_progress() failed");
            }
            status = Dallocate.plasma_alloc_aux_progress(size_elems);
            if (status != Dplasma.PLASMA_SUCCESS) {
                Dauxiliary.plasma_error("plasma_DGESV", "plasma_alloc_aux_progress() failed");
                return status;
            }
        }

        /* Assign arrays to BDL storage */
        bdl_mem = Dcommon.plasma_aux.bdl_mem;
        Abdl = bdl_mem;
        int Abdl_offset = 0;
        Lbdl = bdl_mem;
        int Lbdl_offset = NT * NT * Dcommon.plasma_cntrl.NBNBSIZE;
        Bbdl = bdl_mem;
        int Bbdl_offset = Lbdl_offset + NT * NT * Dcommon.plasma_cntrl.IBNBSIZE;
        /* If BDL storage too small, reallocate & reassign */
        size_elems = Bbdl_offset + NT * NTRHS * Dcommon.plasma_cntrl.NBNBSIZE;
        if (Dcommon.plasma_cntrl.bdl_size_elems < size_elems) {
            status = Dallocate.plasma_free_aux_bdl();
            if (status != Dplasma.PLASMA_SUCCESS) {
                Dauxiliary.plasma_error("plasma_DPOSV", "plasma_free_aux_bdl() failed");
                return status;
            }
            status = Dallocate.plasma_alloc_aux_bdl(size_elems);
            if (status != Dplasma.PLASMA_SUCCESS) {
                Dauxiliary.plasma_error("plasma_DPOSV", "plasma_alloc_aux_bdl() failed");
                return status;
            }
            bdl_mem = Dcommon.plasma_aux.bdl_mem;
            Abdl = bdl_mem;
            Abdl_offset = 0;
            Lbdl = bdl_mem;
            Lbdl_offset = NT * NT * Dcommon.plasma_cntrl.NBNBSIZE;
            Bbdl = bdl_mem;
            Bbdl_offset = Lbdl_offset + NT * NT * Dcommon.plasma_cntrl.IBNBSIZE;
        }

        /* Convert A from LAPACK to BDL */
        /* Set arguments */
        Dcommon.plasma_args.F77 = A;
        Dcommon.plasma_args.F77_offset = A_offset;
        Dcommon.plasma_args.A = Abdl;
        Dcommon.plasma_args.A_offset = Abdl_offset;
        Dcommon.plasma_args.M = N;
        Dcommon.plasma_args.N = N;
        Dcommon.plasma_args.LDA = LDA;
        Dcommon.plasma_args.NB = Dcommon.plasma_cntrl.NB;
        Dcommon.plasma_args.MT = NT;
        Dcommon.plasma_args.NT = NT;
        Dcommon.plasma_args.NBNBSIZE = Dcommon.plasma_cntrl.NBNBSIZE;
        /* Signal workers */
        Lock lock = Dcommon.plasma_cntrl.action_mutex;
        lock.lock();
        try {
            Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_F77_TO_BDL;
            Dcommon.plasma_cntrl.action_condt.signalAll();
        } finally {
            lock.unlock();
        }
        /* Call for master */
        Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);
        Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_STAND_BY;
        DbdlConvert.plasma_lapack_to_bdl(Dcommon.plasma_args.F77, Dcommon.plasma_args.F77_offset,
                Dcommon.plasma_args.A, Dcommon.plasma_args.A_offset, Dcommon.plasma_args.M, Dcommon.plasma_args.N,
                Dcommon.plasma_args.LDA, Dcommon.plasma_args.NB, Dcommon.plasma_args.MT, Dcommon.plasma_args.NT,
                Dcommon.plasma_args.NBNBSIZE, Dcommon.plasma_cntrl.cores_num, 0);
        Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);

        /* Convert B from LAPACK to BDL */
        /* Set arguments */
        Dcommon.plasma_args.F77 = B;
        Dcommon.plasma_args.F77_offset = B_offset;
        Dcommon.plasma_args.A = Bbdl;
        Dcommon.plasma_args.A_offset = Bbdl_offset;
        Dcommon.plasma_args.M = N;
        Dcommon.plasma_args.N = NRHS;
        Dcommon.plasma_args.LDA = LDB;
        Dcommon.plasma_args.NB = Dcommon.plasma_cntrl.NB;
        Dcommon.plasma_args.MT = NT;
        Dcommon.plasma_args.NT = NTRHS;
        Dcommon.plasma_args.NBNBSIZE = Dcommon.plasma_cntrl.NBNBSIZE;
        /* Signal workers */
        lock = Dcommon.plasma_cntrl.action_mutex;
        lock.lock();
        try {
            Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_F77_TO_BDL;
            Dcommon.plasma_cntrl.action_condt.signalAll();
        } finally {
            lock.unlock();
        }
        /* Call for master */
        Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);
        Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_STAND_BY;
        DbdlConvert.plasma_lapack_to_bdl(Dcommon.plasma_args.F77, Dcommon.plasma_args.F77_offset,
                Dcommon.plasma_args.A, Dcommon.plasma_args.A_offset, Dcommon.plasma_args.M, Dcommon.plasma_args.N,
                Dcommon.plasma_args.LDA, Dcommon.plasma_args.NB, Dcommon.plasma_args.MT, Dcommon.plasma_args.NT,
                Dcommon.plasma_args.NBNBSIZE, Dcommon.plasma_cntrl.cores_num, 0);
        Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);

        /* Clear IPIV and Lbdl */
        size_elems = IPIV_offset + NT * NT * Dcommon.plasma_cntrl.NB;
        for (int i = IPIV_offset; i < size_elems; i++) {
            IPIV[i] = 0;
        }
        size_elems = Lbdl_offset + NT * NT * Dcommon.plasma_cntrl.IBNBSIZE;
        for (int i = Lbdl_offset; i < size_elems; i++) {
            Lbdl[i] = 0;
        }
        /* Use LU factorization */
        /* Call parallel DGETRF */
        /* Set arguments */
        Dcommon.plasma_args.M = N;
        Dcommon.plasma_args.N = N;
        Dcommon.plasma_args.A = Abdl;
        Dcommon.plasma_args.A_offset = Abdl_offset;
        Dcommon.plasma_args.NB = Dcommon.plasma_cntrl.NB;
        Dcommon.plasma_args.NBNBSIZE = Dcommon.plasma_cntrl.NBNBSIZE;
        Dcommon.plasma_args.IBNBSIZE = Dcommon.plasma_cntrl.IBNBSIZE;
        Dcommon.plasma_args.IB = Dcommon.plasma_cntrl.IB;
        Dcommon.plasma_args.MT = NT;
        Dcommon.plasma_args.NT = NT;
        Dcommon.plasma_args.L = Lbdl;
        Dcommon.plasma_args.L_offset = Lbdl_offset;
        Dcommon.plasma_args.IPIV = IPIV;
        Dcommon.plasma_args.IPIV_offset = IPIV_offset;

        /* Clear progress table */
        Dauxiliary.plasma_clear_aux_progress(NT * NT, -1);
        /* Signal workers */
        lock = Dcommon.plasma_cntrl.action_mutex;
        lock.lock();
        try {
            Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_DGETRF;
            Dcommon.plasma_cntrl.action_condt.signalAll();
        } finally {
            lock.unlock();
        }
        /* Call for master */
        Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);
        Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_STAND_BY;
        Pdgetrf.plasma_pDGETRF(Dcommon.plasma_args.M, Dcommon.plasma_args.N, Dcommon.plasma_args.A,
                Dcommon.plasma_args.A_offset, Dcommon.plasma_args.NB, Dcommon.plasma_args.NBNBSIZE,
                Dcommon.plasma_args.IBNBSIZE, Dcommon.plasma_args.IB, Dcommon.plasma_args.MT, Dcommon.plasma_args.NT,
                Dcommon.plasma_args.L, Dcommon.plasma_args.L_offset, Dcommon.plasma_args.IPIV,
                Dcommon.plasma_args.IPIV_offset, Dcommon.plasma_args.INFO, Dcommon.plasma_cntrl.cores_num, 0);
        Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);

        if (Dcommon.plasma_args.INFO.val == Dplasma.PLASMA_SUCCESS) {
            /* Call parallel DTRSMPL */
            /* Set arguments */
            Dcommon.plasma_args.M = N;
            Dcommon.plasma_args.N = N;
            Dcommon.plasma_args.NRHS = NRHS;
            Dcommon.plasma_args.A = Abdl;
            Dcommon.plasma_args.A_offset = Abdl_offset;
            Dcommon.plasma_args.NB = Dcommon.plasma_cntrl.NB;
            Dcommon.plasma_args.NBNBSIZE = Dcommon.plasma_cntrl.NBNBSIZE;
            Dcommon.plasma_args.IBNBSIZE = Dcommon.plasma_cntrl.IBNBSIZE;
            Dcommon.plasma_args.IB = Dcommon.plasma_cntrl.IB;
            Dcommon.plasma_args.MT = NT;
            Dcommon.plasma_args.NTRHS = NTRHS;
            Dcommon.plasma_args.NT = NT;
            Dcommon.plasma_args.L = Lbdl;
            Dcommon.plasma_args.L_offset = Lbdl_offset;
            Dcommon.plasma_args.IPIV = IPIV;
            Dcommon.plasma_args.IPIV_offset = IPIV_offset;
            Dcommon.plasma_args.B = Bbdl;
            Dcommon.plasma_args.B_offset = Bbdl_offset;
            /* Clear progress table */
            Dauxiliary.plasma_clear_aux_progress(NT * NTRHS, -1);
            /* Signal workers */
            lock = Dcommon.plasma_cntrl.action_mutex;
            lock.lock();
            try {
                Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_DTRSMPL;
                Dcommon.plasma_cntrl.action_condt.signalAll();
            } finally {
                lock.unlock();
            }
            /* Call for master */
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);
            Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_STAND_BY;
            Pdtrsmpl.plasma_pDTRSMPL(Dcommon.plasma_args.M, Dcommon.plasma_args.NRHS, Dcommon.plasma_args.N,
                    Dcommon.plasma_args.A, Dcommon.plasma_args.A_offset, Dcommon.plasma_args.NB,
                    Dcommon.plasma_args.NBNBSIZE, Dcommon.plasma_args.IBNBSIZE, Dcommon.plasma_args.IB,
                    Dcommon.plasma_args.MT, Dcommon.plasma_args.NTRHS, Dcommon.plasma_args.NT, Dcommon.plasma_args.L,
                    Dcommon.plasma_args.L_offset, Dcommon.plasma_args.IPIV, Dcommon.plasma_args.IPIV_offset,
                    Dcommon.plasma_args.B, Dcommon.plasma_args.B_offset, Dcommon.plasma_args.INFO,
                    Dcommon.plasma_cntrl.cores_num, 0);
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);

            /* Call parallel DTRSM */
            /* Set arguments */
            Dcommon.plasma_args.uplo = Dplasma.PlasmaUpper;
            Dcommon.plasma_args.trans = Dplasma.PlasmaNoTrans;
            Dcommon.plasma_args.diag = Dplasma.PlasmaNonUnit;
            Dcommon.plasma_args.N = N;
            Dcommon.plasma_args.NRHS = NRHS;
            Dcommon.plasma_args.A = Abdl;
            Dcommon.plasma_args.A_offset = Abdl_offset;
            Dcommon.plasma_args.NB = Dcommon.plasma_cntrl.NB;
            Dcommon.plasma_args.NBNBSIZE = Dcommon.plasma_cntrl.NBNBSIZE;
            Dcommon.plasma_args.NT = NT;
            Dcommon.plasma_args.MT = NT;
            Dcommon.plasma_args.B = Bbdl;
            Dcommon.plasma_args.B_offset = Bbdl_offset;
            Dcommon.plasma_args.MTB = NT;
            Dcommon.plasma_args.NTRHS = NTRHS;
            /* Clear progress table */
            Dauxiliary.plasma_clear_aux_progress(NT * NTRHS, -1);
            /* Signal workers */
            lock = Dcommon.plasma_cntrl.action_mutex;
            lock.lock();
            try {
                Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_DTRSM;
                Dcommon.plasma_cntrl.action_condt.signalAll();
            } finally {
                lock.unlock();
            }
            /* Call for master */
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);
            Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_STAND_BY;
            Pdtrsm.plasma_pDTRSM(Dplasma.PlasmaLeft, Dcommon.plasma_args.uplo, Dcommon.plasma_args.trans,
                    Dcommon.plasma_args.diag, Dcommon.plasma_args.N, Dcommon.plasma_args.NRHS, 1.0,
                    Dcommon.plasma_args.A, Dcommon.plasma_args.A_offset, Dcommon.plasma_args.NB,
                    Dcommon.plasma_args.NBNBSIZE, Dcommon.plasma_args.NT, Dcommon.plasma_args.MT,
                    Dcommon.plasma_args.B, Dcommon.plasma_args.B_offset, Dcommon.plasma_args.MTB,
                    Dcommon.plasma_args.NTRHS, Dcommon.plasma_cntrl.cores_num, 0);
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);

            /* Convert A from BDL to LAPACK */
            /* Set arguments */
            Dcommon.plasma_args.A = Abdl;
            Dcommon.plasma_args.A_offset = Abdl_offset;
            Dcommon.plasma_args.F77 = A;
            Dcommon.plasma_args.F77_offset = A_offset;
            Dcommon.plasma_args.M = N;
            Dcommon.plasma_args.N = N;
            Dcommon.plasma_args.LDA = LDA;
            Dcommon.plasma_args.NB = Dcommon.plasma_cntrl.NB;
            Dcommon.plasma_args.MT = NT;
            Dcommon.plasma_args.NT = NT;
            Dcommon.plasma_args.NBNBSIZE = Dcommon.plasma_cntrl.NBNBSIZE;
            /* Signal workers */
            lock = Dcommon.plasma_cntrl.action_mutex;
            lock.lock();
            try {
                Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_BDL_TO_F77;
                Dcommon.plasma_cntrl.action_condt.signalAll();
            } finally {
                lock.unlock();
            }
            /* Call for master */
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);
            Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_STAND_BY;
            DbdlConvert.plasma_bdl_to_lapack(Dcommon.plasma_args.A, Dcommon.plasma_args.A_offset,
                    Dcommon.plasma_args.F77, Dcommon.plasma_args.F77_offset, Dcommon.plasma_args.M,
                    Dcommon.plasma_args.N, Dcommon.plasma_args.LDA, Dcommon.plasma_args.NB, Dcommon.plasma_args.MT,
                    Dcommon.plasma_args.NT, Dcommon.plasma_args.NBNBSIZE, Dcommon.plasma_cntrl.cores_num, 0);
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);

            /* Convert B from BDL to LAPACK */
            /* Set arguments */
            Dcommon.plasma_args.A = Bbdl;
            Dcommon.plasma_args.A_offset = Bbdl_offset;
            Dcommon.plasma_args.F77 = B;
            Dcommon.plasma_args.F77_offset = B_offset;
            Dcommon.plasma_args.M = N;
            Dcommon.plasma_args.N = NRHS;
            Dcommon.plasma_args.LDA = LDB;
            Dcommon.plasma_args.NB = Dcommon.plasma_cntrl.NB;
            Dcommon.plasma_args.MT = NT;
            Dcommon.plasma_args.NT = NTRHS;
            Dcommon.plasma_args.NBNBSIZE = Dcommon.plasma_cntrl.NBNBSIZE;
            /* Signal workers */
            lock = Dcommon.plasma_cntrl.action_mutex;
            lock.lock();
            try {
                Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_BDL_TO_F77;
                Dcommon.plasma_cntrl.action_condt.signalAll();
            } finally {
                lock.unlock();
            }
            /* Call for master */
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);
            Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_STAND_BY;
            DbdlConvert.plasma_bdl_to_lapack(Dcommon.plasma_args.A, Dcommon.plasma_args.A_offset,
                    Dcommon.plasma_args.F77, Dcommon.plasma_args.F77_offset, Dcommon.plasma_args.M,
                    Dcommon.plasma_args.N, Dcommon.plasma_args.LDA, Dcommon.plasma_args.NB, Dcommon.plasma_args.MT,
                    Dcommon.plasma_args.NT, Dcommon.plasma_args.NBNBSIZE, Dcommon.plasma_cntrl.cores_num, 0);
            Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);
        }

        return Dcommon.plasma_args.INFO.val;
    }

}
