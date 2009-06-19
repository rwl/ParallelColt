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

import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.locks.Lock;

import edu.emory.mathcs.jplasma.Barrier;
import edu.emory.mathcs.jplasma.tdouble.Dglobal.Dplasma_aux;
import edu.emory.mathcs.jplasma.tdouble.Dglobal.Dplasma_cntrl;
import edu.emory.mathcs.jplasma.tdouble.Dglobal.Plasma_args;
import edu.emory.mathcs.utils.ConcurrencyUtils;

class Dinitialize {

    private Dinitialize() {

    }

    /*////////////////////////////////////////////////////////////////////////////////////////
    *  PLASMA initialization
    */
    protected static int plasma_Init(int M, int N, int NRHS) {
        int size_elems;
        int NB, IB, MT, NT, NTRHS;
        int status;
        int core;
        int nthreads;

        /* Check if not initialized */
        if (Dcommon.plasma_cntrl != null) {
            if (Dcommon.plasma_cntrl.initialized) {
                Dauxiliary.plasma_warning("plasma_init", "PLASMA re-initialized");
                return Dplasma.PLASMA_ERR_REINITIALIZED;
            }
        }
        /* Check if not more cores than the hard limit */
        nthreads = ConcurrencyUtils.getNumberOfThreads();
        if (nthreads > Dglobal.CORES_MAX) {
            Dauxiliary.plasma_error("plasma_init", "not supporting so many cores");
            return Dplasma.PLASMA_ERR_INTERNAL_LIMIT;
        }

        Dcommon.plasma_cntrl = new Dplasma_cntrl(nthreads);
        Dcommon.plasma_aux = new Dplasma_aux(nthreads);
        Dcommon.plasma_args = new Plasma_args();

        /* Get system size (number of cores)
           Set number of cores to system size */
        Dcommon.plasma_cntrl.cores_max = nthreads;
        Dcommon.plasma_cntrl.cores_num = nthreads;

        /* Allocate temporary kernel workspace */
        status = Dallocate.plasma_alloc_aux_work_tau();
        if (status != Dplasma.PLASMA_SUCCESS) {
            Dauxiliary.plasma_error("plasma_init", "plasma_alloc_work_tau() failed");
            return status;
        }

        /* Allocate progress table using hinted problem size values
         * On failure recursively decrease the size by 25% */
        NB = Dcommon.plasma_cntrl.NB_min;
        MT = (M % NB == 0) ? (M / NB) : (M / NB + 1);
        NT = (N % NB == 0) ? (N / NB) : (N / NB + 1);
        NTRHS = (NRHS % NB == 0) ? (NRHS / NB) : (NRHS / NB + 1);
        size_elems = Math.max(MT, NT) * Math.max(NT, NTRHS);
        do {
            status = Dallocate.plasma_alloc_aux_progress(size_elems);
            if (status != Dplasma.PLASMA_SUCCESS) {
                size_elems = size_elems / 4 * 3;
                if (size_elems == 0) {
                    Dauxiliary.plasma_error("plasma_init", "plasma_alloc_aux_progress() failed");
                    return Dplasma.PLASMA_ERR_OUT_OF_MEMORY;
                }
            }
        } while (status != Dplasma.PLASMA_SUCCESS);

        /* Allocate bdl memory using hinted problem size values
         * On failure recursively decrease the size by 25% */
        NB = Dcommon.plasma_cntrl.NB_max;
        IB = Dcommon.plasma_cntrl.IB_max;
        MT = (M % NB == 0) ? (M / NB) : (M / NB + 1);
        NT = (N % NB == 0) ? (N / NB) : (N / NB + 1);
        NTRHS = (NRHS % NB == 0) ? (NRHS / NB) : (NRHS / NB + 1);
        size_elems = (MT * NT + MT * NTRHS) * NB * NB + (MT * NT) * IB * NB;
        do {
            status = Dallocate.plasma_alloc_aux_bdl(size_elems);
            if (status != Dplasma.PLASMA_SUCCESS) {
                size_elems = size_elems / 4 * 3;
                if (size_elems == 0) {
                    Dauxiliary.plasma_error("plasma_init", "plasma_alloc_aux_bld() failed");
                    return Dplasma.PLASMA_ERR_OUT_OF_MEMORY;
                }
            }
        } while (status != Dplasma.PLASMA_SUCCESS);

        /* Initialize barrier */
        Barrier.plasma_barrier_init(Dcommon.plasma_cntrl.cores_num - 1);

        /*  Launch threads */
        int cores_num = Dcommon.plasma_cntrl.cores_num;
        int[] core_num = Dcommon.plasma_cntrl.core_num;
        Future<?>[] workers = Dcommon.plasma_cntrl.workers;
        for (core = 1; core < cores_num; core++) {
            core_num[core] = core;
            workers[core] = ConcurrencyUtils.submit(new DcoreControl(core));
        }
        core_num[0] = 0;

        Barrier.plasma_barrier(0, cores_num);
        Dcommon.plasma_cntrl.initialized = true;
        return Dplasma.PLASMA_SUCCESS;
    }

    /*////////////////////////////////////////////////////////////////////////////////////////
     *  PLASMA completion
     */
    protected static int plasma_Finalize() {
        int core;
        int status;

        /* Check if initialized */
        if (!Dcommon.plasma_cntrl.initialized) {
            Dauxiliary.plasma_warning("plasma_finalize", "PLASMA not initialized");
            return Dplasma.PLASMA_ERR_NOT_INITIALIZED;
        }

        /* Set termination action */
        Lock lock = Dcommon.plasma_cntrl.action_mutex;
        lock.lock();
        try {
            Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_FINALIZE;
            Dcommon.plasma_cntrl.action_condt.signal();
        } finally {
            lock.unlock();
        }

        /* Barrier and clear action */
        Barrier.plasma_barrier(0, Dcommon.plasma_cntrl.cores_num);
        Dcommon.plasma_cntrl.action = Dglobal.PLASMA_ACT_STAND_BY;

        // Join threads
        int cores_num = Dcommon.plasma_cntrl.cores_num;
        Future<?>[] workers = Dcommon.plasma_cntrl.workers;
        for (core = 1; core < cores_num; core++) {
            try {
                workers[core].get();
            } catch (InterruptedException e) {
                Dauxiliary.plasma_error("plasma_finalize", "joining threads failed");
                e.printStackTrace();
            } catch (ExecutionException e) {
                Dauxiliary.plasma_error("plasma_finalize", "joining threads failed");
                e.printStackTrace();
            }
        }

        /* Release memory for storage in BDL */
        status = Dallocate.plasma_free_aux_bdl();
        if (status != Dplasma.PLASMA_SUCCESS) {
            Dauxiliary.plasma_error("plasma_finalize", "plasma_free_aux_bdl() failed");
        }

        /* Destroy progress table */
        status = Dallocate.plasma_free_aux_progress();
        if (status != Dplasma.PLASMA_SUCCESS) {
            Dauxiliary.plasma_error("plasma_finalize", "plasma_free_aux_progress() failed");
        }

        /* Destroy temporary kernel workspace */
        status = Dallocate.plasma_free_aux_work_tau();
        if (status != Dplasma.PLASMA_SUCCESS) {
            Dauxiliary.plasma_error("plasma_finalize", "plasma_free_aux_work_tau() failed");
        }

        Dcommon.plasma_cntrl.initialized = false;
        return Dplasma.PLASMA_SUCCESS;
    }
}
