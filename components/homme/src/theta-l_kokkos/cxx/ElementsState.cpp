/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "ElementsState.hpp"
#include "utilities/SubviewUtils.hpp"
#include "utilities/SyncUtils.hpp"
#include "utilities/TestUtils.hpp"
#include "HybridVCoord.hpp"

#include <limits>
#include <random>
#include <assert.h>

namespace Homme {

void StateStorage::init_storage(const int num_elems) {

  m_v         = ExecViewManaged<Scalar * [NUM_TIME_LEVELS][2][NP][NP][NUM_LEV  ]>("Horizontal velocity", num_elems);
  m_w_i       = ExecViewManaged<Scalar * [NUM_TIME_LEVELS]   [NP][NP][NUM_LEV_P]>("Vertical velocity at interfaces", num_elems);
  m_vtheta_dp = ExecViewManaged<Scalar * [NUM_TIME_LEVELS]   [NP][NP][NUM_LEV  ]>("Virtual potential temperature", num_elems);
  m_phinh_i   = ExecViewManaged<Scalar * [NUM_TIME_LEVELS]   [NP][NP][NUM_LEV_P]>("Geopotential at interfaces", num_elems);
  m_dp3d      = ExecViewManaged<Scalar * [NUM_TIME_LEVELS]   [NP][NP][NUM_LEV  ]>("Delta p at levels", num_elems);

  m_ps_v = ExecViewManaged<Real * [NUM_TIME_LEVELS][NP][NP]>("PS_V", num_elems);
}

void StateStorage::copy_state(const StateStorage& src) {

  Kokkos::deep_copy(m_v        , src.m_v         );
  Kokkos::deep_copy(m_w_i      , src.m_w_i       );
  Kokkos::deep_copy(m_vtheta_dp, src.m_vtheta_dp );
  Kokkos::deep_copy(m_phinh_i  , src.m_phinh_i   );
  Kokkos::deep_copy(m_dp3d     , src.m_dp3d      );
  Kokkos::deep_copy(m_ps_v     , src.m_ps_v      );
}

void ElementsState::init(const int num_elems) {
  m_num_elems = num_elems;
  StateStorage::init_storage(num_elems);
}

void ElementsState::randomize(const int seed) {
  randomize(seed,1.0);
}

void ElementsState::randomize(const int seed, const Real max_pressure) {
  randomize(seed,max_pressure,max_pressure/100);
}

void ElementsState::randomize(const int seed,
                              const Real max_pressure,
                              const Real ps0,
                              const ExecViewUnmanaged<const Real*[NP][NP]>& phis) {
  randomize(seed,max_pressure,ps0);

  // Re-do phinh so it satisfies phinh_i(bottom)=phis

  // Sanity check
  assert(phis.extent_int(0)==m_num_elems);

  std::mt19937_64 engine(seed);

  // Note: to avoid errors in the equation of state, we need phi to be increasing.
  //       Rather than using a constraint (which may call the function many times,
  //       we simply ask that there are no duplicates, then we sort it later.
  auto sort_and_chek = [](const ExecViewManaged<Real[NUM_PHYSICAL_LEV]>::HostMirror v)->bool {
    Real* start = reinterpret_cast<Real*>(v.data());
    Real* end   = reinterpret_cast<Real*>(v.data()) + NUM_PHYSICAL_LEV;
    std::sort(start,end);
    std::reverse(start,end);
    auto it = std::unique(start,end);
    return it==end;
  };

  auto h_phis = Kokkos::create_mirror_view(phis);
  Kokkos::deep_copy(h_phis,phis);
  for (int ie=0; ie<m_num_elems; ++ie) {
    for (int igp=0; igp<NP; ++igp) {
      for (int jgp=0; jgp<NP; ++ jgp) {
        const Real phis_ij = h_phis(ie,igp,jgp);
        // Ensure generated values are larger than phis
        std::uniform_real_distribution<Real> random_dist(1.001*phis_ij,100.0*phis_ij);
        for (int itl=0; itl<NUM_TIME_LEVELS; ++itl) {
          // Get column
          auto phi_col = Homme::viewAsReal(Homme::subview(m_phinh_i,ie,itl,igp,jgp));

          // Stuff phis at the bottom
          Kokkos::deep_copy(Kokkos::subview(phi_col,NUM_PHYSICAL_LEV),phis_ij);

          // Generate values except at bottom
          ExecViewUnmanaged<Real[NUM_PHYSICAL_LEV]> phi_no_bottom(phi_col.data());
          genRandArray(phi_no_bottom,engine,random_dist,sort_and_chek);
        }
      }
    }
  }
}

void ElementsState::randomize(const int seed,
                              const Real max_pressure,
                              const Real ps0) {
  // Check elements were inited
  assert (m_num_elems>0);

  // Check data makes sense
  assert (max_pressure>ps0);
  assert (ps0>0);

  // Arbitrary minimum value to generate
  constexpr const Real min_value = 0.015625;

  std::mt19937_64 engine(seed);
  std::uniform_real_distribution<Real> random_dist(min_value, 1.0 / min_value);

  genRandArray(m_v,         engine, random_dist);
  genRandArray(m_w_i,       engine, random_dist);
  genRandArray(m_vtheta_dp, engine, random_dist);
  // Note: to avoid errors in the equation of state, we need phi to be increasing.
  //       Rather than using a constraint (which may call the function many times,
  //       we simply ask that there are no duplicates, then we sort it later.
  auto sort_and_chek = [](const ExecViewManaged<Scalar[NUM_LEV_P]>::HostMirror v)->bool {
    Real* start = reinterpret_cast<Real*>(v.data());
    Real* end   = reinterpret_cast<Real*>(v.data()) + NUM_LEV_P*VECTOR_SIZE;
    std::sort(start,end);
    std::reverse(start,end);
    auto it = std::unique(start,end);
    return it==end;
  };
  for (int ie=0; ie<m_num_elems; ++ie) {
    for (int itl=0; itl<NUM_TIME_LEVELS; ++itl) {
      for (int igp=0; igp<NP; ++igp) {
        for (int jgp=0; jgp<NP; ++ jgp) {
          genRandArray(Homme::subview(m_phinh_i,ie,itl,igp,jgp),engine,random_dist,sort_and_chek);
        }
      }
    }
  }

  // Generate ps_v so that it is >> ps0.
  genRandArray(m_ps_v, engine, std::uniform_real_distribution<Real>(100*ps0,1000*ps0));

  // This ensures the pressure in a single column is monotonically increasing
  // and has fixed upper and lower values
  const auto make_pressure_partition = [=](
      ExecViewUnmanaged<Scalar[NUM_LEV]> pt_dp) {

    auto h_pt_dp = Kokkos::create_mirror_view(pt_dp);
    Kokkos::deep_copy(h_pt_dp,pt_dp);
    Real* data = reinterpret_cast<Real*>(h_pt_dp.data());
    Real* data_end = data+NUM_PHYSICAL_LEV;

    // Put in monotonic order
    std::sort(data, data_end);

    // Check for no repetitions
    if (std::unique(data,data_end)!=data_end) {
      return false;
    }

    // Fix minimum pressure
    data[0] = min_value;

    // Compute dp from p (we assume p(last interface)=max_pressure)
    for (int i=0; i<NUM_PHYSICAL_LEV-1; ++i) {
      data[i] = data[i+1]-data[i];
    }
    data[NUM_PHYSICAL_LEV-1] = max_pressure-data[NUM_PHYSICAL_LEV-1];

    // Check that dp>=dp_min
    const Real min_dp = std::numeric_limits<Real>::epsilon()*1000;
    for (auto it=data; it!=data_end; ++it) {
      if (*it < min_dp) {
        return false;
      }
    }

    // Fill remainder of last vector pack with quiet nan's
    Real* alloc_end = data+NUM_LEV*VECTOR_SIZE;
    for (auto it=data_end; it!=alloc_end; ++it) {
      *it = std::numeric_limits<Real>::quiet_NaN();
    }

    return true;
  };

  std::uniform_real_distribution<Real> pressure_pdf(min_value, max_pressure);

  for (int ie = 0; ie < m_num_elems; ++ie) {
    // Because this constraint is difficult to satisfy for all of the tensors,
    // incrementally generate the view
    for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp) {
        for (int tl = 0; tl < NUM_TIME_LEVELS; ++tl) {
          ExecViewUnmanaged<Scalar[NUM_LEV]> pt_dp3d =
              Homme::subview(m_dp3d, ie, tl, igp, jgp);
          do {
            genRandArray(pt_dp3d, engine, pressure_pdf);
          } while (make_pressure_partition(pt_dp3d)==false);
        }
      }
    }
  }
}

void ElementsState::save_state ()
{
  if (m_state0.m_v.extent_int(0)==0) {
    m_state0.init_storage(m_num_elems);
  }
  m_state0.copy_state(*this);
}

void ElementsState::pull_from_f90_pointers (CF90Ptr& state_v,         CF90Ptr& state_w_i,
                                            CF90Ptr& state_vtheta_dp, CF90Ptr& state_phinh_i,
                                            CF90Ptr& state_dp3d,      CF90Ptr& state_ps_v) {
  HostViewUnmanaged<const Real *[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV ][2][NP][NP]> state_v_f90         (state_v,m_num_elems);
  HostViewUnmanaged<const Real *[NUM_TIME_LEVELS][NUM_INTERFACE_LEV]   [NP][NP]> state_w_i_f90       (state_w_i,m_num_elems);
  HostViewUnmanaged<const Real *[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV ]   [NP][NP]> state_vtheta_dp_f90 (state_vtheta_dp,m_num_elems);
  HostViewUnmanaged<const Real *[NUM_TIME_LEVELS][NUM_INTERFACE_LEV]   [NP][NP]> state_phinh_i_f90   (state_phinh_i,m_num_elems);
  HostViewUnmanaged<const Real *[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV ]   [NP][NP]> state_dp3d_f90      (state_dp3d,m_num_elems);
  HostViewUnmanaged<const Real *[NUM_TIME_LEVELS]                      [NP][NP]> ps_v_f90            (state_ps_v,m_num_elems);

  sync_to_device(state_v_f90,         m_v);
  sync_to_device(state_w_i_f90,       m_w_i);
  sync_to_device(state_vtheta_dp_f90, m_vtheta_dp);
  sync_to_device(state_phinh_i_f90,   m_phinh_i);
  sync_to_device(state_dp3d_f90,      m_dp3d);

  // F90 ptrs to arrays (np,np,num_time_levels,nelemd) can be stuffed directly in an unmanaged view
  // with scalar Real*[NUM_TIME_LEVELS][NP][NP] (with runtime dimension nelemd)

  auto ps_v_host = Kokkos::create_mirror_view(m_ps_v);
  Kokkos::deep_copy(ps_v_host,ps_v_f90);
  Kokkos::deep_copy(m_ps_v,ps_v_host);
}

void ElementsState::push_to_f90_pointers (F90Ptr& state_v, F90Ptr& state_w_i, F90Ptr& state_vtheta_dp,
                                          F90Ptr& state_phinh_i, F90Ptr& state_dp3d) const {
  HostViewUnmanaged<Real *[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV ][2][NP][NP]> state_v_f90         (state_v,m_num_elems);
  HostViewUnmanaged<Real *[NUM_TIME_LEVELS][NUM_INTERFACE_LEV]   [NP][NP]> state_w_i_f90       (state_w_i,m_num_elems);
  HostViewUnmanaged<Real *[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV ]   [NP][NP]> state_vtheta_dp_f90 (state_vtheta_dp,m_num_elems);
  HostViewUnmanaged<Real *[NUM_TIME_LEVELS][NUM_INTERFACE_LEV]   [NP][NP]> state_phinh_i_f90   (state_phinh_i,m_num_elems);
  HostViewUnmanaged<Real *[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV ]   [NP][NP]> state_dp3d_f90      (state_dp3d,m_num_elems);

  sync_to_host(m_v,         state_v_f90);
  sync_to_host(m_w_i,       state_w_i_f90);
  sync_to_host(m_vtheta_dp, state_vtheta_dp_f90);
  sync_to_host(m_phinh_i,   state_phinh_i_f90);
  sync_to_host(m_dp3d,      state_dp3d_f90);
}

} // namespace Homme
