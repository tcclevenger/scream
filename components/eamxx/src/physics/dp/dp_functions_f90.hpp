#ifndef SCREAM_DP_FUNCTIONS_F90_HPP
#define SCREAM_DP_FUNCTIONS_F90_HPP

#include "share/scream_types.hpp"
#include "physics/share/physics_test_data.hpp"

#include "dp_functions.hpp"
#include "physics_constants.hpp"

#include <vector>
#include <array>
#include <utility>

//
// Bridge functions to call fortran version of dp functions from C++
//

namespace scream {
namespace dp {

struct AdvanceIopForcingData : public PhysicsTestData {
  // Inputs
  Int plev, pcnst;
  Real scm_dt, ps_in;
  Real *u_in, *v_in, *t_in, *q_in, *t_phys_frc;
  
  // Outputs
  Real *u_update, *v_update, *t_update, *q_update;
  
  AdvanceIopForcingData(Int plev_, Int pcnst_, Real scm_dt_, Real ps_in_) :
    PhysicsTestData({{ plev_ }, { plev_, pcnst_ }}, {{ &u_in, &v_in, &t_in, &t_phys_frc, &u_update, &v_update, &t_update }, { &q_in, &q_update }}), plev(plev_), pcnst(pcnst_), scm_dt(scm_dt_), ps_in(ps_in_) {}
  
  PTD_STD_DEF(AdvanceIopForcingData, 4, plev, pcnst, scm_dt, ps_in);
};


struct AdvanceIopNudgingData : public PhysicsTestData {
  // Inputs
  Int plev;
  Real scm_dt, ps_in;
  Real *t_in, *q_in;
  
  // Outputs
  Real *t_update, *q_update, *relaxt, *relaxq;
  
  AdvanceIopNudgingData(Int plev_, Real scm_dt_, Real ps_in_) :
    PhysicsTestData({{ plev_ }}, {{ &t_in, &q_in, &t_update, &q_update, &relaxt, &relaxq }}), plev(plev_), scm_dt(scm_dt_), ps_in(ps_in_) {}
  
  PTD_STD_DEF(AdvanceIopNudgingData, 3, plev, scm_dt, ps_in);
};

struct AdvanceIopSubsidenceData : public PhysicsTestData {
  // Inputs
  Int plev, pcnst;
  Real scm_dt, ps_in;
  Real *u_in, *v_in, *t_in, *q_in;
  
  // Outputs
  Real *u_update, *v_update, *t_update, *q_update;
  
  AdvanceIopSubsidenceData(Int plev_, Int pcnst_, Real scm_dt_, Real ps_in_) :
    PhysicsTestData({{ plev_ }, { plev_, pcnst_ }}, {{ &u_in, &v_in, &t_in, &u_update, &v_update, &t_update }, { &q_in, &q_update }}), plev(plev_), pcnst(pcnst_), scm_dt(scm_dt_), ps_in(ps_in_) {}
  
  PTD_STD_DEF(AdvanceIopSubsidenceData, 4, plev, pcnst, scm_dt, ps_in);
};

struct IopSetinitialData : public PhysicsTestData {
  // Inputs
  Int plev, nelemd;
  
  // Inputs/Outputs
  element_t *elem;
  
  IopSetinitialData(Int plev_, Int nelemd_) :
    PhysicsTestData({}, {}), plev(plev_), nelemd(nelemd_) {}
  
  PTD_STD_DEF(IopSetinitialData, 2, plev, nelemd);
};

struct IopBroadcastData : public PhysicsTestData {
  // Inputs
  Int plev;

  IopBroadcastData(Int plev_=0) :
    PhysicsTestData({}, {}), plev(plev_) {}
  
  PTD_STD_DEF(IopBroadcastData, 1, plev);
};

struct ApplyIopForcingData : public PhysicsTestData {
  // Inputs
  Int plev, nelemd, n, nets, nete;
  hybrid_t hybrid;
  timelevel_t tl;
  bool t_before_advance;
  
  // Inputs/Outputs
  element_t *elem;
  hvcoord_t hvcoord;
  
  ApplyIopForcingData(Int plev_, Int nelemd_, Int n_, Int nets_, Int nete_, bool t_before_advance_) :
    PhysicsTestData({}, {}), plev(plev_), nelemd(nelemd_), n(n_), nets(nets_), nete(nete_), t_before_advance(t_before_advance_) {}
  
  PTD_STD_DEF(ApplyIopForcingData, 6, plev, nelemd, n, nets, nete, t_before_advance);
};

struct IopDomainRelaxationData : public PhysicsTestData {
  // Inputs
  Int nelemd, np, nlev, t1, nelemd_todo, np_todo;
  hvcoord_t hvcoord;
  hybrid_t hybrid;
  Real dt;
  
  // Inputs/Outputs
  element_t *elem;
  Real *dp;
  
  IopDomainRelaxationData(Int nelemd_, Int np_, Int nlev_, Int t1_, Int nelemd_todo_, Int np_todo_, Real dt_) :
    PhysicsTestData({{ np_, np_, nlev_ }}, {{ &dp }}), nelemd(nelemd_), np(np_), nlev(nlev_), t1(t1_), nelemd_todo(nelemd_todo_), np_todo(np_todo_), dt(dt_) {}
  
  PTD_STD_DEF(IopDomainRelaxationData, 7, nelemd, np, nlev, t1, nelemd_todo, np_todo, dt);
};

// Glue functions to call fortran from from C++ with the Data struct

void advance_iop_forcing(AdvanceIopForcingData& d);
void advance_iop_nudging(AdvanceIopNudgingData& d);
void advance_iop_subsidence(AdvanceIopSubsidenceData& d);
void iop_setinitial(IopSetinitialData& d);
void iop_broadcast(IopBroadcastData& d);
void apply_iop_forcing(ApplyIopForcingData& d);
void iop_domain_relaxation(IopDomainRelaxationData& d);
extern "C" { // _f function decls

void advance_iop_forcing_f(Int plev, Int pcnst, Real scm_dt, Real ps_in, Real* u_in, Real* v_in, Real* t_in, Real* q_in, Real* t_phys_frc, Real* u_update, Real* v_update, Real* t_update, Real* q_update);
void advance_iop_nudging_f(Int plev, Real scm_dt, Real ps_in, Real* t_in, Real* q_in, Real* t_update, Real* q_update, Real* relaxt, Real* relaxq);
void advance_iop_subsidence_f(Int plev, Int pcnst, Real scm_dt, Real ps_in, Real* u_in, Real* v_in, Real* t_in, Real* q_in, Real* u_update, Real* v_update, Real* t_update, Real* q_update);
void iop_setinitial_f(Int nelemd, element_t* elem);
void iop_broadcast_f();
void apply_iop_forcing_f(Int nelemd, element_t* elem, hvcoord_t* hvcoord, hybrid_t hybrid, timelevel_t tl, Int n, bool t_before_advance, Int nets, Int nete);
void iop_domain_relaxation_f(Int nelemd, Int np, Int nlev, element_t* elem, hvcoord_t hvcoord, hybrid_t hybrid, Int t1, Real* dp, Int nelemd_todo, Int np_todo, Real dt);
} // end _f function decls

}  // namespace dp
}  // namespace scream

#endif // SCREAM_DP_FUNCTIONS_F90_HPP
