#include "physics/p3/p3_standalone_field_initializer.hpp"
#include "physics/p3/scream_p3_interface.hpp"

#include <array>

namespace scream
{

void P3StandAloneInit::add_field (const field_type &f)
{
  const auto& id = f.get_header().get_identifier();
  
  m_fields.emplace(id.name(),f);
  m_fields_id.insert(id);
}

// =========================================================================================
void P3StandAloneInit::initialize_fields ()
{
  // Safety check: if we're asked to init anything at all,
  // then we should have been asked to init 7 fields.
  int count = 0;
  count += m_fields.count("q");
  count += m_fields.count("T");
  count += m_fields.count("ast");
  count += m_fields.count("naai");
  count += m_fields.count("npccn");
  count += m_fields.count("pmid");
  count += m_fields.count("dp");
  count += m_fields.count("zi");

  if (count==0) {
    return;
  }

  scream_require_msg (count==8,
    "Error! P3StandAloneInit is expected to init 'q','T','ast','naai','npccn','pmid','dp','zi'.\n"
    "       Only " + std::to_string(count) + " of those have been found.\n"
    "       Please, check the atmosphere processes you are using,"
    "       and make sure they agree on who's initializing each field.\n");

  // Get device views
  auto d_q     = m_fields.at("q").get_view();
  auto d_T     = m_fields.at("T").get_view();
  auto d_ast   = m_fields.at("ast").get_view();
  auto d_naai  = m_fields.at("naai").get_view();
  auto d_npccn = m_fields.at("npccn").get_view();
  auto d_pmid  = m_fields.at("pmid").get_view();
  auto d_pdel  = m_fields.at("dp").get_view();
  auto d_zi    = m_fields.at("zi").get_view();

  // Create host mirrors
  auto h_q     = Kokkos::create_mirror_view(d_q);
  auto h_T     = Kokkos::create_mirror_view(d_T);
  auto h_ast   = Kokkos::create_mirror_view(d_ast);
  auto h_naai  = Kokkos::create_mirror_view(d_naai);
  auto h_npccn = Kokkos::create_mirror_view(d_npccn);
  auto h_pmid  = Kokkos::create_mirror_view(d_pmid);
  auto h_pdel  = Kokkos::create_mirror_view(d_pdel);
  auto h_zi    = Kokkos::create_mirror_view(d_zi);

  // Get host mirros' raw pointers
  auto q     = h_q.data();
  auto T     = h_T.data();
  auto ast   = h_ast.data();
  auto naai  = h_naai.data();
  auto npccn = h_npccn.data();
  auto pmid  = h_pmid.data();
  auto pdel  = h_pdel.data();
  auto zi    = h_zi.data();

  // Call f90 routine
  p3_standalone_init_f90 (q, T, zi, pmid, pdel, ast, naai, npccn);

  // Deep copy back to device
  Kokkos::deep_copy(d_q,h_q);
  Kokkos::deep_copy(d_T,h_T);
  Kokkos::deep_copy(d_ast,h_ast);
  Kokkos::deep_copy(d_naai,h_naai);
  Kokkos::deep_copy(d_npccn,h_npccn);
  Kokkos::deep_copy(d_pmid,h_pmid);
  Kokkos::deep_copy(d_pdel,h_pdel);
  Kokkos::deep_copy(d_zi,h_zi);

  // If we are in charge of init-ing FQ as well, init it to 0.
  if (m_fields.count("FQ")==1) {
    // Init FQ to 0
    auto d_FQ = m_fields.at("FQ").get_view();
    Kokkos::deep_copy(d_FQ,Real(0));
  }
}

} // namespace scream