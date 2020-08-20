#include <iostream>
#include <cmath>
#include <catch2/catch.hpp>
#include "ekat/scream_pack.hpp"
#include "share/atm_process/atmosphere_process.hpp"
#include "share/grid/user_provided_grids_manager.hpp"
#include "share/grid/se_grid.hpp"
#include "control/atmosphere_driver.hpp"

#include "physics/rrtmgp/atmosphere_radiation.hpp"
#include "physics/rrtmgp/scream_rrtmgp_interface.hpp"
#include "netcdf.h"
#include "mo_gas_concentrations.h"
#include "mo_garand_atmos_io.h"
#include "mo_fluxes.h"
#include "mo_cloud_optics.h"
#include "FortranIntrinsics.h"

#include "rrtmgp_test_utils.hpp"

/*
 * This will eventually contain a standalone test for the RRTMGP driver
 * As of now, it is just a shell that at least requires RRTMGP to be built
 * with the SCREAM build and test system.
 */

// #include "physics/rrtmgp/rrtmgp_functions_f90.hpp"

namespace scream {
    // A dummy physics grids for this test //
    class DummyPhysicsGrid : public SEGrid {
        public: DummyPhysicsGrid (const int num_cols) : SEGrid("Physics",GridType::SE_NodeBased,num_cols) {
            // Nothing to do here
        }
        ~DummyPhysicsGrid () = default;
    };

    // Add the RRTMGP stand-alone driver test
    TEST_CASE("rrtmgp_stand_alone", "") {
        using namespace scream;
        using namespace scream::control;

        constexpr int num_iter = 20;
        constexpr int num_cols = 10;

        /* 
         * Setup driver stuff
         */

        // Setup parameter list for inputs to the radiation interface
        ParameterList ad_params("Atmosphere Driver");
        auto& proc_params = ad_params.sublist("Atmosphere Processes");
        proc_params.set("Number of Entries", 1);
        proc_params.set<std::string>("Schedule Type", "Sequential");
        auto& p0 = proc_params.sublist("Process 0");
        p0.set<std::string>("Process Name", "RRTMGP");
        p0.set<std::string>("Grid","Physics");

        auto& gm_params = ad_params.sublist("Grids Manager");
        gm_params.set<std::string>("Type","User Provided");
        gm_params.set<std::string>("Reference Grid","Physics");

        // Need to register products in the factory *before* we create any AtmosphereProcessGroup,
        // which rely on factory for process creation. The initialize method of the AD does that.
        auto& proc_factory = AtmosphereProcessFactory::instance();
        proc_factory.register_product("RRTMGP",&create_atmosphere_process<RRTMGPRadiation>);

        // Need to register grids managers before we create the driver
        auto& gm_factory = GridsManagerFactory::instance();
        gm_factory.register_product("User Provided",create_user_provided_grids_manager);

        // Set the dummy grid in the UserProvidedGridManager
        // Recall that this class stores *static* members, so whatever
        // we set here, will be reflected in the GM built by the factory.
        UserProvidedGridsManager upgm;
        upgm.set_grid(std::make_shared<DummyPhysicsGrid>(num_cols));
        upgm.set_reference_grid("Physics");

        // Create a MPI communicator
        Comm atm_comm (MPI_COMM_WORLD);

        // Create the driver
        AtmosphereDriver ad;

        // Dummy timestamp
        util::TimeStamp time (0,0,0,0);

        // Initialize the driver, run the driver, cleanup
        ad.initialize(atm_comm, ad_params, time);
        ad.run(300.0);
        ad.finalize();
        upgm.clean_up();

        // Setup for standalone (dummy) problem

        // Get reference fluxes from input file; do this here so we can get ncol dimension
        std::string inputfile = "./data/rrtmgp-allsky.nc";
        real2d sw_flux_up_ref;
        real2d sw_flux_dn_ref;
        real2d sw_flux_dn_dir_ref;
        real2d lw_flux_up_ref;
        real2d lw_flux_dn_ref;
        rrtmgpTest::read_fluxes(inputfile, sw_flux_up_ref, sw_flux_dn_ref, sw_flux_dn_dir_ref, lw_flux_up_ref, lw_flux_dn_ref );

        // Get dimension sizes
        int ncol = sw_flux_up_ref.dimension[0];
        int nlev = sw_flux_up_ref.dimension[1];
        int nlay = nlev - 1;

        // Read in dummy Garand atmosphere; if this were an actual model simulation,
        // these would be passed as inputs to the driver
        // NOTE: set ncol to size of col_flx dimension in the input file. This is so
        // that we can compare to the reference data provided in that file. Note that
        // this will copy the first column of the input data (the first profile) ncol
        // times. We will then fill some fraction of these columns with clouds for
        // the test problem.
        real2d p_lay;
        real2d t_lay;
        real2d p_lev;
        real2d t_lev;
        real2d col_dry;
        GasConcs gas_concs;
        real2d sfc_alb_dir;
        real2d sfc_alb_dif;
        real1d mu0;
        real2d lwp;
        real2d iwp;
        real2d rel;
        real2d rei;
        rrtmgpTest::dummy_atmos(
            inputfile, ncol, p_lay, t_lay, p_lev, t_lev, gas_concs, col_dry,
            sfc_alb_dir, sfc_alb_dif, mu0,
            lwp, iwp, rel, rei
        );

        // Setup flux outputs; In a real model run, the fluxes would be
        // input/outputs into the driver (persisting between calls), and
        // we would just have to setup the pointers to them in the
        // FluxesBroadband object
        real2d sw_flux_up ("sw_flux_up" ,ncol,nlay+1);
        real2d sw_flux_dn ("sw_flux_dn" ,ncol,nlay+1);
        real2d sw_flux_dn_dir("sw_flux_dn_dir",ncol,nlay+1);
        real2d lw_flux_up ("lw_flux_up" ,ncol,nlay+1);
        real2d lw_flux_dn ("lw_flux_dn" ,ncol,nlay+1);

        // Run RRTMGP standalone codes and compare with AD run
        // Do something interesting here...
        // NOTE: these will get replaced with AD stuff that handles these
        rrtmgp::rrtmgp_initialize();
        rrtmgp::rrtmgp_main(
            p_lay, t_lay, p_lev, t_lev, gas_concs, col_dry,
            sfc_alb_dir, sfc_alb_dif, mu0,
            lwp, iwp, rel, rei,
            sw_flux_up, sw_flux_dn, sw_flux_dn_dir,
            lw_flux_up, lw_flux_dn
        );
        rrtmgp::rrtmgp_finalize();

        // Check values
        REQUIRE(rrtmgpTest::all_equals(sw_flux_up_ref    , sw_flux_up    ));
        REQUIRE(rrtmgpTest::all_equals(sw_flux_dn_ref    , sw_flux_dn    ));
        REQUIRE(rrtmgpTest::all_equals(sw_flux_dn_dir_ref, sw_flux_dn_dir));
        REQUIRE(rrtmgpTest::all_equals(lw_flux_up_ref    , lw_flux_up    ));
        REQUIRE(rrtmgpTest::all_equals(lw_flux_dn_ref    , lw_flux_dn    ));

    }
}
