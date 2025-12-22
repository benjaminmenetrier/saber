/*
 * (C) Copyright 2025- UCAR
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include "saber/blocks/SaberHybridBlockChain.h"

#include "atlas/field.h"

#include "oops/base/FieldSet4D.h"
#include "oops/util/FieldSetOperations.h"
#include "oops/util/Logger.h"
#include "oops/util/Timer.h"

#include "saber/oops/Utilities.h"

namespace saber {

// -----------------------------------------------------------------------------

void SaberHybridBlockChain::randomize(oops::FieldSet4D & fset4d) const {
  oops::Log::trace() << "SaberHybridBlockChain::randomize starting" << std::endl;
  util::Timer timer("SaberHybridBlockChain", "randomize");

  if (parallelHybrid_) {
    // Run components of the central block in parallel
    oops::Log::debug() << "Parallel execution of doRandomize in Hybrid" << std::endl;
    oops::Log::debug() << "Running Hybrid component " << myComponent_ + 1 << std::endl;
    ASSERT(hybridBlockChain_.size() == 1);

    // global communicator and functionSpace
    const auto & defaultSpaceComm = fset4d[0].commGeom();

    // check global communicator is the default one for atlas MPI
    ASSERT(eckit::mpi::comm().name() == defaultSpaceComm.name());

    // subcommunicator within this component
    const auto spaceCommName = "comm_space_" + std::to_string(myComponent_);
    const auto & localSpaceComm = eckit::mpi::comm(spaceCommName.c_str());

    // Set up atlas MPI
    eckit::mpi::setCommDefault(localSpaceComm.name().c_str());

    // Create temporary FieldSet on subcommunicator
    oops::FieldSet4D fset4dCmp(fset4d.times(), fset4d.commTime(), localSpaceComm);

    hybridBlockChain_[0]->randomize(fset4dCmp);

    // Weight square-root multiplication
    if (hybridScalarWeightSqrt_[0] != 1.0) {
      // Scalar weight
      fset4dCmp *= hybridScalarWeightSqrt_[0];
    }
    if (!hybridFieldWeightSqrt_[0].empty()) {
      // File-based weight
      fset4dCmp *= hybridFieldWeightSqrt_[0];
    }

    // Add components
    defaultSpaceComm.barrier();

    for (size_t jtime = 0; jtime < fset4dCmp.size(); jtime++) {
      // Redistribute to global communicator and sum
       util::gatherAndSumFromSubcommunicator(fset4dCmp[jtime].fieldSet(),
                                             fset4d[jtime].fieldSet(),
                                             localSpaceComm,
                                             defaultSpaceComm,
                                             *localHybridFs_,
                                             *globalHybridFs_);
    }

    // Restore atlas MPI to previous
    eckit::mpi::setCommDefault(defaultSpaceComm.name().c_str());

    fset4d += fset4dCmp;
  } else {
    // Loop over components for the central block
    for (size_t jj = 0; jj < hybridBlockChain_.size(); ++jj) {
      // Randomize covariance
      oops::FieldSet4D fset4dCmp(fset4d.times(), fset4d.commTime(), fset4d[0].commGeom());
      hybridBlockChain_[jj]->randomize(fset4dCmp);

      // Weight square-root multiplication
      if (hybridScalarWeightSqrt_[jj] != 1.0) {
        // Scalar weight
        fset4dCmp *= hybridScalarWeightSqrt_[jj];
      }
      if (!hybridFieldWeightSqrt_[jj].empty()) {
        // File-based weight
        fset4dCmp *= hybridFieldWeightSqrt_[jj];
      }

      // Add component
      fset4d += fset4dCmp;
    }
  }

  if (outerBlockChain_) outerBlockChain_->applyOuterBlocks(fset4d);

  oops::Log::trace() << "SaberHybridBlockChain::randomize done" << std::endl;
}

// -----------------------------------------------------------------------------

void SaberHybridBlockChain::multiply(oops::FieldSet4D & fset4d) const {
  oops::Log::trace() << "SaberHybridBlockChain::multiply starting" << std::endl;
  util::Timer timer("SaberHybridBlockChain", "multiply");

  // Apply outer blocks adjoint
  if (outerBlockChain_) outerBlockChain_->applyOuterBlocksAD(fset4d);

  // Initialize sum to zero
  oops::FieldSet4D fset4dSum = oops::copyFieldSet4D(fset4d);
  fset4dSum.zero();

  // Loop over B components
  if (parallelHybrid_) {
    oops::Log::debug() << "Parallel execution of Hybrid::multiply, component "
                       << myComponent_ + 1 << std::endl;
    ASSERT(hybridBlockChain_.size() == 1);
    ASSERT(hybridScalarWeightSqrt_.size() == 1);
    ASSERT(hybridFieldWeightSqrt_.size() == 1);

    // Global communicator
    const auto & defaultSpaceComm = fset4d[0].commGeom();
    ASSERT(defaultSpaceComm.name() == eckit::mpi::comm().name());

    // Subcommunicator within component
    const std::string spaceCommName = "comm_space_" + std::to_string(myComponent_);
    const auto & localSpaceComm = eckit::mpi::comm(spaceCommName.c_str());

    // Create temporary FieldSet copy on communicator of this component
    oops::FieldSet4D fset4dCmp(fset4d.times(), fset4d.commTime(), localSpaceComm);
    for (size_t jtime = 0; jtime < fset4dCmp.size(); jtime++) {
      util::redistributeToSubcommunicator(fset4d[jtime].fieldSet(),
                                          fset4dCmp[jtime].fieldSet(),
                                          defaultSpaceComm,
                                          localSpaceComm,
                                          *globalHybridFs_,
                                          *localHybridFs_);
    }

    // Set up atlas MPI
    eckit::mpi::setCommDefault(localSpaceComm.name().c_str());

    // Apply weight
    if (hybridScalarWeightSqrt_[0] != 1.0) {
      // Scalar weight
      fset4dCmp *= hybridScalarWeightSqrt_[0];
    }
    if (!hybridFieldWeightSqrt_[0].empty()) {
      // File-based weight
      fset4dCmp *= hybridFieldWeightSqrt_[0];
    }

    // Apply covariance
    hybridBlockChain_[0]->multiply(fset4dCmp);

    // Apply weight
    if (hybridScalarWeightSqrt_[0] != 1.0) {
      // Scalar weight
      fset4dCmp *= hybridScalarWeightSqrt_[0];
    }
    if (!hybridFieldWeightSqrt_[0].empty()) {
      // File-based weight
      fset4dCmp *= hybridFieldWeightSqrt_[0];
    }

    // Wait for all components to have finished multiplying
    defaultSpaceComm.barrier();

    // Gather and sum data across components
    for (size_t jtime = 0; jtime < fset4dCmp.size(); jtime++) {
      util::gatherAndSumFromSubcommunicator(fset4dCmp[jtime].fieldSet(),
                                            fset4dSum[jtime].fieldSet(),
                                            localSpaceComm,
                                            defaultSpaceComm,
                                            *localHybridFs_,
                                            *globalHybridFs_);
    }

    // Set back default MPI communicator
    eckit::mpi::setCommDefault(defaultSpaceComm.name().c_str());

  } else {
    if (hybridBlockChain_.size() > 1) {
        oops::Log::debug() << "Serial execution of Hybrid::multiply" << std::endl;
    }
    for (size_t jj = 0; jj < hybridBlockChain_.size(); ++jj) {
      // Create temporary FieldSet
      oops::FieldSet4D fset4dCmp = oops::copyFieldSet4D(fset4d);

      // Apply weight
      if (hybridScalarWeightSqrt_[jj] != 1.0) {
        // Scalar weight
        fset4dCmp *= hybridScalarWeightSqrt_[jj];
      }
      if (!hybridFieldWeightSqrt_[jj].empty()) {
        // File-based weight
        fset4dCmp *= hybridFieldWeightSqrt_[jj];
      }

      // Apply covariance
      hybridBlockChain_[jj]->multiply(fset4dCmp);

      // Apply weight
      if (hybridScalarWeightSqrt_[jj] != 1.0) {
        // Scalar weight
        fset4dCmp *= hybridScalarWeightSqrt_[jj];
      }
      if (!hybridFieldWeightSqrt_[jj].empty()) {
        // File-based weight
        fset4dCmp *= hybridFieldWeightSqrt_[jj];
      }

      // Add component
      fset4dSum += fset4dCmp;
    }
  }

  // Apply outer blocks forward
  if (outerBlockChain_) outerBlockChain_->applyOuterBlocks(fset4dSum);

  fset4d.deepCopy(fset4dSum);

  oops::Log::trace() << "SaberHybridBlockChain::multiply done" << std::endl;
}

// -----------------------------------------------------------------------------

}  // namespace saber
