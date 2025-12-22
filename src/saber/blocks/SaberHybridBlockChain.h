/*
 * (C) Copyright 2025- UCAR
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "atlas/field.h"

#include "eckit/config/LocalConfiguration.h"
#include "eckit/exception/Exceptions.h"
#include "eckit/mpi/Comm.h"

#include "oops/base/FieldSet3D.h"
#include "oops/base/FieldSet4D.h"
#include "oops/base/FieldSets.h"
#include "oops/base/Geometry.h"
#include "oops/base/State4D.h"
#include "oops/base/Variables.h"
#include "oops/util/FieldSetOperations.h"
#include "oops/util/FieldSetSubCommunicators.h"
#include "oops/util/Logger.h"
#include "oops/util/Timer.h"

#include "saber/blocks/SaberBlockChainBase.h"
#include "saber/blocks/SaberBlockParametersBase.h"
#include "saber/blocks/SaberOuterBlockChain.h"
#include "saber/oops/Utilities.h"

namespace saber {

// Forward declarations
template <typename MODEL> class SaberBlockChainFactory;

// -----------------------------------------------------------------------------

inline std::string parametricIfNotEnsemble(const std::string & blockName) {
  if (blockName == "Ensemble")
    return blockName;
  else if (blockName == "Hybrid")
    return blockName;
  else if (blockName == "gsi hybrid covariance")
    return "GSI";
  else
    return "Parametric";
}


/// Hybrid covariance block chain implementation
class SaberHybridBlockChain : public SaberBlockChainBase {
 public:
  template<typename MODEL>
  SaberHybridBlockChain(const oops::Geometry<MODEL> & geom,
                        const oops::Variables & outerVars,
                        oops::FieldSet4D & fset4dXb,
                        oops::FieldSet4D & fset4dFg,
                        oops::FieldSets & fsetEns,
                        const eckit::LocalConfiguration & covarConf,
                        const eckit::Configuration & conf);
  ~SaberHybridBlockChain() = default;

  /// @brief Randomize the increment according to this hybrid B matrix.
  void randomize(oops::FieldSet4D &) const override;
  /// @brief Multiply the increment by this hybrid B matrix.
  void multiply(oops::FieldSet4D &) const override;

  /// @brief Control vector size
  size_t ctlVecSize() const override {return 0;}
  /// @brief Square-root multiplication
  void multiplySqrt(const atlas::Field &, oops::FieldSet4D &, const size_t &) const override {}
  /// @brief Adjoint of square-root multiplication
  void multiplySqrtAD(const oops::FieldSet4D &, atlas::Field &, const size_t &) const override {}

  /// @brief Accessor to outer function space
  const atlas::FunctionSpace & outerFunctionSpace() const override {return outerFunctionSpace_;}
  /// @brief Accessor to outer variables
  const oops::Variables & outerVariables() const override {return outerVariables_;}

 private:
  /// Function space
  const atlas::FunctionSpace & outerFunctionSpace_;
  /// Variables
  const oops::Variables & outerVariables_;

  /// Chain of outer blocks applied to all components of hybrid covariances.
  std::unique_ptr<SaberOuterBlockChain> outerBlockChain_;
  /// Vector of hybrid B components.
  std::vector<std::unique_ptr<SaberBlockChainBase>> hybridBlockChain_;
  /// Vector of scalar weights for hybrid B components.
  std::vector<double> hybridScalarWeightSqrt_;
  /// Vector of field weights for hybrid B components.
  std::vector<oops::FieldSet3D> hybridFieldWeightSqrt_;

  /// Whether to run Hybrid in parallel
  bool parallelHybrid_;
  /// Index of component if running Hybrid in parallel
  size_t myComponent_;
  /// local geometry for parallel Hybrid block (type-erased)
  std::shared_ptr<atlas::FunctionSpace> localHybridFs_, globalHybridFs_;
};

// -----------------------------------------------------------------------------

template<typename MODEL>
SaberHybridBlockChain::SaberHybridBlockChain(const oops::Geometry<MODEL> & geom,
                       const oops::Variables & outerVars,
                       oops::FieldSet4D & fset4dXb,
                       oops::FieldSet4D & fset4dFg,
                       oops::FieldSets & fsetEns,
                       const eckit::LocalConfiguration & covarConf,
                       const eckit::Configuration & conf)
  : outerFunctionSpace_(geom.functionSpace()), outerVariables_(outerVars),
    parallelHybrid_(false), myComponent_(0) {
  oops::Log::trace() << "SaberHybridBlockChain ctor starting" << std::endl;
  oops::Variables currentOuterVars(outerVars);
  // Build common (for all hybrid components) outer blocks if they exist
  if (conf.has("saber outer blocks")) {
    std::vector<SaberOuterBlockParametersWrapper> cmpOuterBlocksParams;
    for (const auto & cmpOuterBlockConf : conf.getSubConfigurations("saber outer blocks")) {
      SaberOuterBlockParametersWrapper cmpOuterBlockParamsWrapper;
      cmpOuterBlockParamsWrapper.deserialize(cmpOuterBlockConf);
      cmpOuterBlocksParams.push_back(cmpOuterBlockParamsWrapper);
    }
    outerBlockChain_ = std::make_unique<SaberOuterBlockChain>(geom, outerVariables_,
                          fset4dXb, fset4dFg, fsetEns, covarConf,
                          cmpOuterBlocksParams);
    currentOuterVars = outerBlockChain_->innerVars();
  }

  // Hybrid central block
  parallelHybrid_ = conf.getBool("saber central block.run in parallel", false);
  const eckit::mpi::Comm & defaultSpaceComm = geom.getComm();
  const size_t ntasks = defaultSpaceComm.size();
  const size_t nComponents = conf.getSubConfigurations("saber central block.components").size();

  std::vector<double> parallelCovRelativeCpuWeight =
      conf.has("parallel covariance relative cpu weight") ?
      conf.getDoubleVector("parallel covariance relative cpu weight") :
  std::vector<double>(nComponents,
                      1.0 / static_cast<double>(nComponents));

  std::vector<size_t> ntasksPerComponent(nComponents, 0);
  std::vector<size_t> globalTaskOffsetPerComponent(nComponents+1, 0);
  if (parallelHybrid_) {
    oops::Log::info() << "Info     : Creating Hybrid block in parallel" << std::endl;
    // checks
    ASSERT(nComponents == parallelCovRelativeCpuWeight.size());

    // need to check to ensure that the total sum of PEs over components is consistent
    // with the MPI size on the default communicator and that each component
    // has a minimum MPI size of 1.
    for (size_t component = 0; component < nComponents; ++component) {
      ntasksPerComponent[component] =
        std::round(parallelCovRelativeCpuWeight[component] * ntasks);
      ASSERT(ntasksPerComponent[component] > 0);
    }
    int discrepencyPE =
      std::accumulate(ntasksPerComponent.begin(), ntasksPerComponent.end(), 0) - ntasks;

    for (size_t component = 0; component < nComponents && discrepencyPE != 0; ++component) {
      if (discrepencyPE > 0 && ntasksPerComponent[component] >= 2) {
        ntasksPerComponent[component] -= 1;
        discrepencyPE -= 1;
      } else if (discrepencyPE < 0) {
        ntasksPerComponent[component] += 1;
        discrepencyPE += 1;
      }
    }

    ASSERT(std::accumulate(ntasksPerComponent.begin(),
                           ntasksPerComponent.end(), 0) - ntasks == 0);

    for (size_t component = 1; component < nComponents; ++component) {
      globalTaskOffsetPerComponent[component] =
        globalTaskOffsetPerComponent[component-1] + ntasksPerComponent[component-1];
    }
    globalTaskOffsetPerComponent[nComponents] = ntasks;

    const eckit::mpi::Comm & initialDefaultComm = eckit::mpi::comm();
    ASSERT(initialDefaultComm.name() == defaultSpaceComm.name());

    // We split the space communicators only, the time parallelization is untouched
    const size_t myTask = defaultSpaceComm.rank();

    // Set myComponent_  tasksPerComponent
    size_t tasksPerComponent;
    for (size_t component = 0; component < nComponents; ++component) {
      if ((myTask >= globalTaskOffsetPerComponent[component]) &&
          (myTask < globalTaskOffsetPerComponent[component+1])) {
        myComponent_ = component;
        tasksPerComponent = ntasksPerComponent[component];
      }
    }

    oops::Log::info() << "Info     : Creating component " << myComponent_ + 1
                      << "/" << nComponents
                      << " of Hybrid block using " << tasksPerComponent
                      << " MPI tasks." << std::endl;

    // Create communicators for same component, for communications in space
    const auto spaceCommName = ("comm_space_" + std::to_string(myComponent_));
    if (eckit::mpi::hasComm(spaceCommName.c_str())) {
      eckit::mpi::deleteComm(spaceCommName.c_str());
    }
    const auto & localSpaceComm = defaultSpaceComm.split(myComponent_, spaceCommName.c_str());

    // Set up default MPI communicator for atlas
    eckit::mpi::setCommDefault(localSpaceComm.name().c_str());

    // Create block geometry (needed for ensemble reading and local geometries)
    if (!conf.has("saber central block.geometry")) {
      throw eckit::UserError("Parallel hybrid block requires geometry key", Here());
    }
    const auto geomConf = conf.getSubConfiguration("saber central block.geometry");
    // The hybrid Geometry is stored as a class member to ensure it doesn't go
    // out of scope after construction, as it is directly used (not copied) by
    // the hybrid Block Chains.
    oops::Geometry<MODEL> localHybridGeom(geomConf, localSpaceComm, geom.timeComm());
    localHybridFs_.reset(new atlas::FunctionSpace(localHybridGeom.functionSpace()));
    globalHybridFs_.reset(new atlas::FunctionSpace(geom.functionSpace()));
    // Copy and redistribute the background and first guess
    oops::FieldSet4D localFset4dXb(fset4dXb.times(), fset4dXb.commTime(),
                                   localSpaceComm, *localHybridFs_, fset4dXb.variables());
    oops::FieldSet4D localFset4dFg(fset4dFg.times(), fset4dFg.commTime(),
                                   localSpaceComm, *localHybridFs_, fset4dFg.variables());

    for (size_t jtime = 0; jtime < fset4dXb.size(); jtime++) {
      util::redistributeToSubcommunicator(fset4dXb[jtime].fieldSet(),
                                          localFset4dXb[jtime].fieldSet(),
                                          defaultSpaceComm,
                                          localSpaceComm,
                                          geom.functionSpace(),
                                          *localHybridFs_);
      util::redistributeToSubcommunicator(fset4dFg[jtime].fieldSet(),
                                          localFset4dFg[jtime].fieldSet(),
                                          defaultSpaceComm,
                                          localSpaceComm,
                                          geom.functionSpace(),
                                          *localHybridFs_);
    }
    defaultSpaceComm.barrier();

    const auto cmp = conf.getSubConfigurations("saber central block.components")[myComponent_];

    // Initialize component outer variables
    const oops::Variables cmpOuterVars(currentOuterVars);

    // Set weight
    eckit::LocalConfiguration weightConf = cmp.getSubConfiguration("weight");
    // Scalar weight
    hybridScalarWeightSqrt_.push_back(std::sqrt(weightConf.getDouble("value", 1.0)));

    // File-base weight
    oops::FieldSet3D fsetWeight(localFset4dXb[0].validTime(), localSpaceComm);
    if (weightConf.has("file")) {
      // File-base weight
      readHybridWeight(localHybridGeom,
                       cmpOuterVars,
                       localFset4dXb[0].validTime(),
                       weightConf.getSubConfiguration("file"),
                       fsetWeight);
      fsetWeight.sqrt();
    }
    hybridFieldWeightSqrt_.push_back(fsetWeight);

    // Set covariance
    eckit::LocalConfiguration cmpConf = cmp.getSubConfiguration("covariance");

    // Read ensemble
    eckit::LocalConfiguration cmpEnsembleConf;
    const bool iterativeLoading = covarConf.getBool("iterative ensemble loading", false);
    oops::FieldSets localFset4dCmpEns
         = readEnsemble(localHybridGeom,
                        cmpOuterVars,
                        localFset4dXb.times(), localFset4dXb.commTime(), localFset4dXb.commEns(),
                        cmpConf,
                        iterativeLoading,
                        cmpEnsembleConf);

    // Create internal configuration
    eckit::LocalConfiguration cmpCovarConf(covarConf);
    cmpCovarConf.set("ensemble configuration", cmpEnsembleConf);

    SaberCentralBlockParametersWrapper cmpCentralBlockParamsWrapper;
    cmpCentralBlockParamsWrapper.deserialize(cmpConf.getSubConfiguration("saber central block"));
    const auto & centralBlockParams =
                 cmpCentralBlockParamsWrapper.saberCentralBlockParameters.value();

    hybridBlockChain_.push_back(
        SaberBlockChainFactory<MODEL>::create
         (parametricIfNotEnsemble(centralBlockParams.saberBlockName.value()),
          localHybridGeom,
          cmpOuterVars,
          localFset4dXb,
          localFset4dFg,
          localFset4dCmpEns,
          cmpCovarConf,
          cmpConf));

    ASSERT(hybridBlockChain_.size() > 0);

    // Restore previous default MPI communicator for atlas
    eckit::mpi::setCommDefault(defaultSpaceComm.name().c_str());
  } else {
    oops::Log::info() << "Info     : Creating Hybrid block serially" << std::endl;
    // Create block geometry (needed for ensemble reading)
    const oops::Geometry<MODEL> * hybridGeom = &geom;
    if (conf.has("saber central block.geometry")) {
      hybridGeom = new oops::Geometry<MODEL>(
        conf.getSubConfiguration("saber central block.geometry"),
        geom.getComm());
    }
    for (const auto & cmp : conf.getSubConfigurations("saber central block.components")) {
      // Initialize component outer variables
      const oops::Variables cmpOuterVars(currentOuterVars);

      // Set weight
      eckit::LocalConfiguration weightConf = cmp.getSubConfiguration("weight");
      // Scalar weight
      hybridScalarWeightSqrt_.push_back(std::sqrt(weightConf.getDouble("value", 1.0)));
      // File-base weight
      oops::FieldSet3D fsetWeight(fset4dXb[0].validTime(), geom.getComm());
      if (weightConf.has("file")) {
        // File-base weight
        readHybridWeight(*hybridGeom,
                         cmpOuterVars,
                         fset4dXb[0].validTime(),
                         weightConf.getSubConfiguration("file"),
                         fsetWeight);
        fsetWeight.sqrt();
      }
      hybridFieldWeightSqrt_.push_back(fsetWeight);

      // Set covariance
      eckit::LocalConfiguration cmpConf = cmp.getSubConfiguration("covariance");

      // Read ensemble
      eckit::LocalConfiguration cmpEnsembleConf;
      const bool iterativeLoading = covarConf.getBool("iterative ensemble loading", false);
      oops::FieldSets fset4dCmpEns
           = readEnsemble(*hybridGeom,
                          cmpOuterVars,
                          fset4dXb.times(), fset4dXb.commTime(), fset4dXb.commEns(),
                          cmpConf,
                          iterativeLoading,
                          cmpEnsembleConf);

      // Create internal configuration
      eckit::LocalConfiguration cmpCovarConf(covarConf);
      cmpCovarConf.set("ensemble configuration", cmpEnsembleConf);

      SaberCentralBlockParametersWrapper cmpCentralBlockParamsWrapper;
      cmpCentralBlockParamsWrapper.deserialize(
                  cmpConf.getSubConfiguration("saber central block"));
      const auto & centralBlockParams =
                     cmpCentralBlockParamsWrapper.saberCentralBlockParameters.value();

      hybridBlockChain_.push_back
          (SaberBlockChainFactory<MODEL>::create
           (parametricIfNotEnsemble(centralBlockParams.saberBlockName.value()),
            *hybridGeom,
            cmpOuterVars,
            fset4dXb,
            fset4dFg,
            fset4dCmpEns,
            cmpCovarConf,
            cmpConf));
    }
    ASSERT(hybridBlockChain_.size() > 0);
  }

  oops::Log::trace() << "SaberHybridBlockChain ctor done" << std::endl;
}

// -----------------------------------------------------------------------------

}  // namespace saber
