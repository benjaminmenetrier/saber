/*
 * (C) Copyright 2023- UCAR
 * (C) Crown Copyright 2024 Met Office
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "atlas/field.h"

#include "eckit/exception/Exceptions.h"

#include "oops/base/FieldSet4D.h"
#include "oops/base/FieldSets.h"
#include "oops/interface/ModelData.h"
#include "oops/util/ConfigHelpers.h"

#include "saber/blocks/SaberBlockChainBase.h"
#include "saber/blocks/SaberBlockParametersBase.h"
#include "saber/blocks/SaberCentralBlockBase.h"
#include "saber/blocks/SaberOuterBlockChain.h"
#include "saber/blocks/SaberParametricBlockChainGroup.h"

namespace saber {

// -----------------------------------------------------------------------------

/// Chain of outer (optional) and not-ensemble central block. Can be used
/// as static error covariance component and as localization for ensemble
/// error covariance.
class SaberParametricBlockChain : public SaberBlockChainBase {
 public:
  /// @brief Standard constructor using MODEL geometry
  template<typename MODEL>
  SaberParametricBlockChain(const oops::Geometry<MODEL> & geom,
                            const oops::Variables & outerVars,
                            oops::FieldSet4D & fset4dXb,
                            oops::FieldSet4D & fset4dFg,
                            oops::FieldSets & fsetEns,
                            const eckit::LocalConfiguration & covarConf,
                            const eckit::Configuration & conf);
  /// @brief Simpler, limited constructor using only generic GeometryData
  SaberParametricBlockChain(const oops::GeometryData & outerGeometryData,
                            const bool & levelsAreTopDown,
                            const oops::Variables & outerVars,
                            oops::FieldSet4D & fset4dXb,
                            oops::FieldSet4D & fset4dFg,
                            const eckit::LocalConfiguration & covarConf,
                            const eckit::Configuration & conf);
  ~SaberParametricBlockChain() = default;

  /// @brief Filter the increment
  void filter(oops::FieldSet4D &) const;

  /// @brief Randomize the increment according to this B matrix.
  void randomize(oops::FieldSet4D &) const;
  /// @brief Multiply the increment by this B matrix.
  void multiply(oops::FieldSet4D &) const;
  /// @brief Get this B matrix square-root control vector size.
  size_t ctlVecSize() const;
  /// @brief Multiply the control vector by this B matrix square-root.
  void multiplySqrt(const atlas::Field &, oops::FieldSet4D &, const size_t &) const;
  /// @brief Multiply the increment by this B matrix square-root adjoint.
  void multiplySqrtAD(const oops::FieldSet4D &, atlas::Field &, const size_t &) const;

  /// @brief Accessor to outer function space
  const atlas::FunctionSpace & outerFunctionSpace() const {return outerFunctionSpace_;}
  /// @brief Accessor to outer variables
  const oops::Variables & outerVariables() const {return outerVariables_;}

 private:
  /// @brief Initialize groups.
  ///        Used in constructors
  void initGroups(const bool & levelsAreTopDown,
                  const oops::Variables & outerVars,
                  const oops::FieldSet4D & fset4dXb,
                  const eckit::Configuration & covarConf,
                  const eckit::Configuration & conf);

  /// @brief Outer function space
  const atlas::FunctionSpace outerFunctionSpace_;
  /// @brief Outer variables
  const oops::Variables outerVariables_;
  // Multivariate strategy
  std::string strategy_;
  // Groups of variables, with their own chains
  std::vector<SaberParametricBlockChainGroup> groups_;
};

// -----------------------------------------------------------------------------

template<typename MODEL>
SaberParametricBlockChain::SaberParametricBlockChain(const oops::Geometry<MODEL> & geom,
                       const oops::Variables & outerVars,
                       oops::FieldSet4D & fset4dXb,
                       oops::FieldSet4D & fset4dFg,
                       // TODO(AS): read inside the block so there is no need to pass
                       // as non-const
                       oops::FieldSets & fsetEns,
                       const eckit::LocalConfiguration & covarConf,
                       const eckit::Configuration & conf)
  : outerFunctionSpace_(geom.functionSpace()), outerVariables_(outerVars) {
  oops::Log::trace() << "SaberParametricBlockChain ctor starting" << std::endl;

  // Initialize groups
  initGroups(geom.levelsAreTopDown(), outerVars, fset4dXb, covarConf, conf);

  // Loop over groups
  for (auto & group : groups_) {
    // Get central block parameters
    SaberCentralBlockParametersWrapper saberCentralBlockParamsWrapper;
    saberCentralBlockParamsWrapper.deserialize(
      group.conf().getSubConfiguration("saber central block"));

    const SaberBlockParametersBase & saberCentralBlockParams =
      saberCentralBlockParamsWrapper.saberCentralBlockParameters;

    const bool centralDirectCalibration = saberCentralBlockParams.doCalibration();

    // If needed create outer block chain
    if (group.conf().has("saber outer blocks")) {
      std::vector<SaberOuterBlockParametersWrapper> cmpOuterBlocksParams;
      for (const auto & outerBlockConf : group.conf().getSubConfigurations("saber outer blocks")) {
        SaberOuterBlockParametersWrapper cmpOuterBlockParamsWrapper;
        cmpOuterBlockParamsWrapper.deserialize(outerBlockConf);
        cmpOuterBlocksParams.push_back(cmpOuterBlockParamsWrapper);
      }
      group.outerBlockChain() = std::make_unique<SaberOuterBlockChain>(geom, group.chainVars(),
                                   fset4dXb, fset4dFg, fsetEns, covarConf,
                                   cmpOuterBlocksParams, centralDirectCalibration);
    }

    // Set outer geometry data for central block
    const oops::GeometryData & currentOuterGeom = group.outerBlockChain() ?
                               group.outerBlockChain()->innerGeometryData() : geom.generic();


    // Create central block
    oops::Log::info() << "Info     : Creating central block: "
                      << saberCentralBlockParams.saberBlockName.value() << std::endl;

    const auto[currentOuterVars, activeVars]
                = group.initCentralBlock(currentOuterGeom,
                                         covarConf,
                                         saberCentralBlockParams,
                                         fset4dXb,
                                         fset4dFg);

    // Read and add model fields
    group.centralBlock()->read(geom, currentOuterVars);

    // Iterative ensemble loading flag
    const bool iterativeEnsembleLoading = covarConf.getBool("iterative ensemble loading");

    // Ensemble configuration
    eckit::LocalConfiguration ensembleConf
           = covarConf.getSubConfiguration("ensemble configuration");
    if (saberCentralBlockParams.doCalibration()) {
      // Block calibration
      if (iterativeEnsembleLoading) {
        // Iterative calibration
        oops::Log::info() << "Info     : Iterative calibration" << std::endl;

        // Initialization
        group.centralBlock()->iterativeCalibrationInit();

        // Get ensemble size
        size_t nens = ensembleConf.getInt("ensemble size");

        for (size_t ie = 0; ie < nens; ++ie) {
          // Read ensemble member
          oops::FieldSet3D fset(fset4dXb[0].validTime(), geom.getComm());
          readEnsembleMember(geom, outerVariables_, ensembleConf, ie, fset);

          // Apply outer blocks inverse (all of them)
          oops::Log::info() << "Info     : Apply outer blocks inverse (all of them)" << std::endl;
          if (group.outerBlockChain()) group.outerBlockChain()->leftInverseMultiply(fset);

          // Use FieldSet in the central block
          oops::Log::info() << "Info     : Use FieldSet in the central block" << std::endl;
          group.centralBlock()->iterativeCalibrationUpdate(fset);
        }

        // Finalization
        oops::Log::info() << "Info     : Finalization" << std::endl;
        group.centralBlock()->iterativeCalibrationFinal();
      } else {
        // Direct calibration
        oops::Log::info() << "Info     : Direct calibration" << std::endl;
        group.centralBlock()->directCalibration(fsetEns);
      }
    } else if (saberCentralBlockParams.doRead()) {
      // Read data
      oops::Log::info() << "Info     : Read data" << std::endl;
      group.centralBlock()->read();
    }

    // Write calibration data
    if (saberCentralBlockParams.forceWrite.value() || saberCentralBlockParams.doCalibration()) {
      oops::Log::info() << "Info     : Write data" << std::endl;
      group.centralBlock()->write(geom);
      group.centralBlock()->write();
    }

    // Write final ensemble
    if (covarConf.has("output ensemble")) {
      // Get output parameters configuration
      const eckit::LocalConfiguration outputEnsembleConf(covarConf, "output ensemble");

      // Check whether geometry grid is similar to the last outer block inner geometry
      const bool useModelWriter = (util::getGridUid(geom.functionSpace())
        == util::getGridUid(currentOuterGeom.functionSpace()));

      // Get ensemble size
      size_t ensembleSize = ensembleConf.getInt("ensemble size");

      // Estimate mean
      oops::FieldSet3D fsetMean(fset4dXb[0].validTime(), geom.getComm());
      if (iterativeEnsembleLoading) {
        for (size_t ie = 0; ie < ensembleSize; ++ie) {
          // Read member
          oops::FieldSet3D fsetMem(fset4dXb[0].validTime(), geom.getComm());
          readEnsembleMember(geom, activeVars, ensembleConf, ie, fsetMem);

          // Update mean
          if (ie == 0) {
            fsetMean.deepCopy(fsetMem);
          } else {
            fsetMean += fsetMem;
          }
        }

        // Normalize mean
        fsetMean *= 1.0/static_cast<double>(ensembleSize);
      }

      // Write first member only
      const bool firstMemberOnly = outputEnsembleConf.getBool("first member only", false);
      if (firstMemberOnly) {
        ensembleSize = 1;
      }

      for (size_t ie = 0; ie < ensembleSize; ++ie) {
        oops::Log::info() << "Info     : Write member " << ie << std::endl;

        // Increment pointer
        oops::Increment<MODEL> dx(geom, activeVars, fset4dXb[0].validTime());

        // Get ensemble member
        if (iterativeEnsembleLoading) {
          // Read ensemble member
          oops::FieldSet3D fset(fset4dXb[0].validTime(), geom.getComm());
          readEnsembleMember(geom, activeVars, ensembleConf, ie, fset);

          // Remove mean
          fset -= fsetMean;

          // Apply outer blocks inverse
          if (group.outerBlockChain()) group.outerBlockChain()->leftInverseMultiply(fset);

          // ATLAS fieldset to Increment_
          dx.fromFieldSet(fset.fieldSet());
        } else {
          // ATLAS fieldset to Increment_
          dx.fromFieldSet(fsetEns[ie].fieldSet());
        }

        if (useModelWriter) {
          // Use model writer

          // Set member index
          eckit::LocalConfiguration outputMemberConf(outputEnsembleConf);
          util::setMember(outputMemberConf, ie+1);

          // Write Increment
          dx.write(outputMemberConf);
          oops::Log::test() << "Norm of ensemble member " << ie << ": " << dx.norm() << std::endl;
        } else {
          // Use generic ATLAS writer
          throw eckit::NotImplemented("generic output ensemble write not implemented yet", Here());
        }
      }
    }

    // Test central block
    group.testCentralBlock(covarConf, saberCentralBlockParams, currentOuterGeom, activeVars);
  }

  // Strategy-specific check
  if (strategy_ == "crossed") {
    // Check that all groups have the same control vector size
    const size_t ctlVecSize = groups_[0].centralBlock()->ctlVecSize();
    for (const auto & group : groups_) {
      ASSERT(group.centralBlock()->ctlVecSize() == ctlVecSize);
    }
  }

  oops::Log::trace() << "SaberParametricBlockChain ctor done" << std::endl;
}

// -----------------------------------------------------------------------------

}  // namespace saber
