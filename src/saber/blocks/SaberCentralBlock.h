/*
 * (C) Copyright 2025- UCAR
 * (C) Crown Copyright 2024 Met Office
 * (C) Copyright 2025 Meteorologisk Institutt
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#pragma once

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "atlas/field.h"

#include "eckit/memory/NonCopyable.h"

#include "oops/base/FieldSet3D.h"
#include "oops/base/GeometryData.h"
#include "oops/util/AssociativeContainers.h"
#include "oops/util/FieldSetHelpers.h"
#include "oops/util/Logger.h"
#include "oops/util/parameters/OptionalParameter.h"
#include "oops/util/parameters/OptionalPolymorphicParameter.h"
#include "oops/util/parameters/Parameters.h"
#include "oops/util/parameters/RequiredPolymorphicParameter.h"
#include "oops/util/Printable.h"

#include "saber/blocks/SaberBlockParametersBase.h"
#include "saber/blocks/SaberCentralBlockBase.h"

namespace saber {

// -----------------------------------------------------------------------------
class OffDiagWeightParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(OffDiagWeightParameters, Parameters)

 public:
  oops::RequiredParameter<std::vector<std::string>> varPair{"variables pair", this};
  oops::RequiredParameter<double> weight{"value", this};
};

// -----------------------------------------------------------------------------
class SaberCentralBlockGroupParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(SaberCentralBlockGroupParameters, Parameters)

 public:
  oops::RequiredParameter<std::string> groupName{"group name", this};
  oops::RequiredParameter<oops::Variables> variables{"variables", this};
  oops::RequiredPolymorphicParameter<SaberBlockParametersBase, SaberCentralBlockFactory>
    block{"saber block name", this};
  // Optional parameters specific to "duplicated and weighted" strategy
  oops::Parameter<double> defOffDiagWeight{"default off-diagonal weight", 0.0, this};
  oops::OptionalParameter<std::vector<OffDiagWeightParameters>>
    offDiagWeights{"specific off-diagonal weights", this};
};

// -----------------------------------------------------------------------------
class SaberCentralBlockParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(SaberCentralBlockParameters, Parameters)

 public:
  oops::Parameter<std::string> strategy{"multivariate strategy", "deprecated", this};
  // Single block:
  oops::OptionalPolymorphicParameter<SaberBlockParametersBase, SaberCentralBlockFactory>
    singleBlock{"saber block name", this};
  // Or multiple blocks:
  oops::OptionalParameter<std::vector<SaberCentralBlockGroupParameters>>
    groups{"groups", this};

  // Type of setup
  bool doCalibration() const;
  bool doRead() const;
};

// -----------------------------------------------------------------------------

class SaberCentralBlock : public util::Printable {
 public:
  SaberCentralBlock(const oops::GeometryData & geometryData,
                    const bool levelsAreTopDown,
                    const oops::Variables & outerVars,
                    const eckit::Configuration & covarConf,
                    const SaberCentralBlockParameters & params,
                    const oops::FieldSet3D & xb,
                    const oops::FieldSet3D & fg);
  ~SaberCentralBlock() = default;

  // Application methods

  // Block randomization
  void randomize(oops::FieldSet3D &) const;

  // Block multiplication
  void multiply(oops::FieldSet3D &) const;

  // Setup / calibration methods

  // Read block data
  void read() {
    for (size_t i = 0; i < groups_.size(); ++i) {
      if (doRead_[i]) {
        groups_[i]->read();
      }
    }
  }

  // Direct calibration
  void directCalibration(const oops::FieldSets & fsets) {
    for (size_t i = 0; i < groups_.size(); ++i) {
      if (doCalibration_[i]) {
        groups_[i]->directCalibration(fsets);
      }
    }
  }

  // Iterative calibration
  void iterativeCalibrationInit() {
    for (size_t i = 0; i < groups_.size(); ++i) {
      if (doCalibration_[i]) {
        groups_[i]->iterativeCalibrationInit();
      }
    }
  }
  void iterativeCalibrationUpdate(const oops::FieldSet3D & fset) {
    for (size_t i = 0; i < groups_.size(); ++i) {
      if (doCalibration_[i]) {
        groups_[i]->iterativeCalibrationUpdate(fset);
      }
    }
  }
  void iterativeCalibrationFinal() {
    for (size_t i = 0; i < groups_.size(); ++i) {
      if (doCalibration_[i]) {
        groups_[i]->iterativeCalibrationFinal();
      }
    }
  }

  // Write block data
  void write() const {
    for (size_t i = 0; i < groups_.size(); ++i) {
      groups_[i]->write();
    }
  }

  // Square-root formulation
  size_t ctlVecSize() const;
  void randomCtlVec(atlas::Field &, const size_t &) const;
  void multiplySqrt(const atlas::Field &, oops::FieldSet3D &, const size_t &) const;
  void multiplySqrtAD(const oops::FieldSet3D &, atlas::Field &, const size_t &) const;

  // Return date/time
  const util::DateTime validTime() const {return validTime_;}

  // Read model fields
  template <typename MODEL>
  void read(const oops::Geometry<MODEL> &,
            const oops::Variables &);

  // Write model fields
  template <typename MODEL>
  void write(const oops::Geometry<MODEL> &) const;

  // Adjoint test
  void adjointTest(const oops::GeometryData & geomdata,
                   const double & tol) const;
  // Square-root test
  void sqrtTest(const oops::GeometryData & geomdata,
                const double & tol) const;

  bool doCalibration() const {
    return std::any_of(doCalibration_.begin(), doCalibration_.end(), [](bool v)
      { return v; });
  }
  bool doRead() const {
    return std::any_of(doRead_.begin(), doRead_.end(), [](bool v) { return v; });
  }
  bool forceWrite() const {
    return std::any_of(forceWrite_.begin(), forceWrite_.end(), [](bool v) { return v; });}

 private:
  const oops::GeometryData & geometryData_;
  const util::DateTime validTime_;
  const SaberCentralBlockParameters params_;
  // Multivariate strategy
  std::string strategy_;
  // Groups of central blocks for different variable subsets
  std::vector<std::unique_ptr<SaberCentralBlockBase>> groups_;
  // Group variables
  std::vector<oops::Variables> groupInputVars_;
  // Group reference variable name (for fields summation)
  std::vector<oops::Variable> groupRefVars_;
  // Group names
  std::vector<std::string> groupNames_;
  // Group inner variables (containing only the reference variable, with the group name)
  std::vector<oops::Variables> groupInnerVars_;
  // Weights for the "duplicated and weighted" strategy
  std::vector<Eigen::MatrixXd> wgtSqrt_;
  // Level for 2D fields (for 3D and 2D fields summation)
  std::unordered_map<std::string, size_t> lev2d_;

  // Groups need to be calibrated
  std::vector<bool> doCalibration_;
  // Groups need to read MODEL data
  std::vector<bool> doRead_;
  // Groups need to write MODEL data
  std::vector<bool> forceWrite_;

  void print(std::ostream &) const {}
};


// -----------------------------------------------------------------------------

template <typename MODEL>
void SaberCentralBlock::read(const oops::Geometry<MODEL> & geom,
                             const oops::Variables & vars) {
  oops::Log::trace() << "SaberCentralBlock::read starting" << std::endl;
  for (size_t i = 0; i < groups_.size(); ++i) {
    groups_[i]->read(geom, vars);
  }
  oops::Log::trace() << "SaberCentralBlock::read done" << std::endl;
}

// -----------------------------------------------------------------------------

template <typename MODEL>
void SaberCentralBlock::write(const oops::Geometry<MODEL> & geom) const {
  oops::Log::trace() << "SaberCentralBlock::write starting" << std::endl;
  for (size_t i = 0; i < groups_.size(); ++i) {
    groups_[i]->write(geom);
  }
  oops::Log::trace() << "SaberCentralBlock::write done" << std::endl;
}

// -----------------------------------------------------------------------------

}  // namespace saber
