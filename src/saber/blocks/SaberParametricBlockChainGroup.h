/*
 * (C) Copyright 2023- UCAR
 * (C) Crown Copyright 2024 Met Office
 * (C) Copyright 2025 Meteorologisk Institutt
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "atlas/field.h"

#include "eckit/exception/Exceptions.h"

#include "oops/base/FieldSet4D.h"
#include "oops/base/FieldSets.h"
#include "oops/base/Variables.h"

#include "saber/blocks/SaberBlockParametersBase.h"
#include "saber/blocks/SaberCentralBlockBase.h"
#include "saber/blocks/SaberOuterBlockChain.h"
#include "saber/oops/Utilities.h"

namespace saber {

// -----------------------------------------------------------------------------

class SaberParametricBlockChainGroup {
 public:
  SaberParametricBlockChainGroup(const oops::Variables & vars,
                                 const std::string & refVarName,
                                 const oops::FieldSet4D & fset4dXb,
                                 const eckit::Configuration & covarConf,
                                 const eckit::Configuration & groupConf)
    : conf_(groupConf),
      name_(groupConf.getString("group name")),
      vars_(vars),
      refVarName_(refVarName),
      outerBlockChain_(),
      centralBlock_(),
      crossTimeCov_(covarConf.getString("time covariance") == "multivariate duplicated"),
      timeComm_(fset4dXb.commTime()),
      size4D_(fset4dXb.size())
    {}

  /// @brief Initialize central block, central function space and central variables.
  ///        Used in constructors.
  std::tuple<oops::Variables, oops::Variables>
      initCentralBlock(const oops::GeometryData & outerGeom,
                       const eckit::LocalConfiguration & covarConf,
                       const SaberBlockParametersBase & saberCentralBlockParams,
                       const oops::FieldSet4D & fset4dXb,
                       const oops::FieldSet4D & fset4dFg);

  /// @brief Run adjoint and square-root tests on central block. Used in constructors.
  void testCentralBlock(const eckit::LocalConfiguration & covarConf,
                        const SaberBlockParametersBase & saberCentralBlockParams,
                        const oops::GeometryData & outerGeom,
                        const oops::Variables & activeVars) const;

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

  // Accessors
  const eckit::LocalConfiguration & conf() const
    {return conf_;}
  const std::string & name() const
    {return name_;}
  const oops::Variables & variables() const
    {return vars_;}
  const std::string & refVarName() const
    {return refVarName_;}
  const oops::Variables & chainVars() const
    {return chainVars_;}
  oops::Variables & chainVars()
    {return chainVars_;}
  const std::unique_ptr<SaberOuterBlockChain> & outerBlockChain() const
    {return outerBlockChain_;}
  std::unique_ptr<SaberOuterBlockChain> & outerBlockChain()
    {return outerBlockChain_;}
  const std::unique_ptr<SaberCentralBlockBase> & centralBlock() const
    {return centralBlock_;}
  std::unique_ptr<SaberCentralBlockBase> & centralBlock()
    {return centralBlock_;}
  const Eigen::MatrixXd & wgtSqrt() const
    {return wgtSqrt_;}
  Eigen::MatrixXd & wgtSqrt()
    {return wgtSqrt_;}

  // 2D levels
  const size_t get2dLevel(const std::string & var) const
    {return lev2d_.at(var);}
  void set2dLevel(const std::string & var,
                  const size_t & lev2d)
    {lev2d_.insert({var, lev2d});}

 private:
  // Group configuration
  const eckit::LocalConfiguration conf_;

  // Group name
  const std::string name_;

  // Group variables
  const oops::Variables vars_;

  // Group reference variable name (for fields summation)
  const std::string refVarName_;

  // Group reference variables (containing only the reference variable, with the group name)
  oops::Variables chainVars_;

  // Level for 2D fields (for 3D and 2D fields summation)
  std::unordered_map<std::string, size_t> lev2d_;

  // Outer block chain
  std::unique_ptr<SaberOuterBlockChain> outerBlockChain_;

  // Central block chain
  std::unique_ptr<SaberCentralBlockBase> centralBlock_;

  // Weights for the "duplicated and weighted" strategy
  Eigen::MatrixXd wgtSqrt_;

  /// Central function space
  atlas::FunctionSpace centralFunctionSpace_;

  /// Central variables
  oops::Variables centralVars_;

  /// @brief Cross-time covariance flag
  const bool crossTimeCov_;

  /// @brief Time communicator
  const eckit::mpi::Comm & timeComm_;

  /// @brief Number of sub-windows
  size_t size4D_;
};

// -----------------------------------------------------------------------------

}  // namespace saber
