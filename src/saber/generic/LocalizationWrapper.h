/*
 * (C) Copyright 2025 Meteorologisk Institutt
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#pragma once

#include <Eigen/Dense>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "atlas/field.h"

#include "eckit/config/Configuration.h"

#include "oops/base/FieldSets.h"
#include "oops/base/Geometry.h"
#include "oops/base/State4D.h"
#include "oops/base/Variables.h"
#include "oops/util/Duration.h"
#include "oops/util/Logger.h"
#include "oops/util/Timer.h"

#include "saber/blocks/SaberParametricBlockChain.h"
#include "saber/oops/Utilities.h"

namespace saber {
namespace generic {

// -----------------------------------------------------------------------------

class LocalizationGroup {
 public:
  LocalizationGroup(const std::string & name,
                    const oops::Variables & vars,
                    const std::string & refVar)
    : name_(name), vars_(vars), refVar_(refVar), loc_() {}

  // Accessors
  const std::string & name() const
    {return name_;}
  const oops::Variables & variables() const
    {return vars_;}
  const std::string & referenceVariable() const
    {return refVar_;}
  const std::unique_ptr<SaberParametricBlockChain> & localization() const
    {return loc_;}
  std::unique_ptr<SaberParametricBlockChain> & localization()
    {return loc_;}
  const Eigen::MatrixXd & locWgtSqrt() const
    {return locWgtSqrt_;}
  Eigen::MatrixXd & locWgtSqrt()
    {return locWgtSqrt_;}

  // 2D levels
  const size_t get2dLevel(const std::string & var) const
    {return lev2d_.at(var);}
  void set2dLevel(const std::string & var,
                  const size_t & lev2d)
    {lev2d_.insert({var, lev2d});}

 private:
  // Group name
  const std::string name_;

  // Group variables
  const oops::Variables vars_;

  // Group reference variable (for fields summation)
  const std::string refVar_;

  // Level for 2D fields (for 3D and 2D fields summation)
  std::unordered_map<std::string, size_t> lev2d_;

  // Localization as a parametric block chain
  std::unique_ptr<SaberParametricBlockChain> loc_;

  // Localization weights for the "duplicated and weighted" strategy
  Eigen::MatrixXd locWgtSqrt_;
};

// -----------------------------------------------------------------------------

class LocalizationWrapper  {
 public:
  template<typename MODEL>
  LocalizationWrapper(const oops::Geometry<MODEL> &,
                      const oops::Geometry<MODEL> &,
                      const oops::GeometryData &,
                      const oops::Variables &,
                      oops::FieldSet4D &,
                      oops::FieldSet4D &,
                      oops::FieldSets &,
                      oops::FieldSets &,
                      const eckit::LocalConfiguration &,
                      const eckit::Configuration &);
  ~LocalizationWrapper();
  static const std::string classname() {return "saber::generic::LocalizationWrapper";}

  void randomize(oops::FieldSet4D &) const;
  void multiply(oops::FieldSet4D &) const;
  size_t ctlVecSize() const;
  void multiplySqrt(const atlas::Field &,
                    oops::FieldSet4D &,
                    const size_t &) const;
  void multiplySqrtAD(const oops::FieldSet4D &,
                      atlas::Field &,
                      const size_t &) const;

 private:
  // Save levels direction
  const bool levelsAreTopDown_;

  // Multivariate strategy
  std::string strategy_;

  // Groups of variables, with their own localization
  std::vector<LocalizationGroup> groups_;
};

// -----------------------------------------------------------------------------

template<typename MODEL>
LocalizationWrapper::LocalizationWrapper(const oops::Geometry<MODEL> & geom,
                                         const oops::Geometry<MODEL> & dualResGeom,
                                         const oops::GeometryData & outerGeomData,
                                         const oops::Variables & outerVars,
                                         oops::FieldSet4D & fset4dXb,
                                         oops::FieldSet4D & fset4dFg,
                                         oops::FieldSets & fsetEns,
                                         oops::FieldSets & fsetDualResEns,
                                         const eckit::LocalConfiguration & covarConf,
                                         const eckit::Configuration & conf)
  : levelsAreTopDown_(geom.levelsAreTopDown()) {
  oops::Log::trace() << classname() <<  "::LocalizationWrapper starting" << std::endl;
  util::Timer timer(classname(), "LocalizationWrapper");

  if (!conf.has("groups")) {
    // For backward compatibility, only one group is created in this case
    strategy_ = "deprecated";

    // Create group
    LocalizationGroup group("deprecated", outerVars, "");

    // The localization is a parametric block chain constructed with the same geometry
    // as the ensemble block chain by default. If the outer blocks or transform block
    // chain include a change of geometries, we need to use build the localization from
    // the current geometryData, using a generic constructor of the parametric block chain.

    // Check consistency of `geom` and the current geometry
    const auto & currentFspace = outerGeomData.functionSpace();
    if (util::getGridUid(geom.functionSpace()) != util::getGridUid(currentFspace)) {
      oops::Log::info() << "Info     : Localization and ensemble are on different "
                           "functionSpaces, building localization with generic "
                           "constructor" << std::endl;
      // Note QUENCH could just build another geometry here and use the standard
      // constructor, but other models usually don't have this ability to create a
      // Geometry on any mesh.
      group.localization().reset(new SaberParametricBlockChain(outerGeomData,
                                                               outerVars,
                                                               fset4dXb,
                                                               fset4dFg,
                                                               covarConf,
                                                               conf));
    } else {
      oops::Log::info() << "Info     : Localization and ensemble are on same "
                           "functionSpaces, building localization with standard "
                           "constructor" << std::endl;
      group.localization().reset(new SaberParametricBlockChain(geom,
                                                               dualResGeom,
                                                               outerVars,
                                                               fset4dXb,
                                                               fset4dFg,
                                                               fsetEns,
                                                               fsetDualResEns,
                                                               covarConf,
                                                               conf));
    }

    // Add group
    groups_.emplace_back(std::move(group));
  } else {
    // Get multivariate strategy:
    // - Univariate: localization of each group is applied to each variable of the group.
    // - Duplicated: the localization of each group is to the sum of all the fields of the group,
    //   the result is split into the different fields.
    // - Duplicated and weighted: the localization of each group is applied to each variable,
    //   but variables of the same group are combined with user-specified weights.
    // - Crossed: the localization of each group is to the sum of all the fields of the group,
    //   the result is split into the different fields. All the groups share the same control
    //   vector: square-root formulation is necessary.
    strategy_ = conf.getString("multivariate strategy");
    ASSERT(strategy_ == "univariate" ||
           strategy_ == "duplicated" ||
           strategy_ == "duplicated and weighted" ||
           strategy_ == "crossed");

    // Get groups
    const std::vector<eckit::LocalConfiguration> groupConfs = conf.getSubConfigurations("groups");
    for (const auto & groupConf : groupConfs) {
      // Define group name
      const std::string groupName = groupConf.getString("group name");

      // Define group variables
      oops::Variables groupVars;
      const std::vector<std::string> varNames = groupConf.getStringVector("variables");
      size_t levelsCheck = outerVars[varNames[0]].getLevels();
      std::string refVar = varNames[0];
      for (const auto & varName : varNames) {
        // Get number of levels
        const size_t varLevels = outerVars[varName].getLevels();

        if ((strategy_ == "univariate") || (strategy_ == "duplicated and weighted")) {
          // All the fields of the group should have the same number of levels.
          ASSERT(levelsCheck == varLevels);
        } else {
          // All the fields of the group should have the same number of levels,
          // or only one level if 3D and 2D fields are mixed (in this case, the reference
          // field should be 3D).
          if (varLevels > 1) {
            if (levelsCheck == 1) {
              levelsCheck = varLevels;
              refVar = varName;
            } else {
              ASSERT(levelsCheck == varLevels);
            }
          }
        }

        // Add variable
        groupVars.push_back(outerVars[varName]);
      }

      // Create group
      LocalizationGroup group(groupName, groupVars, refVar);

      // Define active variables
      oops::Variables activeVars;
      activeVars.push_back(oops::Variable(groupName, groupVars[refVar].metaData(),
        groupVars[refVar].getLevels()));

      // The localization is a parametric block chain constructed with the same geometry
      // as the ensemble block chain by default. If the outer blocks or transform block
      // chain include a change of geometries, we need to use build the localization from
      // the current geometryData, using a generic constructor of the parametric block chain.

      // Check consistency of `geom` and the current geometry
      const auto & currentFspace = outerGeomData.functionSpace();
      if (util::getGridUid(geom.functionSpace()) != util::getGridUid(currentFspace)) {
        oops::Log::info() << "Info     : Localization and ensemble are on different "
                             "functionSpaces, building localization with generic "
                             "constructor" << std::endl;
        // Note QUENCH could just build another geometry here and use the standard
        // constructor, but other models usually don't have this ability to create a
        // Geometry on any mesh.
        group.localization().reset(new SaberParametricBlockChain(outerGeomData,
                                                                 activeVars,
                                                                 fset4dXb,
                                                                 fset4dFg,
                                                                 covarConf,
                                                                 conf));
      } else {
        oops::Log::info() << "Info     : Localization and ensemble are on same "
                             "functionSpaces, building localization with standard "
                             "constructor" << std::endl;
        group.localization().reset(new SaberParametricBlockChain(geom,
                                                                 dualResGeom,
                                                                 activeVars,
                                                                 fset4dXb,
                                                                 fset4dFg,
                                                                 fsetEns,
                                                                 fsetDualResEns,
                                                                 covarConf,
                                                                 groupConf));
      }

      // Get the nearest 3D level for 2D variables

      // Get reference field
      const auto refField = fset4dXb[0][refVar];

      for (auto & var : groupVars) {
        // Get field
        const auto field = fset4dXb[0][var.name()];

        if ((refField.levels() > 1) && (field.levels() == 1)) {
          // 2D level only
          const std::string nearest3dLevel = field.metadata().getString("nearest 3d level");
          ASSERT((nearest3dLevel == "top") || (nearest3dLevel == "bottom"));
          size_t lev2d;
          if (levelsAreTopDown_) {
            lev2d = (nearest3dLevel == "top") ? 0 : refField.levels()-1;
          } else {
             lev2d = (nearest3dLevel == "bottom") ? 0 : refField.levels()-1;
          }
          group.set2dLevel(var.name(), lev2d);
        }
      }


      // Strategy-specific setup
      if (strategy_ == "duplicated and weighted") {
        // Prepare weights for the "duplicated and weighted" strategy

        // Allocation
        const size_t nv = groupVars.size();
        Eigen::MatrixXd locWgt = Eigen::MatrixXd::Zero(nv, nv);
        group.locWgtSqrt().resize(nv, nv);

        // Set default weights
        const double defaultWeight = groupConf.getDouble("default off-diagonal weight", 0.0);
        for (size_t jvarJ = 0; jvarJ < groupVars.size(); ++jvarJ) {
          for (size_t jvarI = 0; jvarI < groupVars.size(); ++jvarI) {
            if (jvarJ == jvarI) {
              // Unit diagonal
              locWgt(jvarJ, jvarI) = 1.0;
            } else {
              // Default off-diagonal weight
              locWgt(jvarJ, jvarI) = defaultWeight;
            }
          }
        }

        // Set specific weights
        if (groupConf.has("specific off-diagonal weights")) {
          // Get specific weights
          const std::vector<eckit::LocalConfiguration> specWeights =
            groupConf.getSubConfigurations("specific off-diagonal weights");

          for (const auto & specWeight : specWeights) {
            // Get variables pair and weight
            const std::vector<std::string> varPair = specWeight.getStringVector("variables pair");
            ASSERT(varPair.size() == 2);
            const double weight = specWeight.getDouble("value");

            // Get variables pair indices
            const size_t jvarJ = groupVars.find(varPair[0]);
            const size_t jvarI = groupVars.find(varPair[1]);

            // Check that variables are different
            ASSERT(jvarJ != jvarI);

            // Set weight symmetrically
            locWgt(jvarI, jvarJ) = weight;
            locWgt(jvarJ, jvarI) = weight;
          }
        }

        // Cholesky decomposition
        group.locWgtSqrt() = locWgt.llt().matrixL();
      }

      // Add group
      groups_.emplace_back(std::move(group));
    }

    // Strategy-specific check
    if (strategy_ == "crossed") {
      // Check that all groups have the same control vector size
      const size_t ctlVecSize = groups_[0].localization()->ctlVecSize();
      for (const auto & group : groups_) {
        ASSERT(group.localization()->ctlVecSize() == ctlVecSize);
      }
    }
  }

  oops::Log::trace() << classname() <<  "::LocalizationWrapper done" << std::endl;
}

// -----------------------------------------------------------------------------

}  // namespace generic
}  // namespace saber
