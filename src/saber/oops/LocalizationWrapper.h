/*
 * (C) Copyright 2021 UCAR
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#pragma once

#include <Eigen/Dense>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "atlas/field.h"

#include "eckit/config/Configuration.h"

#include "oops/base/FieldSets.h"
#include "oops/base/State4D.h"
#include "oops/base/Variables.h"
#include "oops/util/Duration.h"
#include "oops/util/Logger.h"
#include "oops/util/Timer.h"

#include "saber/blocks/SaberParametricBlockChain.h"
#include "saber/oops/Utilities.h"

namespace saber {

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

 private:
  // Group name
  const std::string name_;
  const oops::Variables vars_;
  const std::string refVar_;
  std::unique_ptr<SaberParametricBlockChain> loc_;
};

// -----------------------------------------------------------------------------

template<typename MODEL>
class LocalizationWrapper  {
  typedef oops::Geometry<MODEL> Geometry_;

 public:
  LocalizationWrapper(const Geometry_ &,
                      const Geometry_ &,
                      const oops::GeometryData &,
                      const oops::Variables &,
                      oops::FieldSet4D &,
                      oops::FieldSet4D &,
                      oops::FieldSets &,
                      oops::FieldSets &,
                      const eckit::LocalConfiguration &,
                      const eckit::Configuration &);
  ~LocalizationWrapper();
  static const std::string classname() {return "saber::LocalizationWrapper";}

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

  // Duplicated and weighted strategy weights
  Eigen::MatrixXd locWgtSqrt_;
};

// -----------------------------------------------------------------------------

template<typename MODEL>
LocalizationWrapper<MODEL>::LocalizationWrapper(const Geometry_ & geom,
                                                const Geometry_ & dualResGeom,
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

    // Initialize localization blockchain
    group.localization().reset(new SaberParametricBlockChain(geom, dualResGeom, outerVars,
      fset4dXb, fset4dFg, fsetEns, fsetDualResEns, covarConf, conf));

    // Add group
    groups_.emplace_back(std::move(group));
  } else {
    // Get multivariate strategy
    strategy_ = conf.getString("multivariate strategy");
    ASSERT(strategy_ == "univariate" ||
           strategy_ == "duplicated" ||
           strategy_ == "crossed" ||
           strategy_ == "duplicated and weighted");

    // Get groups
    const std::vector<eckit::LocalConfiguration> groupConfs = conf.getSubConfigurations("groups");
    for (const auto & groupConf : groupConfs) {
      // Define group name
      const std::string groupName = groupConf.getString("group name");

      // Define group variables
      oops::Variables groupVars;
      const std::vector<std::string> varNames = groupConf.getStringVector("variables");
      size_t levelsCheck = 1;
      std::string refVar;
      for (const auto & varName : varNames) {
        // Get number of levels
        const size_t varLevels = outerVars[varName].getLevels();

        // All the variables of a group should have the same number of levels, or only one level if
        // 3D and 2D variables are mixed
        if (levelsCheck == 1) {
          // Assign number of levels
          levelsCheck = varLevels;
          refVar = varName;
        } else {
          if (varLevels > 1) {
            // Check number of levels
            ASSERT(levelsCheck == varLevels);
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

      // Add group
      groups_.emplace_back(std::move(group));
    }
  }

  // Strategy-specific setup
  if (strategy_ == "duplicated and weighted") {
    // Prepare weights for the "duplicated and weighted" strategy

    // Allocation
    const size_t nv = outerVars.size();
    Eigen::MatrixXd locWgt = Eigen::MatrixXd::Zero(nv, nv);
    locWgtSqrt_.resize(nv, nv);

    // Set default weights
    const double defaultWeight = conf.getDouble("default off-diagonal weight", 0.0);
    for (size_t jg = 0; jg < groups_.size(); ++jg) {
      for (const auto & var1 : groups_[jg].variables()) {
        // Get variable 1 index
        const size_t jv1 = outerVars.find(var1.name());

        for (const auto & var2 : groups_[jg].variables()) {
          // Get variable 2 index
          const size_t jv2 = outerVars.find(var2.name());

          if (jv1 == jv2) {
            // Unit diagonal
            locWgt(jv2, jv1) = 1.0;
          } else {
            // Default off-diagonal weight
            locWgt(jv2, jv1) = defaultWeight;
          }
        }
      }
    }

    // Set specific weights
    if (conf.has("specific off-diagonal weights")) {
      // Get specific weights
      const std::vector<eckit::LocalConfiguration> specWeights =
        conf.getSubConfigurations("specific off-diagonal weights");

      for (const auto & specWeight : specWeights) {
        // Get variables pair and weight
        const std::vector<std::string> varPair = specWeight.getStringVector("variables pair");
        ASSERT(varPair.size() == 2);
        const double weight = specWeight.getDouble("value");

        // Get variables pair indices
        const size_t jv1 = outerVars.find(varPair[0]);
        const size_t jv2 = outerVars.find(varPair[1]);

        // Check that variables are different
        ASSERT(jv1 != jv2);

        // Check that variables are in the same group
        for (const auto & group : groups_) {
          if (group.variables().has(varPair[0]) || group.variables().has(varPair[1])) {
            ASSERT(group.variables().has(varPair[0]) && group.variables().has(varPair[1]));
          }
        }

        // Set weight symmetrically
        locWgt(jv2, jv1) = weight;
        locWgt(jv1, jv2) = weight;
      }

      // Cholesky decomposition
      locWgtSqrt_ = locWgt.llt().matrixL();
    }
  }

  oops::Log::trace() << classname() <<  "::LocalizationWrapper done" << std::endl;
}

// -----------------------------------------------------------------------------

template<typename MODEL>
LocalizationWrapper<MODEL>::~LocalizationWrapper() {
  util::Timer timer(classname(), "~LocalizationWrapper");
  oops::Log::trace() << classname() << "::~LocalizationWrapper destructed" << std::endl;
}

// -----------------------------------------------------------------------------

template<typename MODEL>
void LocalizationWrapper<MODEL>::randomize(oops::FieldSet4D & fset4d) const {
  util::Timer timer(classname(), "randomize");
  oops::Log::trace() << classname() << "::randomize starting" << std::endl;

  // SABER block chain randomization
  if (strategy_ == "deprecated") {
    groups_[0].localization()->randomize(fset4d);
  } else {
    throw eckit::Exception("not implemented yet", Here());
  }

  oops::Log::trace() << classname() << "::randomize done" << std::endl;
}

// -----------------------------------------------------------------------------

template<typename MODEL>
void LocalizationWrapper<MODEL>::multiply(oops::FieldSet4D & fset4d) const {
  util::Timer timer(classname(), "multiply");
  oops::Log::trace() << classname() << "::multiply starting" << std::endl;

  // SABER block chain multiplication
  if (strategy_ == "deprecated") {
    groups_[0].localization()->multiply(fset4d);
  } else {
    if (strategy_ == "univariate") {
      // Univariate: localization of each group is applied to each variable of the group
      for (const auto & group : groups_) {
        for (const auto & var : group.variables()) {
          // Get field
          auto field = fset4d[0][var.name()];

          // Rename the field with the name of the group
          field.rename(group.name());

          // Create a FieldSet4D with a single variable
          atlas::FieldSet fset;
          fset.add(field);
          oops::FieldSet4D fset4dTmp({fset4d[0].validTime(), fset4d[0].commGeom()});
          fset4dTmp[0].shallowCopy(fset);

          // Apply localization
          group.localization()->multiply(fset4dTmp);

          // Rename the field with its initial name
          field.rename(var.name());
        }
      }
    } else if (strategy_ == "duplicated") {
      // Duplicated: the localization of each group is to the sum of all the fields of the group,
      // the result is split into the different fields
      for (const auto & group : groups_) {
        // Get the reference field
        auto refField = fset4d[0][group.referenceVariable()];

        // Rename the field with the name of the group
        refField.rename(group.name());

        // Get reference field view
        auto refView = atlas::array::make_view<double, 2>(refField);

        // Create a FieldSet4D with a single variable, the reference variable
        atlas::FieldSet fset;
        fset.add(refField);
        oops::FieldSet4D fset4dTmp({fset4d[0].validTime(), fset4d[0].commGeom()});
        fset4dTmp[0].shallowCopy(fset);

        // Sum of fields
        for (const auto & var : group.variables()) {
          if (var.name() != group.referenceVariable()) {
            // Get field
            const auto field = fset4d[0][var.name()];

            // Get field view
            const auto view = atlas::array::make_view<double, 2>(field);

            // Check whether the field is a 2D field added to a reference 3D field
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
              for (int jnode = 0; jnode < field.shape(0); ++jnode) {
                refView(jnode, lev2d) += view(jnode, 0);
              }
            } else {
              // All levels
              for (int jnode = 0; jnode < field.shape(0); ++jnode) {
                for (int jlevel = 0; jlevel < field.shape(1); ++jlevel) {
                  refView(jnode, jlevel) += view(jnode, jlevel);
                }
              }
            }
          }
        }

        // Apply localization
        group.localization()->multiply(fset4dTmp);

        // Split field
        for (const auto & var : group.variables()) {
          if (var.name() == group.referenceVariable()) {
            // Rename the field with its initial name
            refField.rename(var.name());
          } else {
            // Get field
            auto field = fset4d[0][var.name()];

            // Get field view
            auto view = atlas::array::make_view<double, 2>(field);

            // Check whether the field is a 2D field added to a reference 3D field
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
              for (int jnode = 0; jnode < field.shape(0); ++jnode) {
                view(jnode, 0) = refView(jnode, lev2d);
              }
            } else {
              // All levels
              for (int jnode = 0; jnode < field.shape(0); ++jnode) {
                for (int jlevel = 0; jlevel < field.shape(1); ++jlevel) {
                  view(jnode, jlevel) = refView(jnode, jlevel);
                }
              }
            }
          }
        }
      }
    } else {
      throw eckit::Exception("not implemented yet", Here());
    }
  }

  oops::Log::trace() << classname() << "::multiply done" << std::endl;
}

// -----------------------------------------------------------------------------

template<typename MODEL>
size_t LocalizationWrapper<MODEL>::ctlVecSize() const {
  oops::Log::trace() << classname() << "::ctlVecSize done" << std::endl;

  oops::Log::trace() << classname() << "::ctlVecSize done" << std::endl;
}

// -----------------------------------------------------------------------------

template<typename MODEL>
void LocalizationWrapper<MODEL>::multiplySqrt(const atlas::Field & cv,
                                              oops::FieldSet4D & fset4d,
                                              const size_t & index) const {
  oops::Log::trace() << classname() << "::multiplySqrt done" << std::endl;

  oops::Log::trace() << classname() << "::multiplySqrt done" << std::endl;
}

// -----------------------------------------------------------------------------

template<typename MODEL>
void LocalizationWrapper<MODEL>::multiplySqrtAD(const oops::FieldSet4D & fset4d,
                                                atlas::Field & cv,
                                                const size_t & index) const {
  oops::Log::trace() << classname() << "::multiplySqrtAD done" << std::endl;

  oops::Log::trace() << classname() << "::multiplySqrtAD done" << std::endl;
}

// -----------------------------------------------------------------------------

}  // namespace saber
