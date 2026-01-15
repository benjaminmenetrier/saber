/*
 * (C) Copyright 2023- UCAR
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include "saber/blocks/SaberParametricBlockChain.h"

#include <utility>

using atlas::array::make_datatype;
using atlas::array::make_shape;
using atlas::array::make_view;

namespace saber {


// -----------------------------------------------------------------------------

// Generic constructor for the SaberParametricBlockChain, not templated on
// MODEL. This constructor is only used for localization matrices, when the
// outer geometry cannot be a MODEL geometry.
SaberParametricBlockChain::SaberParametricBlockChain(
                          const oops::GeometryData & outerGeometryData,
                          const bool & levelsAreTopDown,
                          const oops::Variables & outerVars,
                          oops::FieldSet4D & fset4dXb,
                          oops::FieldSet4D & fset4dFg,
                          const eckit::LocalConfiguration & covarConf,
                          const eckit::Configuration & conf)
  : outerFunctionSpace_(outerGeometryData.functionSpace()),
    outerVariables_(outerVars) {
  oops::Log::trace() << "SaberParametricBlockChain generic ctor starting" << std::endl;

  // Initialize groups
  initGroups(levelsAreTopDown, outerVars, fset4dXb, covarConf, conf);

  // Loop over groups
  for (auto & group : groups_) {
    // If needed create generic outer block chain
    if (group.conf().has("saber outer blocks")) {
      std::vector<SaberOuterBlockParametersWrapper> cmpOuterBlocksParams;
      for (const auto & outerBlockConf : group.conf().getSubConfigurations("saber outer blocks")) {
        SaberOuterBlockParametersWrapper cmpOuterBlockParamsWrapper;
        cmpOuterBlockParamsWrapper.deserialize(outerBlockConf);
        cmpOuterBlocksParams.push_back(cmpOuterBlockParamsWrapper);
      }
      group.outerBlockChain() = std::make_unique<SaberOuterBlockChain>(outerGeometryData,
                                                                group.chainVars(),
                                                                fset4dXb,
                                                                fset4dFg,
                                                                covarConf,
                                                                cmpOuterBlocksParams);
    }

    // Set outer geometry data for central block
    const oops::GeometryData & currentOuterGeom = group.outerBlockChain() ?
                               group.outerBlockChain()->innerGeometryData() : outerGeometryData;

    SaberCentralBlockParametersWrapper saberCentralBlockParamsWrapper;
    saberCentralBlockParamsWrapper.deserialize(
      group.conf().getSubConfiguration("saber central block"));

    const SaberBlockParametersBase & saberCentralBlockParams =
      saberCentralBlockParamsWrapper.saberCentralBlockParameters;
    oops::Log::info() << "Info     : Creating central block: "
                      << saberCentralBlockParams.saberBlockName.value() << std::endl;

    const auto[currentOuterVars, activeVars]
                = group.initCentralBlock(currentOuterGeom,
                                         covarConf,
                                         saberCentralBlockParams,
                                         fset4dXb,
                                         fset4dFg);

    // Check block doesn't expect model fields to be read as this is a generic ctor
    if (group.centralBlock()->getReadConfs().size() != 0) {
      throw eckit::UserError("The generic constructor of the SABER parametric block chain "
                             "does not allow to read MODEL fields.", Here());
    }

    // Check block doesn't expect calibration, as this could be done with the standard ctor
    if (saberCentralBlockParams.doCalibration()) {
      throw eckit::UserError("The generic constructor of the SABER parametric block chain "
                             "does not allow covariance calibration.", Here());
    }
    if (covarConf.has("dual resolution ensemble configuration")) {
      throw eckit::UserError("The generic constructor of the SABER parametric block chain "
                             "does not allow dual resolution ensemble.", Here());
    }
    if (covarConf.has("output ensemble")) {
      throw eckit::UserError("The generic constructor of the SABER parametric block chain "
                             "does not allow ensemble output.", Here());
    }

    if (saberCentralBlockParams.doRead()) {
      // Read data
      oops::Log::info() << "Info     : Read data" << std::endl;
      group.centralBlock()->read();
    }

    if (saberCentralBlockParams.forceWrite.value()) {
      // Write data
      oops::Log::info() << "Info     : Write data" << std::endl;
      group.centralBlock()->write();
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

  oops::Log::trace() << "SaberParametricBlockChain generic ctor done" << std::endl;
}

// -----------------------------------------------------------------------------

void SaberParametricBlockChain::initGroups(const bool & levelsAreTopDown,
                  const oops::Variables & outerVars,
                  const oops::FieldSet4D & fset4dXb,
                  const eckit::Configuration & covarConf,
                  const eckit::Configuration & conf) {
  oops::Log::trace() << "SaberParametricBlockChain::initGroups starting" << std::endl;

  // Get strategy and group configurations
  std::vector<eckit::LocalConfiguration> groupConfs;

  if (!conf.has("groups")) {
    // For backward compatibility, only one group is created in this case
    strategy_ = "deprecated";

    // Finalize group configuration
    eckit::LocalConfiguration groupConf(conf);
    groupConf.set("group name", "deprecated");
    groupConf.set("variables", outerVars.variables());

    // Add group configuration
    groupConfs.push_back(groupConf);
  } else {
    // Get strategy
    strategy_ = conf.getString("multivariate strategy");

    // Get group configurations from conf
    groupConfs = conf.getSubConfigurations("groups");
  }

  // Check multivariate strategy:
  // - Univariate: localization of each group is applied to each variable of the group.
  // - Duplicated: the localization of each group is to the sum of all the fields of the group,
  //   the result is split into the different fields.
  // - Duplicated and weighted: the localization of each group is applied to each variable,
  //   but variables of the same group are combined with user-specified weights.
  // - Crossed: the localization of each group is to the sum of all the fields of the group,
  //   the result is split into the different fields. All the groups share the same control
  //   vector: square-root formulation is necessary.
  ASSERT(strategy_ == "deprecated" ||
         strategy_ == "univariate" ||
         strategy_ == "duplicated" ||
         strategy_ == "duplicated and weighted" ||
         strategy_ == "crossed");

  // Loop over groups
  for (const auto & groupConf : groupConfs) {
    // Get group variables names
    const std::vector<std::string> varNames = groupConf.getStringVector("variables");

    // Define group variables
    oops::Variables groupVars;
    for (const auto & varName : varNames) {
      groupVars.push_back(outerVars[varName]);
    }

    // Check group variables consistency
    size_t levelsCheck = groupVars[0].getLevels();
    oops::Variable refVar = groupVars[0];
    for (const auto & var : groupVars) {
      // Get number of levels
      const size_t varLevels = var.getLevels();

      if ((strategy_ == "univariate") || (strategy_ == "duplicated and weighted")) {
        // All the fields of the group should have the same number of levels.
        ASSERT(levelsCheck == varLevels);
      } else if ((strategy_ == "duplicated") || (strategy_ == "crossed")) {
        // All the fields of the group should have the same number of levels,
        // or only one level if 3D and 2D fields are mixed (in this case, the reference
        // field should be 3D).
        if (varLevels > 1) {
          if (levelsCheck == 1) {
            levelsCheck = varLevels;
            refVar = var;
          } else {
            ASSERT(levelsCheck == varLevels);
          }
        }
      }
    }

    // Create group
    SaberParametricBlockChainGroup group(groupVars, refVar.name(), fset4dXb, covarConf, groupConf);

    if (strategy_ == "deprecated") {
      // Chain variables are outer variables
      group.chainVars() = outerVars;
    } else {
      // Chain variables contain only the reference variable, with the group name
      group.chainVars().push_back(
        oops::Variable(group.name(), refVar.metaData(), refVar.getLevels()));
    }

    // Get the nearest 3D level for 2D variables
    for (auto & var : groupVars) {
      if ((refVar.getLevels() > 1) && (var.getLevels() == 1)) {
        // Get field
        const auto field = fset4dXb[0][var.name()];

        // 2D level only
        const std::string nearest3dLevel = field.metadata().getString("nearest 3d level");
        ASSERT((nearest3dLevel == "top") || (nearest3dLevel == "bottom"));
        size_t lev2d;
        if (levelsAreTopDown) {
          lev2d = (nearest3dLevel == "top") ? 0 : refVar.getLevels()-1;
        } else {
           lev2d = (nearest3dLevel == "bottom") ? 0 : refVar.getLevels()-1;
        }
        group.set2dLevel(var.name(), lev2d);
      }
    }

    // Strategy-specific setup
    if (strategy_ == "duplicated and weighted") {
      // Prepare weights for the "duplicated and weighted" strategy

      // Allocation
      const size_t nv = groupVars.size();
      Eigen::MatrixXd wgt = Eigen::MatrixXd::Zero(nv, nv);
      group.wgtSqrt().resize(nv, nv);

      // Set default weights
      const double defaultWeight = groupConf.getDouble("default off-diagonal weight", 0.0);
      for (size_t jvarJ = 0; jvarJ < groupVars.size(); ++jvarJ) {
        for (size_t jvarI = 0; jvarI < groupVars.size(); ++jvarI) {
          if (jvarJ == jvarI) {
            // Unit diagonal
            wgt(jvarJ, jvarI) = 1.0;
          } else {
            // Default off-diagonal weight
            wgt(jvarJ, jvarI) = defaultWeight;
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
          wgt(jvarI, jvarJ) = weight;
          wgt(jvarJ, jvarI) = weight;
        }
      }

      // Cholesky decomposition
      group.wgtSqrt() = wgt.llt().matrixL();
    }

    // Add group
    groups_.emplace_back(std::move(group));
  }

  oops::Log::trace() << "SaberParametricBlockChain::initGroups done" << std::endl;
}

// -----------------------------------------------------------------------------

void SaberParametricBlockChain::filter(oops::FieldSet4D & fset4d) const {
  if (strategy_ == "deprecated") {
    // Deprecated mode
    groups_[0].filter(fset4d);
  } else {
    throw eckit::Exception("Filter mode should disappear soon, no need to implement it.", Here());
  }
}

// -----------------------------------------------------------------------------

void SaberParametricBlockChain::multiply(oops::FieldSet4D & fset4d) const {
  if (strategy_ == "deprecated") {
    // Deprecated mode
    groups_[0].multiply(fset4d);
  } else {
    if (strategy_ == "univariate") {
      // Univariate strategy
      for (const auto & group : groups_) {
        for (const auto & var : group.variables()) {
          // Create an empty FieldSet4D
          oops::FieldSet4D fset4dTmp({fset4d[0].validTime(), fset4d[0].commGeom()});

          // Get field
          auto field = fset4d[0][var.name()];

          // Add field to empty FieldSet4D
          fset4dTmp[0].add(field);

          // Rename field with the name of the group
          field.rename(group.name());

          // Apply localization
          group.multiply(fset4dTmp);

          // Rename field with its initial name
          field.rename(var.name());
        }
      }
    } else if (strategy_ == "duplicated") {
      // Duplicated strategy
      for (const auto & group : groups_) {
        // Create an empty FieldSet4D
        oops::FieldSet4D fset4dTmp({fset4d[0].validTime(), fset4d[0].commGeom()});

        // Get reference field
        auto refField = fset4d[0][group.refVarName()];

        // Add field to empty FieldSet4D
        fset4dTmp[0].add(refField);

        // Rename field with the name of the group
        refField.rename(group.name());

        // Get reference field view
        auto refView = make_view<double, 2>(refField);

        // Sum of fields
        for (const auto & var : group.variables()) {
          if (var.name() != group.refVarName()) {
            // Get field
            const auto field = fset4d[0][var.name()];

            // Get field view
            const auto view = make_view<double, 2>(field);

            // Check whether the field is a 2D field added to a reference 3D field
            if ((refField.levels() > 1) && (field.levels() == 1)) {
              // 2D level only
              const size_t lev2d = group.get2dLevel(var.name());
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
        group.multiply(fset4dTmp);

        // Split field
        for (const auto & var : group.variables()) {
          if (var.name() == group.refVarName()) {
            // Rename field with its initial name
            refField.rename(var.name());
          } else {
            // Get field
            auto field = fset4d[0][var.name()];

            // Get field view
            auto view = make_view<double, 2>(field);

            // Check whether the field is a 2D field added to a reference 3D field
            if ((refField.levels() > 1) && (field.levels() == 1)) {
              // 2D level only
              const size_t lev2d = group.get2dLevel(var.name());
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
    } else if ((strategy_ == "duplicated and weighted") || (strategy_ == "crossed")) {
      // Crossed strategy or duplicated and weighted strategy

      // Initialization
      atlas::Field ctlVec = atlas::Field("genericCtlVec", make_datatype<double>(),
        make_shape(ctlVecSize()));
      const size_t offset = 0;

      // Apply multiplySqrtAD
      multiplySqrtAD(fset4d, ctlVec, offset);

      // Remove existing fields from output FieldSet4D
      std::vector<std::string> varsToRemove;
      for (const auto & group : groups_) {
        for (const auto & var : group.variables()) {
          varsToRemove.push_back(var.name());
        }
      }
      util::removeFieldsFromFieldSet(fset4d[0].fieldSet(), varsToRemove);

      // Apply multiplySqrt
      multiplySqrt(ctlVec, fset4d, offset);
    } else {
      throw eckit::Exception("invalid multivariate strategy", Here());
    }
  }
}


// -----------------------------------------------------------------------------

void SaberParametricBlockChain::randomize(oops::FieldSet4D & fset4d) const {
  if (strategy_ == "deprecated") {
    // Deprecated mode
    groups_[0].randomize(fset4d);
  } else {
    if (strategy_ == "univariate") {
      // Univariate strategy
      for (const auto & group : groups_) {
        for (const auto & var : group.variables()) {
          // Create an empty FieldSet4D
          oops::FieldSet4D fset4dTmp({fset4d[0].validTime(), fset4d[0].commGeom()});

          // Apply localization
          group.randomize(fset4dTmp);

          // Get field
          auto field = fset4dTmp[0][group.name()];

          // Rename field with its initial name
          field.rename(var.name());

          // Add field to output FieldSet4D
          fset4d[0].add(field);
        }
      }
    } else if ((strategy_ == "duplicated") || (strategy_ == "crossed")) {
      // Duplicated strategy or crossed strategy
      for (const auto & group : groups_) {
        // Create an empty FieldSet4D
        oops::FieldSet4D fset4dTmp({fset4d[0].validTime(), fset4d[0].commGeom()});

        // Apply localization
        group.randomize(fset4dTmp);

        // Get reference field
        auto refField = fset4dTmp[0][group.name()];

        // Rename field with its initial name
        refField.rename(group.refVarName());

        // Add field to output FieldSet4D
        fset4d[0].add(refField);

        // Get reference field view
        const auto refView = make_view<double, 2>(refField);

        // Split field
        for (const auto & var : group.variables()) {
          if (var.name() != group.refVarName()) {
            // Get field
            auto field = refField.functionspace().createField<double>(
              atlas::option::name(var.name()) | atlas::option::levels(var.getLevels()));

            // Add field to output FieldSet4D
            fset4d[0].add(field);

            // Get field view
            auto view = make_view<double, 2>(field);

            // Check whether the field is a 2D field added to a reference 3D field
            if ((refField.levels() > 1) && (field.levels() == 1)) {
              // 2D level only
              const size_t lev2d = group.get2dLevel(var.name());
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
    } else if (strategy_ == "duplicated and weighted") {
      // Duplicated and weighted strategy
      for (const auto & group : groups_) {
        // Get variables
        const auto vars = group.variables();

        for (size_t jvarI = 0; jvarI < vars.size(); ++jvarI) {
          // Get variable
          const auto varI = vars[jvarI];

          // Create an empty FieldSet4D
          oops::FieldSet4D fset4dTmp({fset4d[0].validTime(), fset4d[0].commGeom()});

          // Apply localization
          group.randomize(fset4dTmp);

          // Get field
          const auto field = fset4dTmp[0][group.name()];

          // Get field view
          const auto view = make_view<double, 2>(field);

          for (size_t jvarJ = jvarI; jvarJ < vars.size(); ++jvarJ) {
            // Get variable
            const auto varJ = vars[jvarJ];

            // Other field
            atlas::Field otherField;

            if (fset4d[0].has(varJ.name())) {
              // Get other field
              otherField = fset4d[0][varJ.name()];
            } else {
              // Create other field
              otherField = field.functionspace().createField<double>(
                atlas::option::name(varJ.name()) | atlas::option::levels(varJ.getLevels()));

              // Get other field view
              auto otherView = make_view<double, 2>(otherField);

              otherView.assign(0.0);

              // Add other field
              fset4d[0].add(otherField);
            }

            // Get other field view
            auto otherView = make_view<double, 2>(otherField);

            // Sum weighted off-diagonal fields
            for (int jnode = 0; jnode < otherField.shape(0); ++jnode) {
              for (int jlevel = 0; jlevel < otherField.shape(1); ++jlevel) {
                otherView(jnode, jlevel) += group.wgtSqrt()(jvarJ, jvarI)*view(jnode, jlevel);
              }
            }
          }
        }
      }
    } else {
      throw eckit::Exception("invalid multivariate strategy", Here());
    }
  }
}

// -----------------------------------------------------------------------------

size_t SaberParametricBlockChain::ctlVecSize() const {
  // Initialize control vector size
  size_t ctlVecSize = 0;

  if (strategy_ == "deprecated") {
    // Deprecated mode
    ctlVecSize += groups_[0].ctlVecSize();
  } else {
    if (strategy_ == "univariate") {
      // Univariate strategy
      for (const auto & group : groups_) {
        // Add the group control vector size for each variable
        ctlVecSize += group.ctlVecSize()*group.variables().size();
      }
    } else if (strategy_ == "duplicated") {
      // Duplicated strategy
      for (const auto & group : groups_) {
        // Add the group control vector
        ctlVecSize += group.ctlVecSize();
      }
    } else if (strategy_ == "duplicated and weighted") {
      // Duplicated and weighted strategy
      for (const auto & group : groups_) {
        // Add the group control vector size for each variable
        ctlVecSize += group.ctlVecSize()*group.variables().size();
      }
    } else if (strategy_ == "crossed") {
      // Crossed strategy
      ctlVecSize += groups_[0].ctlVecSize();
    } else {
      throw eckit::Exception("invalid multivariate strategy", Here());
    }
  }

  // Return control vector size
  return ctlVecSize;
}

// -----------------------------------------------------------------------------

void SaberParametricBlockChain::multiplySqrt(const atlas::Field & cv,
                                             oops::FieldSet4D & fset4d,
                                             const size_t & offset) const {
  if (strategy_ == "deprecated") {
    // Deprecated mode
    groups_[0].multiplySqrt(cv, fset4d, offset);
  } else {
    // Initialize index
    size_t index = offset;

    if (strategy_ == "univariate") {
      // Univariate strategy
      for (const auto & group : groups_) {
        for (const auto & var : group.variables()) {
          // Create an empty FieldSet4D
          oops::FieldSet4D fset4dTmp({fset4d[0].validTime(), fset4d[0].commGeom()});

          // Apply localization
          group.multiplySqrt(cv, fset4dTmp, index);

          // Update index
          index += group.ctlVecSize();

          // Get field
          auto field = fset4dTmp[0][group.name()];

          // Rename field with its initial name
          field.rename(var.name());

          // Add field to output FieldSet4D
          fset4d[0].add(field);
        }
      }
    } else if ((strategy_ == "duplicated") || (strategy_ == "crossed")) {
      // Duplicated strategy or crossed strategy
      for (const auto & group : groups_) {
        // Create an empty FieldSet4D
        oops::FieldSet4D fset4dTmp({fset4d[0].validTime(), fset4d[0].commGeom()});

        // Apply localization
        group.multiplySqrt(cv, fset4dTmp, index);

        if (strategy_ == "duplicated") {
          // Update index
          index += group.ctlVecSize();
        }

        // Get reference field
        auto refField = fset4dTmp[0][group.name()];

        // Rename field with its initial name
        refField.rename(group.refVarName());

        // Add field to output FieldSet4D
        fset4d[0].add(refField);

        // Get reference field view
        const auto refView = make_view<double, 2>(refField);

        // Split field
        for (const auto & var : group.variables()) {
          if (var.name() != group.refVarName()) {
            // Get field
            auto field = refField.functionspace().createField<double>(
              atlas::option::name(var.name()) | atlas::option::levels(var.getLevels()));

            // Add field to output FieldSet4D
            fset4d[0].add(field);

            // Get field view
            auto view = make_view<double, 2>(field);

            // Check whether the field is a 2D field added to a reference 3D field
            if ((refField.levels() > 1) && (field.levels() == 1)) {
              // 2D level only
              const size_t lev2d = group.get2dLevel(var.name());
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
    } else if (strategy_ == "duplicated and weighted") {
      // Duplicated and weighted strategy
      for (const auto & group : groups_) {
        // Get variables
        const auto vars = group.variables();

        for (size_t jvarI = 0; jvarI < vars.size(); ++jvarI) {
          // Get variable
          const auto varI = vars[jvarI];

          // Create an empty FieldSet4D
          oops::FieldSet4D fset4dTmp({fset4d[0].validTime(), fset4d[0].commGeom()});

          // Apply localization
          group.multiplySqrt(cv, fset4dTmp, index);

          // Update index
          index += group.ctlVecSize();

          // Get field
          const auto field = fset4dTmp[0][group.name()];

          // Get field view
          const auto view = make_view<double, 2>(field);

          for (size_t jvarJ = jvarI; jvarJ < vars.size(); ++jvarJ) {
            // Get variable
            const auto varJ = vars[jvarJ];

            // Other field
            atlas::Field otherField;

            if (fset4d[0].has(varJ.name())) {
              // Get other field
              otherField = fset4d[0][varJ.name()];
            } else {
              // Create other field
              otherField = field.functionspace().createField<double>(
                atlas::option::name(varJ.name()) | atlas::option::levels(varJ.getLevels()));

              // Get other field view
              auto otherView = make_view<double, 2>(otherField);

              otherView.assign(0.0);

              // Add other field
              fset4d[0].add(otherField);
            }

            // Get other field view
            auto otherView = make_view<double, 2>(otherField);

            // Sum weighted off-diagonal fields
            for (int jnode = 0; jnode < otherField.shape(0); ++jnode) {
              for (int jlevel = 0; jlevel < otherField.shape(1); ++jlevel) {
                otherView(jnode, jlevel) += group.wgtSqrt()(jvarJ, jvarI)*view(jnode, jlevel);
              }
            }
          }
        }
      }
    } else {
      throw eckit::Exception("invalid multivariate strategy", Here());
    }
  }
}

// -----------------------------------------------------------------------------

void SaberParametricBlockChain::multiplySqrtAD(const oops::FieldSet4D & fset4d,
                                               atlas::Field & cv,
                                               const size_t & offset) const {
  if (strategy_ == "deprecated") {
    // Deprecated mode
    groups_[0].multiplySqrtAD(fset4d, cv, offset);
  } else {
    // Initialize index
    size_t index = offset;

    // Initialize control vector
    auto ctlVecView = make_view<double, 1>(cv);
    for (size_t jnode = 0; jnode < ctlVecSize(); ++jnode) {
      ctlVecView(index+jnode) = 0.0;
    }

    if (strategy_ == "univariate") {
      // Univariate strategy
      for (const auto & group : groups_) {
        for (const auto & var : group.variables()) {
          // Create an empty FieldSet4D
          oops::FieldSet4D fset4dTmp({fset4d[0].validTime(), fset4d[0].commGeom()});

          // Clone field
          auto field = fset4d[0][var.name()].clone();

          // Rename field with the name of the group
          field.rename(group.name());

          // Add field
          fset4dTmp[0].add(field);

          // Apply localization
          group.multiplySqrtAD(fset4dTmp, cv, index);

          // Update index
          index += group.ctlVecSize();
        }
      }
    } else if ((strategy_ == "duplicated") || (strategy_ == "crossed")) {
      // Duplicated strategy or crossed strategy
      for (const auto & group : groups_) {
        // Create an empty FieldSet4D
        oops::FieldSet4D fset4dTmp({fset4d[0].validTime(), fset4d[0].commGeom()});

        // Clone reference field
        auto refField = fset4d[0][group.refVarName()].clone();

        // Rename reference field with the name of the group
        refField.rename(group.name());

        // Add reference field
        fset4dTmp[0].add(refField);

        // Get reference field view
        auto refView = make_view<double, 2>(refField);

        // Sum of fields
        for (const auto & var : group.variables()) {
          if (var.name() != group.refVarName()) {
            // Get field
            const auto field = fset4d[0][var.name()];

            // Get field view
            const auto view = make_view<double, 2>(field);

            // Check whether the field is a 2D field added to a reference 3D field
            if ((refField.levels() > 1) && (field.levels() == 1)) {
              // 2D level only
              const size_t lev2d = group.get2dLevel(var.name());
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

        if (strategy_ == "duplicated") {
          // Apply localization
          group.multiplySqrtAD(fset4dTmp, cv, index);

          // Update index
          index += group.ctlVecSize();
        } else if (strategy_ == "crossed") {
          // Create temporary control vector
          atlas::Field ctlVecTmp = atlas::Field("genericCtlVec", make_datatype<double>(),
            make_shape(group.ctlVecSize()));

          // Apply localization
          group.multiplySqrtAD(fset4dTmp, ctlVecTmp, 0);

          // Add control vector contribution
          const auto ctlVecTmpView = make_view<double, 1>(ctlVecTmp);
          for (int jnode = 0; jnode < ctlVecTmp.shape(0); ++jnode) {
            ctlVecView(index+jnode) += ctlVecTmpView(jnode);
          }
        }
      }
    } else if (strategy_ == "duplicated and weighted") {
      // Duplicated and weighted strategy
      for (const auto & group : groups_) {
        // Get variables
        const auto vars = group.variables();

        for (size_t jvarI = 0; jvarI < vars.size(); ++jvarI) {
          // Get variable
          const auto varI = vars[jvarI];

          // Create an empty FieldSet4D
          oops::FieldSet4D fset4dTmp({fset4d[0].validTime(), fset4d[0].commGeom()});

          // Clone field
          auto field = fset4d[0][varI.name()].clone();

          // Rename field with the name of the group
          field.rename(group.name());

          // Add field
          fset4dTmp[0].add(field);

          // Get field view
          auto view = make_view<double, 2>(field);

          // Set to zero
          view.assign(0.0);

          for (size_t jvarJ = jvarI; jvarJ < vars.size(); ++jvarJ) {
            // Get variable
            const auto varJ = vars[jvarJ];

            // Get other field
            const auto otherField = fset4d[0][varJ.name()];

            // Get other field view
            const auto otherView = make_view<double, 2>(otherField);

            // Sum weighted off-diagonal fields
            for (int jnode = 0; jnode < otherField.shape(0); ++jnode) {
              for (int jlevel = 0; jlevel < otherField.shape(1); ++jlevel) {
                view(jnode, jlevel) += group.wgtSqrt()(jvarJ, jvarI)*otherView(jnode, jlevel);
              }
            }
          }

          // Apply localization
          group.multiplySqrtAD(fset4dTmp, cv, index);

          // Update index
          index += group.ctlVecSize();
        }
      }
    } else {
      throw eckit::Exception("invalid multivariate strategy", Here());
    }
  }
}

// -----------------------------------------------------------------------------


}  // namespace saber
