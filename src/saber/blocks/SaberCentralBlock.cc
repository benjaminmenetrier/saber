/*
 * (C) Crown Copyright 2024 Met Office
 * (C) Copyright 2025 Meteorologisk Institutt
 * (C) Copyright 2025- UCAR
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include "saber/blocks/SaberCentralBlock.h"

#include "oops/util/FieldSetOperations.h"
#include "oops/util/Random.h"

using atlas::array::make_datatype;
using atlas::array::make_shape;
using atlas::array::make_view;

namespace saber {

// -----------------------------------------------------------------------------

bool SaberCentralBlockParameters::doCalibration() const {
  oops::Log::trace() << "SaberCentralBlockParameters::doCalibration starting" << std::endl;

  if (this->singleBlock.value()) {
    // Single block
    return this->singleBlock.value()->doCalibration();
  }
  if (this->groups.value()) {
    // Multiple blocks
    for (const auto & groupParams : this->groups.value().get()) {
      if (groupParams.block.value().doCalibration()) {
        return true;
      }
    }
  }
  return false;
}

// -----------------------------------------------------------------------------

bool SaberCentralBlockParameters::doRead() const {
  oops::Log::trace() << "SaberCentralBlockParameters::doRead starting" << std::endl;

  if (this->singleBlock.value()) {
    // Single block
    return this->singleBlock.value()->doRead();
  }
  if (this->groups.value()) {
    // Multiple blocks
    for (const auto & groupParams : this->groups.value().get()) {
      if (groupParams.block.value().doRead()) {
        return true;
      }
    }
  }
  return false;
}

// -----------------------------------------------------------------------------

SaberCentralBlock::SaberCentralBlock(const oops::GeometryData & outerGeom,
                                     const bool levelsAreTopDown,
                                     const oops::Variables & outerVars,
                                     const eckit::Configuration & covarConf,
                                     const SaberCentralBlockParameters & params,
                                     const oops::FieldSet3D & xb,
                                     const oops::FieldSet3D & fg)
  : geometryData_(outerGeom), validTime_(xb.validTime()), params_(params),
    strategy_(params.strategy) {
  oops::Log::trace() << "SaberCentralBlock constructor starting" << std::endl;

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

  oops::Log::info() << "Info     : SaberCentralBlock using multivariate strategy: "
                    << strategy_ << std::endl;
  // Check if it's a single group
  if (params.singleBlock.value() && (strategy_ != "deprecated")) {
    throw eckit::UserError("SaberCentralBlock: single block can only be used with the "
                           "'deprecated' strategy.", Here());
  }
  if (strategy_ == "deprecated" && params.groups.value() &&
             (params.groups.value().get().size() > 1)) {
    throw eckit::UserError("SaberCentralBlock: 'deprecated' strategy can only be used with "
                           "a single block.", Here());
  }
  if (params.singleBlock.value() && params.groups.value()) {
    throw eckit::UserError("SaberCentralBlock: please specify either a single block or multiple "
                           "groups, not both.", Here());
  }

  if (params.singleBlock.value()) {
    // Single block case
    oops::Log::info() << "Info     : Creating single central block." << std::endl;
    groupInputVars_.push_back(outerVars);
    groupNames_.push_back("single block");
    groupInnerVars_.push_back(outerVars);
    groupRefVars_.push_back(outerVars[0]);
    doCalibration_.push_back(params.singleBlock.value()->doCalibration());
    doRead_.push_back(params.singleBlock.value()->doRead());
    forceWrite_.push_back(params.singleBlock.value()->forceWrite.value());
    groups_.push_back(SaberCentralBlockFactory::create(outerGeom,
                                                       groupInnerVars_.back(),
                                                       covarConf,
                                                       *(params.singleBlock.value()),
                                                       xb,
                                                       fg));
  } else {
    // Loop over groups
    for (const auto & groupParams : params.groups.value().get()) {
      // Get group variables names
      oops::Log::info() << "Info     : Creating central block group: "
                        << groupParams.groupName.value() << std::endl;
      const oops::Variables varNames = groupParams.variables.value();

      // Define group variables
      oops::Variables groupVars;
      for (const auto & varName : varNames) {
        groupVars.push_back(outerVars[varName.name()]);
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

      // Create central block for this group
      groupInputVars_.push_back(groupVars);
      groupNames_.push_back(groupParams.groupName.value());

      // Chain variables contain only the reference variable, with the group name
      groupInnerVars_.push_back(oops::Variables(
        {oops::Variable(groupNames_.back(), refVar.metaData(), refVar.getLevels())}));
      groupRefVars_.push_back(refVar);
      doCalibration_.push_back(groupParams.block.value().doCalibration());
      doRead_.push_back(groupParams.block.value().doRead());
      forceWrite_.push_back(groupParams.block.value().forceWrite.value());
      groups_.push_back(SaberCentralBlockFactory::create(outerGeom,
                                                         groupInnerVars_.back(),
                                                         covarConf,
                                                         groupParams.block.value(),
                                                         xb,
                                                         fg));


      // Get the nearest 3D level for 2D variables
      for (auto & var : groupVars) {
        if ((refVar.getLevels() > 1) && (var.getLevels() == 1)) {
          // Get field
          const auto field = xb[var.name()];

          // 2D level only
          const std::string nearest3dLevel = field.metadata().getString("nearest 3d level");
          ASSERT((nearest3dLevel == "top") || (nearest3dLevel == "bottom"));
          size_t lev2d;
          if (levelsAreTopDown) {
            lev2d = (nearest3dLevel == "top") ? 0 : refVar.getLevels()-1;
          } else {
             lev2d = (nearest3dLevel == "bottom") ? 0 : refVar.getLevels()-1;
          }
          lev2d_.insert({var.name(), lev2d});
        }
      }

      // Strategy-specific setup
      if (strategy_ == "duplicated and weighted") {
        // Prepare weights for the "duplicated and weighted" strategy

        // Allocation
        const size_t nv = groupVars.size();
        Eigen::MatrixXd wgt = Eigen::MatrixXd::Zero(nv, nv);

        // Set default weights
        const double defaultWeight = groupParams.defOffDiagWeight.value();
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
        if (groupParams.offDiagWeights.value()) {
          for (const auto & specWeight : *groupParams.offDiagWeights.value()) {
            // Get variables pair and weight
            const std::vector<std::string> varPair = specWeight.varPair.value();
            ASSERT(varPair.size() == 2);
            const double weight = specWeight.weight.value();

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
        wgtSqrt_.push_back(wgt.llt().matrixL());
        oops::Log::info() << "Info     : Weights square-root matrix for group "
                        << groupNames_.back() << " :" << std::endl;
        oops::Log::info() << wgtSqrt_.back() << std::endl;
      }
    }
  }

  oops::Log::trace() << "SaberCentralBlock::SaberCentralBlock done" << std::endl;
}

// -----------------------------------------------------------------------------

void SaberCentralBlock::filter(oops::FieldSet3D & fset3d) const {
  if (strategy_ == "deprecated") {
    // Deprecated mode
    groups_[0]->filter(fset3d);
  } else {
    throw eckit::Exception("Filter mode should disappear soon, no need to implement it.", Here());
  }
}

// -----------------------------------------------------------------------------

void SaberCentralBlock::multiply(oops::FieldSet3D & fset3d) const {
  oops::Log::trace() << "SaberCentralBlock::multiply starting" << std::endl;

  if (strategy_ == "deprecated") {
    // Deprecated mode
    groups_[0]->multiply(fset3d);
  } else {
    if (strategy_ == "univariate") {
      // Univariate strategy
      for (size_t igroup = 0; igroup < groups_.size(); ++igroup) {
        const auto & group = groups_[igroup];
        for (const auto & var : groupInputVars_[igroup]) {
          // Create an empty FieldSet3D
          oops::FieldSet3D fset3dTmp({fset3d.validTime(), fset3d.commGeom()});

          // Get field
          auto field = fset3d[var.name()];

          // Add field to empty FieldSet3D
          fset3dTmp.add(field);

          // Rename field with the name of the group
          field.rename(groupNames_[igroup]);

          // Apply localization
          group->multiply(fset3dTmp);

          // Rename field with its initial name
          field.rename(var.name());
        }
      }
    } else if (strategy_ == "duplicated") {
      // Duplicated strategy
      for (size_t igroup = 0; igroup < groups_.size(); ++igroup) {
        const auto & group = groups_[igroup];
        // Create an empty FieldSet3D
        oops::FieldSet3D fset3dTmp({fset3d.validTime(), fset3d.commGeom()});

        // Get reference field
        auto refField = fset3d[groupRefVars_[igroup].name()];

        // Add field to empty FieldSet3D
        fset3dTmp.add(refField);

        // Rename field with the name of the group
        refField.rename(groupNames_[igroup]);

        // Get reference field view
        auto refView = make_view<double, 2>(refField);

        // Sum of fields
        for (const auto & var : groupInputVars_[igroup]) {
          if (var.name() != groupRefVars_[igroup].name()) {
            // Get field
            const auto field = fset3d[var.name()];

            // Get field view
            const auto view = make_view<double, 2>(field);

            // Check whether the field is a 2D field added to a reference 3D field
            if ((refField.levels() > 1) && (field.levels() == 1)) {
              // 2D level only
              const size_t lev2d = lev2d_.at(var.name());
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
        group->multiply(fset3dTmp);

        // Split field
        for (const auto & var : groupInputVars_[igroup]) {
          if (var.name() == groupRefVars_[igroup].name()) {
            // Rename field with its initial name
            refField.rename(var.name());
          } else {
            // Get field
            auto field = fset3d[var.name()];

            // Get field view
            auto view = make_view<double, 2>(field);

            // Check whether the field is a 2D field added to a reference 3D field
            if ((refField.levels() > 1) && (field.levels() == 1)) {
              // 2D level only
              const size_t lev2d = lev2d_.at(var.name());
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
      multiplySqrtAD(fset3d, ctlVec, offset);

      // Apply multiplySqrt
      multiplySqrt(ctlVec, fset3d, offset);
    } else {
      throw eckit::Exception("invalid multivariate strategy", Here());
    }
  }

  oops::Log::trace() << "SaberCentralBlock::multiply done" << std::endl;
}

// -----------------------------------------------------------------------------

void SaberCentralBlock::randomize(oops::FieldSet3D & fset3d) const {
  oops::Log::trace() << "SaberCentralBlock::randomize starting" << std::endl;

  if (strategy_ == "deprecated") {
    // Deprecated mode
    groups_[0]->randomize(fset3d);
  } else {
    if (strategy_ == "univariate") {
      // Univariate strategy
      for (size_t igroup = 0; igroup < groups_.size(); ++igroup) {
        const auto & group = groups_[igroup];
        for (const auto & var : groupInputVars_[igroup]) {
          // Create an empty FieldSet3D
          oops::FieldSet3D fset3dTmp({fset3d.validTime(), fset3d.commGeom()});

          // Get field
          auto field = fset3d[var.name()];

          // Add field to empty FieldSet3D
          fset3dTmp.add(field);

          // Rename field with the name of the group
          field.rename(groupNames_[igroup]);

          // Apply localization
          group->randomize(fset3dTmp);

          // Rename field with its initial name
          field.rename(var.name());
        }
      }
    } else if ((strategy_ == "duplicated") || (strategy_ == "crossed")) {
      // Duplicated strategy or crossed strategy
      for (size_t igroup = 0; igroup < groups_.size(); ++igroup) {
        const auto & group = groups_[igroup];
        // Create an empty FieldSet3D
        oops::FieldSet3D fset3dTmp({fset3d.validTime(), fset3d.commGeom()});

        // Get reference field
        auto refField = fset3d[groupRefVars_[igroup].name()];

        // Add field to empty FieldSet3D
        fset3dTmp.add(refField);

        // Rename field with the name of the group
        refField.rename(groupNames_[igroup]);

        // Apply localization
        group->randomize(fset3dTmp);

        // Rename field with its initial name
        refField.rename(groupRefVars_[igroup].name());

        // Get reference field view
        const auto refView = make_view<double, 2>(refField);

        // Split field
        for (const auto & var : groupInputVars_[igroup]) {
          if (var.name() != groupRefVars_[igroup].name()) {
            // Get field
            auto field = fset3d[var.name()];

            // Get field view
            auto view = make_view<double, 2>(field);

            // Check whether the field is a 2D field added to a reference 3D field
            if ((refField.levels() > 1) && (field.levels() == 1)) {
              // 2D level only
              const size_t lev2d = lev2d_.at(var.name());
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
      for (size_t igroup = 0; igroup < groups_.size(); ++igroup) {
        const auto & group = groups_[igroup];
        // Get variables
        const auto vars = groupInputVars_[igroup];
        for (size_t jvarI = 0; jvarI < vars.size(); ++jvarI) {
          // Get variable
          const auto varI = vars[jvarI];

          // Create an empty FieldSet3D
          oops::FieldSet3D fset3dTmp({fset3d.validTime(), fset3d.commGeom()});
          fset3dTmp.init(geometryData_.functionSpace(), groupInnerVars_[igroup]);

          // Apply localization
          group->randomize(fset3dTmp);
          // Get field
          const auto field = fset3dTmp[groupInnerVars_[igroup][0].name()];

          // Get field view
          const auto view = make_view<double, 2>(field);

          for (size_t jvarJ = jvarI; jvarJ < vars.size(); ++jvarJ) {
            // Get variable
            const auto varJ = vars[jvarJ];

            // Other field
            atlas::Field otherField;

            if (fset3d.has(varJ.name())) {
              // Get other field
              otherField = fset3d[varJ.name()];
            } else {
              // Create other field
              otherField = field.functionspace().createField<double>(
                atlas::option::name(varJ.name()) | atlas::option::levels(varJ.getLevels()));

              // Get other field view
              auto otherView = make_view<double, 2>(otherField);

              otherView.assign(0.0);

              // Add other field
              fset3d.add(otherField);
            }

            // Get other field view
            auto otherView = make_view<double, 2>(otherField);

            // Sum weighted off-diagonal fields
            for (int jnode = 0; jnode < otherField.shape(0); ++jnode) {
              for (int jlevel = 0; jlevel < otherField.shape(1); ++jlevel) {
                otherView(jnode, jlevel) += wgtSqrt_[igroup](jvarJ, jvarI)*view(jnode, jlevel);
              }
            }
          }
        }
      }
    } else {
      throw eckit::Exception("invalid multivariate strategy", Here());
    }
  }

  oops::Log::trace() << "SaberCentralBlock::randomize done" << std::endl;
}

// -----------------------------------------------------------------------------

size_t SaberCentralBlock::ctlVecSize() const {
  oops::Log::trace() << "SaberCentralBlock::ctlVecSize starting" << std::endl;

  // Initialize control vector size
  size_t ctlVecSize = 0;

  if (strategy_ == "deprecated") {
    // Deprecated mode
    ctlVecSize += groups_[0]->ctlVecSize();
  } else {
    if (strategy_ == "univariate") {
      // Univariate strategy
      for (size_t igroup = 0; igroup < groups_.size(); ++igroup) {
        // Add the group control vector size for each variable
        ctlVecSize += groups_[igroup]->ctlVecSize()*groupInputVars_[igroup].size();
      }
    } else if (strategy_ == "duplicated") {
      // Duplicated strategy
      for (const auto & group : groups_) {
        // Add the group control vector
        ctlVecSize += group->ctlVecSize();
      }
    } else if (strategy_ == "duplicated and weighted") {
      // Duplicated and weighted strategy
      for (size_t igroup = 0; igroup < groups_.size(); ++igroup) {
        // Add the group control vector size for each variable
        ctlVecSize += groups_[igroup]->ctlVecSize()*groupInputVars_[igroup].size();
      }
    } else if (strategy_ == "crossed") {
      // Crossed strategy
      ctlVecSize += groups_[0]->ctlVecSize();
    } else {
      throw eckit::Exception("invalid multivariate strategy", Here());
    }
  }

  // Return control vector size
  return ctlVecSize;
}

// -----------------------------------------------------------------------------

void SaberCentralBlock::multiplySqrt(const atlas::Field & cv,
                                     oops::FieldSet3D & fset3d,
                                     const size_t & offset) const {
  oops::Log::trace() << "SaberCentralBlock::multiplySqrt starting" << std::endl;

  if (strategy_ == "deprecated") {
    // Deprecated mode
    groups_[0]->multiplySqrt(cv, fset3d, offset);
  } else {
    // Initialize index
    size_t index = offset;

    if (strategy_ == "univariate") {
      // Univariate strategy
      for (size_t igroup = 0; igroup < groups_.size(); ++igroup) {
        const auto & group = groups_[igroup];
        for (const auto & var : groupInputVars_[igroup]) {
          // Create an empty FieldSet3D
          oops::FieldSet3D fset3dTmp({fset3d.validTime(), fset3d.commGeom()});

          // Get field
          auto field = fset3d[var.name()];

          // Add field to empty FieldSet3D
          fset3dTmp.add(field);

          // Rename field with the name of the group
          field.rename(groupNames_[igroup]);

          // Apply localization
          group->multiplySqrt(cv, fset3dTmp, index);

          // Update index
          index += group->ctlVecSize();

          // Rename field with its initial name
          field.rename(var.name());
        }
      }
    } else if ((strategy_ == "duplicated") || (strategy_ == "crossed")) {
      // Duplicated strategy or crossed strategy
      for (size_t igroup = 0; igroup < groups_.size(); ++igroup) {
        const auto & group = groups_[igroup];
        // Create an empty FieldSet3D
        oops::FieldSet3D fset3dTmp({fset3d.validTime(), fset3d.commGeom()});

        // Get reference field
        auto refField = fset3d[groupRefVars_[igroup].name()];

        // Add field to empty FieldSet3D
        fset3dTmp.add(refField);

        // Rename field with the name of the group
        refField.rename(groupNames_[igroup]);

        // Apply localization
        group->multiplySqrt(cv, fset3dTmp, index);

        if (strategy_ == "duplicated") {
          // Update index
          index += group->ctlVecSize();
        }

        // Rename field with its initial name
        refField.rename(groupRefVars_[igroup].name());

        // Get reference field view
        const auto refView = make_view<double, 2>(refField);

        // Split field
        for (const auto & var : groupInputVars_[igroup]) {
          if (var.name() != groupRefVars_[igroup].name()) {
            // Get field
            auto field = fset3d[var.name()];

            // Get field view
            auto view = make_view<double, 2>(field);

            // Check whether the field is a 2D field added to a reference 3D field
            if ((refField.levels() > 1) && (field.levels() == 1)) {
              // 2D level only
              const size_t lev2d = lev2d_.at(var.name());
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
      for (size_t igroup = 0; igroup < groups_.size(); ++igroup) {
        const auto & group = groups_[igroup];
        // Get variables
        const auto vars = groupInputVars_[igroup];
        util::zeroFieldSet(fset3d.fieldSet());
        for (size_t jvarI = 0; jvarI < vars.size(); ++jvarI) {
          // Get variable
          const auto varI = vars[jvarI];

          // Create an empty FieldSet3D
          oops::FieldSet3D fset3dTmp({fset3d.validTime(), fset3d.commGeom()});
          fset3dTmp.init(geometryData_.functionSpace(), groupInnerVars_[igroup]);

          // Apply localization
          group->multiplySqrt(cv, fset3dTmp, index);

          // Update index
          index += group->ctlVecSize();

          // Get field
          const auto field = fset3dTmp[groupNames_[igroup]];
          // Get field view
          const auto view = make_view<double, 2>(field);

          for (size_t jvarJ = jvarI; jvarJ < vars.size(); ++jvarJ) {
            // Get variable
            const auto varJ = vars[jvarJ];

            // Other field
            atlas::Field otherField = fset3d[varJ.name()];

            // Get other field view
            auto otherView = make_view<double, 2>(otherField);

            // Sum weighted off-diagonal fields
            for (int jnode = 0; jnode < otherField.shape(0); ++jnode) {
              for (int jlevel = 0; jlevel < otherField.shape(1); ++jlevel) {
                otherView(jnode, jlevel) += wgtSqrt_[igroup](jvarJ, jvarI)*view(jnode, jlevel);
              }
            }
          }
        }
      }
    } else {
      throw eckit::Exception("invalid multivariate strategy", Here());
    }
  }

  oops::Log::trace() << "SaberCentralBlock::multiplySqrt done" << std::endl;
}

// -----------------------------------------------------------------------------

void SaberCentralBlock::multiplySqrtAD(const oops::FieldSet3D & fset3d,
                                               atlas::Field & cv,
                                               const size_t & offset) const {
  oops::Log::trace() << "SaberCentralBlock::multiplySqrtAD starting" << std::endl;

  if (strategy_ == "deprecated") {
    // Deprecated mode
    groups_[0]->multiplySqrtAD(fset3d, cv, offset);
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
      for (size_t igroup = 0; igroup < groups_.size(); ++igroup) {
        const auto & group = groups_[igroup];
        for (const auto & var : groupInputVars_[igroup]) {
          // Create an empty FieldSet3D
          oops::FieldSet3D fset3dTmp({fset3d.validTime(), fset3d.commGeom()});

          // Clone field
          auto field = fset3d[var.name()].clone();

          // Rename field with the name of the group
          field.rename(groupNames_[igroup]);

          // Add field
          fset3dTmp.add(field);

          // Apply localization
          group->multiplySqrtAD(fset3dTmp, cv, index);

          // Update index
          index += group->ctlVecSize();
        }
      }
    } else if ((strategy_ == "duplicated") || (strategy_ == "crossed")) {
      // Duplicated strategy or crossed strategy
      for (size_t igroup = 0; igroup < groups_.size(); ++igroup) {
        const auto & group = groups_[igroup];
        // Create an empty FieldSet3D
        oops::FieldSet3D fset3dTmp({fset3d.validTime(), fset3d.commGeom()});

        // Clone reference field
        auto refField = fset3d[groupRefVars_[igroup].name()].clone();

        // Rename reference field with the name of the group
        refField.rename(groupNames_[igroup]);

        // Add reference field
        fset3dTmp.add(refField);

        // Get reference field view
        auto refView = make_view<double, 2>(refField);

        // Sum of fields
        for (const auto & var : groupInputVars_[igroup]) {
          if (var.name() != groupRefVars_[igroup].name()) {
            // Get field
            const auto field = fset3d[var.name()];

            // Get field view
            const auto view = make_view<double, 2>(field);

            // Check whether the field is a 2D field added to a reference 3D field
            if ((refField.levels() > 1) && (field.levels() == 1)) {
              // 2D level only
              const size_t lev2d = lev2d_.at(var.name());
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
          group->multiplySqrtAD(fset3dTmp, cv, index);

          // Update index
          index += group->ctlVecSize();
        } else if (strategy_ == "crossed") {
          // Create temporary control vector
          atlas::Field ctlVecTmp = atlas::Field("genericCtlVec", make_datatype<double>(),
            make_shape(group->ctlVecSize()));
          // Apply localization
          group->multiplySqrtAD(fset3dTmp, ctlVecTmp, 0);

          // Add control vector contribution
          const auto ctlVecTmpView = make_view<double, 1>(ctlVecTmp);
          for (int jnode = 0; jnode < ctlVecTmp.shape(0); ++jnode) {
            ctlVecView(index+jnode) += ctlVecTmpView(jnode);
          }
        }
      }
    } else if (strategy_ == "duplicated and weighted") {
      // Duplicated and weighted strategy
      for (size_t igroup = 0; igroup < groups_.size(); ++igroup) {
        const auto & group = groups_[igroup];
        // Get variables
        const auto vars = groupInputVars_[igroup];

        for (size_t jvarI = 0; jvarI < vars.size(); ++jvarI) {
          // Get variable
          const auto varI = vars[jvarI];

          // Create an empty FieldSet3D
          oops::FieldSet3D fset3dTmp({fset3d.validTime(), fset3d.commGeom()});

          // Clone field
          auto field = fset3d[varI.name()].clone();

          // Rename field with the name of the group
          field.rename(groupNames_[igroup]);

          // Add field
          fset3dTmp.add(field);

          // Get field view
          auto view = make_view<double, 2>(field);

          // Set to zero
          view.assign(0.0);

          for (size_t jvarJ = jvarI; jvarJ < vars.size(); ++jvarJ) {
            // Get variable
            const auto varJ = vars[jvarJ];

            // Get other field
            const auto otherField = fset3d[varJ.name()];

            // Get other field view
            const auto otherView = make_view<double, 2>(otherField);

            // Sum weighted off-diagonal fields
            for (int jnode = 0; jnode < otherField.shape(0); ++jnode) {
              for (int jlevel = 0; jlevel < otherField.shape(1); ++jlevel) {
                view(jnode, jlevel) += wgtSqrt_[igroup](jvarJ, jvarI)*otherView(jnode, jlevel);
              }
            }
          }

          // Apply localization
          group->multiplySqrtAD(fset3dTmp, cv, index);

          // Update index
          index += group->ctlVecSize();
        }
      }
    } else {
      throw eckit::Exception("invalid multivariate strategy", Here());
    }
  }

  oops::Log::trace() << "SaberCentralBlock::multiplySqrtAD done" << std::endl;
}

// -----------------------------------------------------------------------------

void SaberCentralBlock::adjointTest(const oops::GeometryData & geometryData,
                                    const double & globalAdjointTolerance) const {
  oops::Log::trace() << "SaberCentralBlock::adjointTest starting" << std::endl;

  for (size_t igroup = 0; igroup < groups_.size(); ++igroup) {
    // Override adjoint tolerance if specified in the configuration
    double adjointTolerance = globalAdjointTolerance;
    if (params_.singleBlock.value() && params_.singleBlock.value()->adjointTolerance.value()) {
      adjointTolerance = *params_.singleBlock.value()->adjointTolerance.value();
    } else if (params_.groups.value() &&
               params_.groups.value().get()[igroup].block.value().adjointTolerance.value()) {
      adjointTolerance =
        *params_.groups.value().get()[igroup].block.value().adjointTolerance.value();
    }
    // Create random FieldSets
    oops::FieldSet3D fset1 = oops::randomFieldSet3D(validTime_,
                                                    geometryData.comm(),
                                                    geometryData.functionSpace(),
                                                    groupInnerVars_[igroup]);
    oops::FieldSet3D fset2 = oops::randomFieldSet3D(validTime_,
                                                    geometryData.comm(),
                                                    geometryData.functionSpace(),
                                                    groupInnerVars_[igroup]);

    // Copy FieldSets
    oops::FieldSet3D fset1Save(fset1);
    oops::FieldSet3D fset2Save(fset2);

    // Apply forward multiplication only (self-adjointness test)
    groups_[igroup]->multiply(fset1);
    groups_[igroup]->multiply(fset2);

    // Compute adjoint test
    const double dp1 = fset1.dot_product_with(fset2Save, groupInnerVars_[igroup]);
    const double dp2 = fset2.dot_product_with(fset1Save, groupInnerVars_[igroup]);
    oops::Log::info() << std::setprecision(16) << "Info     : Adjoint test: (Ax)^t y = " << dp1
                      << ": x^t (Ay) = " << dp2 << " : adjoint tolerance = "
                      << adjointTolerance << std::endl;
    oops::Log::test() << "Adjoint test for block " << groups_[igroup]->blockName();
    if (std::abs(dp1-dp2)/std::abs(0.5*(dp1+dp2)) < adjointTolerance) {
      oops::Log::test() << " passed" << std::endl;
    } else {
      oops::Log::test() << " failed" << std::endl;
      throw eckit::Exception("Adjoint test failure for block " +
        groups_[igroup]->blockName(), Here());
    }
  }

  oops::Log::trace() << "SaberCentralBlock::adjointTest done" << std::endl;
}

// -----------------------------------------------------------------------------

void SaberCentralBlock::sqrtTest(const oops::GeometryData & geometryData,
                                 const double & globalSqrtTolerance) const {
  oops::Log::trace() << "SaberOuterBlockBase::sqrtTest starting" << std::endl;

  for (size_t igroup = 0; igroup < groups_.size(); ++igroup) {
    // Override square root tolerance if specified in the configuration
    double sqrtTolerance = globalSqrtTolerance;
    if (params_.singleBlock.value() && params_.singleBlock.value()->sqrtTolerance.value()) {
      sqrtTolerance = *params_.singleBlock.value()->sqrtTolerance.value();
    } else if (params_.groups.value() &&
               params_.groups.value().get()[igroup].block.value().sqrtTolerance.value()) {
      sqrtTolerance = *params_.groups.value().get()[igroup].block.value().sqrtTolerance.value();
    }

    // Square-root test
    // Create FieldSet
    oops::FieldSet3D fset = oops::randomFieldSet3D(validTime_,
                                                   geometryData.comm(),
                                                   geometryData.functionSpace(),
                                                   groupInnerVars_[igroup]);

    // Copy FieldSet
    oops::FieldSet3D fsetSave(fset);

    // Create control vector for this group
    const size_t ctlVecSize = groups_[igroup]->ctlVecSize();
    oops::Log::info() << "Control vector size for block " << groups_[igroup]->blockName() << ": "
                      << ctlVecSize << std::endl;
    atlas::Field ctlVec = atlas::Field("genericCtlVec",
                                       atlas::array::make_datatype<double>(),
                                       atlas::array::make_shape(ctlVecSize));
    size_t seed = 7;  // To avoid impact on future random generator calls
    util::NormalDistribution<double> dist(ctlVecSize, 0.0, 1.0, seed);
    auto view = atlas::array::make_view<double, 1>(ctlVec);
    for (size_t jnode = 0; jnode < ctlVecSize; ++jnode) {
      view(jnode) = dist[jnode];
    }

    // Copy control vector
    atlas::Field ctlVecSave = atlas::Field("genericCtlVec",
                                           atlas::array::make_datatype<double>(),
                                           atlas::array::make_shape(ctlVecSize));
    auto viewSave = atlas::array::make_view<double, 1>(ctlVecSave);
    viewSave.assign(view);

    // Apply square-root multiplication
    groups_[igroup]->multiplySqrt(ctlVecSave, fset, 0);

    // Apply square-root adjoint multiplication
    groups_[igroup]->multiplySqrtAD(fsetSave, ctlVec, 0);

    // Compute adjoint test
    const double dp1 = fset.dot_product_with(fsetSave, groupInnerVars_[igroup]);
    double dp2 = 0.0;
    for (size_t jnode = 0; jnode < ctlVecSize; ++jnode) {
      dp2 += view(jnode)*viewSave(jnode);
    }
    geometryData.comm().allReduceInPlace(dp2, eckit::mpi::sum());
    oops::Log::info() << std::setprecision(16) << "Info     : Square-root test: y^t (Ux) = " << dp1
                      << ": x^t (U^t y) = " << dp2 << " : square-root tolerance = "
                      << sqrtTolerance << std::endl;
    const bool adjComparison = (std::abs(dp1-dp2)/std::abs(0.5*(dp1+dp2)) < sqrtTolerance);

    // Apply square-root multiplication
    groups_[igroup]->multiplySqrt(ctlVec, fset, 0);

    // Apply full multiplication
    groups_[igroup]->multiply(fsetSave);

    // Check that the fieldsets are similar within tolerance
    const bool sqrtComparison = fset.compare_with(fsetSave, sqrtTolerance,
                                                  util::ToleranceType::relative);
    if (sqrtComparison) {
      oops::Log::info() << "Info     : Square-root test passed: U U^t x == B x" << std::endl;
    } else {
      oops::Log::info() << "Info     : Square-root test failed: U U^t x != B x" << std::endl;
    }

    // Print results
    oops::Log::test() << "Square-root test for block " << groups_[igroup]->blockName();
    if (adjComparison && sqrtComparison) {
      oops::Log::test() << " passed" << std::endl;
    } else {
      oops::Log::test() << " failed" << std::endl;
      throw eckit::Exception("Square-root test failure for block "
         + groups_[igroup]->blockName(), Here());
    }
  }

  oops::Log::trace() << "SaberCentralBlock::sqrtTest done" << std::endl;
}


}  // namespace saber

