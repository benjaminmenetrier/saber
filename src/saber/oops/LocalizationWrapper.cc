/*
 * (C) Copyright 2025 Meteorologisk Institutt
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include "saber/oops/LocalizationWrapper.h"

using atlas::array::make_datatype;
using atlas::array::make_shape;
using atlas::array::make_view;

namespace saber {

// -----------------------------------------------------------------------------

LocalizationWrapper::~LocalizationWrapper() {
  util::Timer timer(classname(), "~LocalizationWrapper");
  oops::Log::trace() << classname() << "::~LocalizationWrapper destructed" << std::endl;
}

// -----------------------------------------------------------------------------

void LocalizationWrapper::randomize(oops::FieldSet4D & fset4d) const {
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

void LocalizationWrapper::multiply(oops::FieldSet4D & fset4d) const {
  util::Timer timer(classname(), "multiply");
  oops::Log::trace() << classname() << "::multiply starting" << std::endl;

  if (strategy_ == "deprecated") {
    // Deprecated mode
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

          // Apply localization
          group.localization()->multiply(fset4d);

          // Rename the field with its initial name
          field.rename(var.name());
        }
      }
    } else if (strategy_ == "duplicated") {
      // Duplicated: the localization of each group is to the sum of all the fields of the group,
      // the result is split into the different fields
      for (const auto & group : groups_) {
        // Get reference field
        auto refField = fset4d[0][group.referenceVariable()];

        // Rename the field with the name of the group
        refField.rename(group.name());

        // Get reference field view
        auto refView = make_view<double, 2>(refField);

        // Sum of fields
        for (const auto & var : group.variables()) {
          if (var.name() != group.referenceVariable()) {
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
        group.localization()->multiply(fset4d);

        // Split field
        for (const auto & var : group.variables()) {
          if (var.name() == group.referenceVariable()) {
            // Rename the field with its initial name
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
    } else if (strategy_ == "crossed") {
      // Crossed: the localization of each group is to the sum of all the fields of the group,
      // the result is split into the different fields. All the groups share the same control
      // vector: square-root formulation is necessary.

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
    } else if (strategy_ == "duplicated and weighted") {
      throw eckit::Exception("not implemented yet", Here());
    } else {
      throw eckit::Exception("invalid multivariate strategy", Here());
    }
  }

  oops::Log::trace() << classname() << "::multiply done" << std::endl;
}

// -----------------------------------------------------------------------------

size_t LocalizationWrapper::ctlVecSize() const {
  oops::Log::trace() << classname() << "::ctlVecSize starting" << std::endl;

  // Initialize control vector size
  size_t ctlVecSize = 0;

  if (strategy_ == "deprecated") {
    // Deprecated mode
    ctlVecSize += groups_[0].localization()->ctlVecSize();
  } else {
    if (strategy_ == "univariate") {
      // Univariate: localization of each group is applied to each variable of the group
      for (const auto & group : groups_) {
        // Add the group control vector size for each variable
        ctlVecSize += group.localization()->ctlVecSize()*group.variables().size();
      }
    } else if (strategy_ == "duplicated") {
      // Duplicated: the localization of each group is to the sum of all the fields of the group,
      // the result is split into the different fields
      for (const auto & group : groups_) {
        // Add the group control vector
        ctlVecSize += group.localization()->ctlVecSize();
      }
    } else if (strategy_ == "crossed") {
      // Crossed: the localization of each group is to the sum of all the fields of the group,
      // the result is split into the different fields. All the groups share the same control
      // vector: square-root formulation is necessary.
      ctlVecSize += groups_[0].localization()->ctlVecSize();
    } else if (strategy_ == "duplicated and weighted") {
      throw eckit::Exception("not implemented yet", Here());
    } else {
      throw eckit::Exception("invalid multivariate strategy", Here());
    }
  }

  // Return control vector size
  oops::Log::trace() << classname() << "::ctlVecSize done" << std::endl;
  return ctlVecSize;
}

// -----------------------------------------------------------------------------

void LocalizationWrapper::multiplySqrt(const atlas::Field & cv,
                                       oops::FieldSet4D & fset4d,
                                       const size_t & offset) const {
  oops::Log::trace() << classname() << "::multiplySqrt starting" << std::endl;

  if (strategy_ == "deprecated") {
    // Deprecated mode
    groups_[0].localization()->multiplySqrt(cv, fset4d, offset);
  } else {
    // Initialize index
    size_t index = offset;

    if (strategy_ == "univariate") {
      // Univariate: localization of each group is applied to each variable of the group
      for (const auto & group : groups_) {
        for (const auto & var : group.variables()) {
          // Create an empty FieldSet4D
          oops::FieldSet4D fset4dTmp({fset4d[0].validTime(), fset4d[0].commGeom()});

          // Apply localization
          group.localization()->multiplySqrt(cv, fset4dTmp, index);

          // Update index
          index += group.localization()->ctlVecSize();

          // Get field
          auto field = fset4dTmp[0][group.name()];

          // Rename the field with its initial name
          field.rename(var.name());

          // Add field to output FieldSet4D
          fset4d[0].add(field);
        }
      }
    } else if ((strategy_ == "duplicated") || (strategy_ == "crossed")) {
      // Duplicated: the localization of each group is to the sum of all the fields of the group,
      // the result is split into the different fields
      // Crossed: the localization of each group is to the sum of all the fields of the group,
      // the result is split into the different fields. All the groups share the same control
      // vector: square-root formulation is necessary.
      for (const auto & group : groups_) {
        // Create an empty FieldSet4D
        oops::FieldSet4D fset4dTmp({fset4d[0].validTime(), fset4d[0].commGeom()});

        // Apply localization
        group.localization()->multiplySqrt(cv, fset4dTmp, index);

        if (strategy_ == "duplicated") {
          // Update index
          index += group.localization()->ctlVecSize();
        }

        // Get reference field
        auto refField = fset4dTmp[0][group.name()];

        // Rename the field with its initial name
        refField.rename(group.referenceVariable());

        // Add field to output FieldSet4D
        fset4d[0].add(refField);

        // Get reference field view
        const auto refView = make_view<double, 2>(refField);

        // Split field
        for (const auto & var : group.variables()) {
          if (var.name() != group.referenceVariable()) {
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
      throw eckit::Exception("not implemented yet", Here());
    } else {
      throw eckit::Exception("invalid multivariate strategy", Here());
    }
  }

  oops::Log::trace() << classname() << "::multiplySqrt done" << std::endl;
}

// -----------------------------------------------------------------------------

void LocalizationWrapper::multiplySqrtAD(const oops::FieldSet4D & fset4d,
                                         atlas::Field & cv,
                                         const size_t & offset) const {
  oops::Log::trace() << classname() << "::multiplySqrtAD starting" << std::endl;

  if (strategy_ == "deprecated") {
    // Deprecated mode
    groups_[0].localization()->multiplySqrtAD(fset4d, cv, offset);
  } else {
    // Initialize index
    size_t index = offset;

    // Initialize control vector
    auto ctlVecView = make_view<double, 1>(cv);
    for (size_t jnode = 0; jnode < ctlVecSize(); ++jnode) {
      ctlVecView(index+jnode) = 0.0;
    }

    if (strategy_ == "univariate") {
      // Univariate: localization of each group is applied to each variable of the group
      for (const auto & group : groups_) {
        for (const auto & var : group.variables()) {
          // Get field
          auto field = fset4d[0][var.name()];

          // Rename the field with the name of the group
          field.rename(group.name());

          // Apply localization
          group.localization()->multiplySqrtAD(fset4d, cv, index);

          // Update index
          index += group.localization()->ctlVecSize();

          // Rename the field with its initial name
          field.rename(var.name());
        }
      }
    } else if ((strategy_ == "duplicated") || (strategy_ == "crossed")) {
      // Duplicated: the localization of each group is to the sum of all the fields of the group,
      // the result is split into the different fields
      // Crossed: the localization of each group is to the sum of all the fields of the group,
      // the result is split into the different fields. All the groups share the same control
      // vector: square-root formulation is necessary.

      for (const auto & group : groups_) {
        // Get reference field
        auto refField = fset4d[0][group.referenceVariable()];

        // Rename the field with the name of the group
        refField.rename(group.name());

        // Get reference field view
        auto refView = make_view<double, 2>(refField);

        // Sum of fields
        for (const auto & var : group.variables()) {
          if (var.name() != group.referenceVariable()) {
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
          group.localization()->multiplySqrtAD(fset4d, cv, index);

          // Update index
          index += group.localization()->ctlVecSize();
        } else if (strategy_ == "crossed") {
          // Create temporary control vector
          atlas::Field ctlVecTmp = atlas::Field("genericCtlVec", make_datatype<double>(),
            make_shape(group.localization()->ctlVecSize()));

          // Apply localization
          group.localization()->multiplySqrtAD(fset4d, ctlVecTmp, 0);

          // Add control vector contribution
          const auto ctlVecTmpView = make_view<double, 1>(ctlVecTmp);
          for (int jnode = 0; jnode < ctlVecTmp.shape(0); ++jnode) {
            ctlVecView(index+jnode) += ctlVecTmpView(jnode);
          }
        }

        // Split field
        for (const auto & var : group.variables()) {
          if (var.name() == group.referenceVariable()) {
            // Rename the field with its initial name
            refField.rename(var.name());
          }
        }
      }
    } else if (strategy_ == "duplicated and weighted") {
      throw eckit::Exception("not implemented yet", Here());
    } else {
      throw eckit::Exception("invalid multivariate strategy", Here());
    }
  }

  oops::Log::trace() << classname() << "::multiplySqrtAD done" << std::endl;
}

// -----------------------------------------------------------------------------

}  // namespace saber
