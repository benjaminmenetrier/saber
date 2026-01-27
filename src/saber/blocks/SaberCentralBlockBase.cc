/*
 * (C) Copyright 2021 UCAR
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include "saber/blocks/SaberCentralBlockBase.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "atlas/field.h"

#include "eckit/exception/Exceptions.h"

#include "oops/base/GeometryData.h"
#include "oops/base/Variables.h"
#include "oops/util/AssociativeContainers.h"
#include "oops/util/FieldSetHelpers.h"
#include "oops/util/FieldSetOperations.h"
#include "oops/util/Logger.h"
#include "oops/util/parameters/OptionalParameter.h"
#include "oops/util/parameters/Parameter.h"
#include "oops/util/parameters/Parameters.h"
#include "oops/util/parameters/RequiredParameter.h"
#include "oops/util/parameters/RequiredPolymorphicParameter.h"
#include "oops/util/Printable.h"
#include "oops/util/Random.h"

#include "saber/blocks/SaberBlockParametersBase.h"

namespace saber {

// -----------------------------------------------------------------------------

SaberCentralBlockFactory::SaberCentralBlockFactory(const std::string & name) {
  if (getMakers().find(name) != getMakers().end()) {
    oops::Log::error() << name << " already registered in saber::SaberCentralBlockFactory."
                       << std::endl;
    throw eckit::Exception("Element already registered in saber::SaberCentralBlockFactory.",
      Here());
  }
  getMakers()[name] = this;
}

// -----------------------------------------------------------------------------

std::unique_ptr<SaberCentralBlockBase> SaberCentralBlockFactory::create(
  const oops::GeometryData & geometryData,
  const oops::Variables & vars,
  const eckit::Configuration & covarConf,
  const SaberBlockParametersBase & params,
  const oops::FieldSet3D & xb,
  const oops::FieldSet3D & fg) {
  oops::Log::trace() << "SaberCentralBlockBase::create starting" << std::endl;
  const std::string id = params.saberBlockName;
  typename std::map<std::string, SaberCentralBlockFactory*>::iterator jsb = getMakers().find(id);
  if (jsb == getMakers().end()) {
    oops::Log::error() << id << " does not exist in saber::SaberCentralBlockFactory." << std::endl;
    throw eckit::UserError("Element does not exist in saber::SaberCentralBlockFactory.", Here());
  }
  std::unique_ptr<SaberCentralBlockBase> ptr =
    jsb->second->make(geometryData, vars, covarConf,
                      params, xb, fg);
  oops::Log::trace() << "SaberCentralBlockBase::create done" << std::endl;
  return ptr;
}

// -----------------------------------------------------------------------------

std::unique_ptr<SaberBlockParametersBase>
SaberCentralBlockFactory::createParameters(const std::string &name) {
  typename std::map<std::string, SaberCentralBlockFactory*>::iterator it =
      getMakers().find(name);
  if (it == getMakers().end()) {
    throw std::runtime_error(name + " does not exist in saber::SaberCentralBlockFactory");
  }
  return it->second->makeParameters();
}

// -----------------------------------------------------------------------------

void SaberCentralBlockBase::randomize(oops::FieldSet3D & fset3d) const {
  oops::Log::trace() << "SaberCentralBlockBase::randomize starting" << std::endl;

  // Create control vector
  atlas::Field cv("genericCtlVec", atlas::array::make_datatype<double>(),
    atlas::array::make_shape(ctlVecSize()));

  // Generate random control vector
  randomCtlVec(cv, 0);

  // Square-root multiply
  multiplySqrt(cv, fset3d, 0);

  oops::Log::trace() << "SaberCentralBlockBase::randomize done" << std::endl;
}

// -----------------------------------------------------------------------------

void SaberCentralBlockBase::multiply(oops::FieldSet3D & fset3d) const {
  oops::Log::trace() << "SaberCentralBlockBase::multiply starting" << std::endl;

  // Create control vector
  atlas::Field cv("genericCtlVec", atlas::array::make_datatype<double>(),
    atlas::array::make_shape(ctlVecSize()));

  // Square-root adjoint multiply
  multiplySqrtAD(fset3d, cv, 0);

  // Square-root multiply
  multiplySqrt(cv, fset3d, 0);

  oops::Log::trace() << "SaberCentralBlockBase::multiply done" << std::endl;
}

// -----------------------------------------------------------------------------

}  // namespace saber
