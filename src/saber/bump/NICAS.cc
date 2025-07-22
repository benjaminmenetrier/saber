/*
 * (C) Copyright 2021 UCAR
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include "saber/bump/NICAS.h"

#include <tuple>
#include <unordered_map>

#include "eckit/exception/Exceptions.h"

#include "oops/util/FieldSetOperations.h"
#include "oops/util/Logger.h"
#include "oops/util/Timer.h"

#include "saber/oops/Utilities.h"

namespace saber {
namespace bump {

// -----------------------------------------------------------------------------

static SaberCentralBlockMaker<NICAS> makerNICAS_("BUMP_NICAS");

// -----------------------------------------------------------------------------

static std::unordered_map<std::string, std::tuple<std::unique_ptr<BUMP>, bool, bool>> bumps;

// -----------------------------------------------------------------------------

NICAS::NICAS(const oops::GeometryData & geometryData,
             const oops::Variables & centralVars,
             const eckit::Configuration & covarConf,
             const Parameters_ & params,
             const oops::FieldSet3D & xb,
             const oops::FieldSet3D & fg)
  : SaberCentralBlockBase(params, xb.validTime()),
    activeVars_(getActiveVars(params, centralVars)),
    bumpParams_(params.calibrationParams.value() != boost::none ? *params.calibrationParams.value()
      : *params.readParams.value()),
    memberIndex_(0) {
  oops::Log::trace() << classname() << "::NICAS starting" << std::endl;

  // Create temporary BUMP configuration to define BUMP UID
  eckit::LocalConfiguration bumpConf;
  bumpParams_.serialize(bumpConf);

  // Add grid UID in BUMP configuration
  bumpConf.set("grid uid", util::getGridUid(geometryData.functionSpace()));

  // Add variables in BUMP configuration
  bumpConf.set("central variables", centralVars.variables());

  // Add covariance configuration in BUMP configuration
  eckit::LocalConfiguration localCovarConf(covarConf);
  bumpConf.set("covariance configuration", localCovarConf);

  // Define BUMP parameters UID
  std::unique_ptr<eckit::Hash> h(eckit::HashFactory::instance().build("MD5"));
  bumpConf.hash(*h);
  bumpUid_ = h->digest();

  // Create BUMP instance if it does not exist yet
  const auto itBUMP = bumps.find(bumpUid_);
  if (itBUMP == bumps.end()) {
    oops::Log::info() << "Create BUMP ID: " << bumpUid_ << std::endl;

    // Create and store BUMP instance
    std::unique_ptr<BUMP> bump(new BUMP(geometryData, activeVars_, covarConf, bumpParams_,
      params.fieldsMetaData.value(), xb));
    bumps.emplace(bumpUid_, std::make_tuple(std::move(bump), false, false));

    // Read input ATLAS files
    std::get<0>(bumps[bumpUid_])->readAtlasFiles();
  } else {
    oops::Log::info() << "Re-use BUMP ID: " << bumpUid_ << std::endl;
  }

  oops::Log::trace() << classname() << "::NICAS done" << std::endl;
}

// -----------------------------------------------------------------------------

NICAS::~NICAS() {
  oops::Log::trace() << classname() << "::~NICAS starting" << std::endl;
  util::Timer timer(classname(), "~NICAS");
  oops::Log::trace() << classname() << "::~NICAS done" << std::endl;
}

// -----------------------------------------------------------------------------

void NICAS::randomize(oops::FieldSet3D & fset) const {
  oops::Log::trace() << classname() << "::randomize starting" << std::endl;
  std::get<0>(bumps[bumpUid_])->randomizeNicas(fset);
  oops::Log::trace() << classname() << "::randomize done" << std::endl;
}

// -----------------------------------------------------------------------------

size_t NICAS::ctlVecSize() const {
  return std::get<0>(bumps[bumpUid_])->getCvSize();
}


// -----------------------------------------------------------------------------

void NICAS::multiply(oops::FieldSet3D & fset) const {
  oops::Log::trace() << classname() << "::multiply starting" << std::endl;
  std::get<0>(bumps[bumpUid_])->multiplyNicas(fset);
  oops::Log::trace() << classname() << "::multiply done" << std::endl;
}

// -----------------------------------------------------------------------------

std::vector<std::pair<std::string, eckit::LocalConfiguration>> NICAS::getReadConfs() const {
  oops::Log::trace() << classname() << "::getReadConfs starting" << std::endl;
  std::vector<eckit::LocalConfiguration> inputModelFilesConf;
  if (!std::get<1>(bumps[bumpUid_])) {
    inputModelFilesConf = bumpParams_.inputModelFilesConf.value().get_value_or({});
  }
  oops::Log::trace() << classname() << "::getReadConfs done" << std::endl;
  return std::get<0>(bumps[bumpUid_])->getReadConfs(inputModelFilesConf);
}

// -----------------------------------------------------------------------------

void NICAS::setReadFields(const std::vector<oops::FieldSet3D> & fsetVec) {
  oops::Log::trace() << classname() << "::setReadFields starting" << std::endl;
  if (!std::get<1>(bumps[bumpUid_])) {
    for (const auto & fset : fsetVec) {
      std::get<0>(bumps[bumpUid_])->addField(fset);
    }
  }
  oops::Log::trace() << classname() << "::setReadFields done" << std::endl;
}

// -----------------------------------------------------------------------------

void NICAS::read() {
  oops::Log::trace() << classname() << "::read starting" << std::endl;
  if (!std::get<1>(bumps[bumpUid_])) {
    std::get<0>(bumps[bumpUid_])->runDrivers();
    std::get<1>(bumps[bumpUid_]) = true;
  }
  oops::Log::trace() << classname() << "::read done" << std::endl;
}

// -----------------------------------------------------------------------------

void NICAS::directCalibration(const oops::FieldSets & fsetEns) {
  oops::Log::trace() << classname() << "::directCalibration starting" << std::endl;
  if (!std::get<1>(bumps[bumpUid_])) {
    std::get<0>(bumps[bumpUid_])->addEnsemble(fsetEns);
    std::get<0>(bumps[bumpUid_])->runDrivers();
    std::get<1>(bumps[bumpUid_]) = true;
  }
  oops::Log::trace() << classname() << "::directCalibration done" << std::endl;
}

// -----------------------------------------------------------------------------

void NICAS::iterativeCalibrationInit() {
  oops::Log::trace() << classname() << "::iterativeCalibrationInit starting" << std::endl;
  if (!std::get<1>(bumps[bumpUid_])) {
    memberIndex_ = 0;
  }
  oops::Log::trace() << classname() << "::iterativeCalibrationInit done" << std::endl;
}

// -----------------------------------------------------------------------------

void NICAS::iterativeCalibrationUpdate(const oops::FieldSet3D & fset) {
  oops::Log::trace() << classname() << "::iterativeCalibrationUpdate starting" << std::endl;
  if (!std::get<1>(bumps[bumpUid_])) {
    std::get<0>(bumps[bumpUid_])->iterativeUpdate(fset, memberIndex_);
    ++memberIndex_;
  }
  oops::Log::trace() << classname() << "::iterativeCalibrationUpdate done" << std::endl;
}

// -----------------------------------------------------------------------------

void NICAS::iterativeCalibrationFinal() {
  oops::Log::trace() << classname() << "::iterativeCalibrationFinal starting" << std::endl;
  if (!std::get<1>(bumps[bumpUid_])) {
    std::get<0>(bumps[bumpUid_])->runDrivers();
    std::get<1>(bumps[bumpUid_]) = true;
  }
  oops::Log::trace() << classname() << "::iterativeCalibrationFinal done" << std::endl;
}

// -----------------------------------------------------------------------------

void NICAS::dualResolutionSetup(const oops::GeometryData & geometryData) {
  oops::Log::trace() << classname() << "::dualResolutionSetup starting" << std::endl;
  if (!std::get<2>(bumps[bumpUid_])) {
    std::get<0>(bumps[bumpUid_])->dualResolutionSetup(geometryData.functionSpace(),
      geometryData.fieldSet());
    std::get<1>(bumps[bumpUid_]) = false;
    std::get<2>(bumps[bumpUid_]) = true;
  }
  oops::Log::trace() << classname() << "::dualResolutionSetup done" << std::endl;
}

// -----------------------------------------------------------------------------

void NICAS::multiplySqrt(const atlas::Field & cv,
                         oops::FieldSet3D & fset,
                         const size_t & offset) const {
  oops::Log::trace() << classname() << "::multiplySqrt starting" << std::endl;
  std::get<0>(bumps[bumpUid_])->multiplyNicasSqrt(cv, fset, offset);
  oops::Log::trace() << classname() << "::multiplySqrt done" << std::endl;
}

// -----------------------------------------------------------------------------

void NICAS::multiplySqrtAD(const oops::FieldSet3D & fset,
                           atlas::Field & cv,
                           const size_t & offset) const {
  oops::Log::trace() << classname() << "::multiplySqrtAD starting" << std::endl;
  std::get<0>(bumps[bumpUid_])->multiplyNicasSqrtAd(fset, cv, offset);
  oops::Log::trace() << classname() << "::multiplySqrtAD done" << std::endl;
}

// -----------------------------------------------------------------------------

std::vector<std::pair<eckit::LocalConfiguration, oops::FieldSet3D>> NICAS::fieldsToWrite() const {
  oops::Log::trace() << classname() << "::fieldsToWrite starting" << std::endl;
  std::vector<eckit::LocalConfiguration> outputModelFilesConf
    = bumpParams_.outputModelFilesConf.value().get_value_or({});
  if (outputModelFilesConf.size() > 0) {
    // Print log info
    oops::Log::info() << "Info     : "
                      << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                      << std::endl;
    oops::Log::info() << "Info     : +++ Get parameters from BUMP"
                      << std::endl;
  }

  // Return configuration/fieldset pairs
  std::vector<std::pair<eckit::LocalConfiguration, oops::FieldSet3D>> pairs
    = std::get<0>(bumps[bumpUid_])->fieldsToWrite(outputModelFilesConf);

  if (outputModelFilesConf.size() > 0) {
    // Print log info
    oops::Log::info() << "Info     : "
                      << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                      << std::endl;
    oops::Log::info() << "Info     : +++ Write files"
                      << std::endl;
  }
  oops::Log::trace() << classname() << "::fieldsToWrite done" << std::endl;
  return pairs;
}

// -----------------------------------------------------------------------------

void NICAS::write() const {
  oops::Log::trace() << classname() << "::write starting" << std::endl;
  std::get<0>(bumps[bumpUid_])->writeAtlasFiles();
  oops::Log::trace() << classname() << "::write done" << std::endl;
}

// -----------------------------------------------------------------------------

void NICAS::print(std::ostream & os) const {
  os << classname();
}

// -----------------------------------------------------------------------------

}  // namespace bump
}  // namespace saber
