/*
 * (C) Copyright 2025 Meteorologisk Institutt
 *
 */

#include "saber/bifourier/BifourierBalance.h"

namespace saber {
namespace bifourier {

// -----------------------------------------------------------------------------

static SaberOuterBlockMaker<BifourierBalance> makerBifourierBalance_("BifourierBalance");

// -----------------------------------------------------------------------------

BifourierBalance::BifourierBalance(const oops::GeometryData & outerGeometryData,
                                   const oops::Variables & outerVars,
                                   const eckit::Configuration & covarConfig,
                                   const Parameters_ & params,
                                   const oops::FieldSet3D & xb,
                                   const oops::FieldSet3D & fg)
  : SaberOuterBlockBase(params, xb.validTime(), outerGeometryData, outerVars)
{
  oops::Log::trace() << classname() << "::BifourierBalance starting" << std::endl;

  // Setup balance implementation
  balance_ = std::make_unique<BifourierBalanceImpl>(outerGeometryData, outerVars, covarConfig,
    params, xb, fg);

  oops::Log::trace() << classname() << "::BifourierBalance done" << std::endl;
}

// -----------------------------------------------------------------------------

BifourierBalance::~BifourierBalance() {
  oops::Log::trace() << classname() << "::~BifourierBalance starting" << std::endl;

  // Reset balance implementation
  balance_.reset();

  oops::Log::trace() << classname() << "::~BifourierBalance done" << std::endl;
}

// -----------------------------------------------------------------------------

}  // namespace bifourier
}  // namespace saber
