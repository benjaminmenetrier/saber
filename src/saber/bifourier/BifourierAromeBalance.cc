/*
 * (C) Copyright 2025 Meteorologisk Institutt
 *
 */

#include "saber/bifourier/BifourierAromeBalance.h"

namespace saber {
namespace bifourier {

// -----------------------------------------------------------------------------

static SaberOuterBlockMaker<BifourierAromeBalance> makerBifourierAromeBalance_("BifourierAromeBalance");

// -----------------------------------------------------------------------------

BifourierAromeBalance::BifourierAromeBalance(const oops::GeometryData & outerGeometryData,
                                   const oops::Variables & outerVars,
                                   const eckit::Configuration & covarConfig,
                                   const Parameters_ & params,
                                   const oops::FieldSet3D & xb,
                                   const oops::FieldSet3D & fg)
  : SaberOuterBlockBase(params, xb.validTime(), outerGeometryData, outerVars)
{
  oops::Log::trace() << classname() << "::BifourierAromeBalance starting" << std::endl;

  // Setup balance implementation
  balance_ = std::make_unique<BifourierAromeBalanceImpl>(outerGeometryData, outerVars, covarConfig,
    params, xb, fg);

  oops::Log::trace() << classname() << "::BifourierAromeBalance done" << std::endl;
}

// -----------------------------------------------------------------------------

BifourierAromeBalance::~BifourierAromeBalance() {
  oops::Log::trace() << classname() << "::~BifourierAromeBalance starting" << std::endl;

  // Reset balance implementation
  balance_.reset();

  oops::Log::trace() << classname() << "::~BifourierAromeBalance done" << std::endl;
}

// -----------------------------------------------------------------------------

}  // namespace bifourier
}  // namespace saber
