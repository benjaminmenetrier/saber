/*
 * (C) Copyright 2026 Meteorologisk Institutt
 *
 */

#include "saber/bifourier/BifourierCombined.h"

#include "saber/bifourier/BifourierDiagnostics.h"

namespace saber {
namespace bifourier {

// -----------------------------------------------------------------------------

static SaberCentralBlockMaker<BifourierCombined> makerBifourierCombined_("BifourierCombined");

// -----------------------------------------------------------------------------

BifourierCombined::BifourierCombined(const oops::GeometryData & geometryData,
                                     const oops::Variables & centralVars,
                                     const eckit::Configuration & covarConf,
                                     const Parameters_ & params,
                                     const oops::FieldSet3D & xb,
                                     const oops::FieldSet3D & fg)
  : SaberCentralBlockBase(params, xb.validTime(), geometryData, centralVars)
{
  oops::Log::trace() << classname() << "::BifourierCombined starting" << std::endl;

  // Setup balance implementation
  balance_ = std::make_unique<BifourierAromeBalanceImpl>(geometryData, centralVars, covarConf,
    params.balance.value(), xb, fg);

  // Setup covariance implementation
  covar_ = std::make_unique<BifourierAromeCovarianceImpl>(balance_->innerGeometryData(),
    balance_->innerVars(), covarConf, params.covariance.value(), xb, fg);

  oops::Log::trace() << classname() << "::BifourierCombined done" << std::endl;
}

// -----------------------------------------------------------------------------

BifourierCombined::~BifourierCombined() {
  oops::Log::trace() << classname() << "::~BifourierCombined starting" << std::endl;

  // Reset balance implementation
  balance_.reset();

  // Reset covariance implementation
  covar_.reset();

  oops::Log::trace() << classname() << "::~BifourierCombined done" << std::endl;
}

// -----------------------------------------------------------------------------

void BifourierCombined::read() {
  oops::Log::trace() << classname() << "::read starting" << std::endl;

  // Read data
  balance_->read();
  covar_->read();

  if (true) {
    // Compute diagnostics
    computeDiagnostics(*balance_, *covar_, diagsData_);
  }

  oops::Log::trace() << classname() << "::read done" << std::endl;
}

// -----------------------------------------------------------------------------

}  // namespace bifourier
}  // namespace saber

