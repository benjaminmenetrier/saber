/*
 * (C) Copyright 2026 Meteorologisk Institutt
 *
 */

#pragma once

#include "atlas/field.h"

#include "saber/bifourier/BifourierBalanceImpl.h"
#include "saber/bifourier/BifourierCovarianceImpl.h"

namespace saber {
namespace bifourier {

// -----------------------------------------------------------------------------

// Compute diagnostics
void computeDiagnostics(const BifourierBalanceImpl &,
                        const BifourierCovarianceImpl &,
                        atlas::FieldSet &);

// -----------------------------------------------------------------------------

}  // namespace bifourier
}  // namespace saber
