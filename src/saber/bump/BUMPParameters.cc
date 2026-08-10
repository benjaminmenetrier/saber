/*
 * (C) Copyright 2021 UCAR
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include "saber/bump/BUMPParameters.h"

namespace saber {
namespace bump {

// -----------------------------------------------------------------------------

void bump_config_init_f90(eckit::LocalConfiguration * config) {
  // Create parameters with default values
  BUMPParameters bumpParams;

  // Serialize parameters (to get values)
  bumpParams.serialize(*config);
}

// -----------------------------------------------------------------------------

}  // namespace bump
}  // namespace saber
