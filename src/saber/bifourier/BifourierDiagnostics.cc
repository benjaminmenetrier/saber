/*
 * (C) Copyright 2026 Meteorologisk Institutt
 *
 */

#include "saber/bifourier/BifourierDiagnostics.h"

#include "oops/base/Variable.h"
#include "oops/util/Logger.h"

#include "saber/bifourier/BifourierUtilities.h"

using atlas::array::make_datatype;
using atlas::array::make_shape;
using atlas::array::make_view;

namespace saber {
namespace bifourier {

// -----------------------------------------------------------------------------

void computeDiagnostics(const BifourierBalanceImpl & balance,
                        const BifourierCovarianceImpl & covar,
                        atlas::FieldSet & diagsData) {
  oops::Log::trace() << "saber::bifourier::computeDiagnostics starting" << std::endl;

  // Create diagnostics FieldSet
  atlas::FieldSet fset;

  for (const auto & row : balance.params().rows.value()) {
    // Get output variable
    const oops::Variable outputVar = balance.balVars()[row.outputVar.value()];

    // Get number of output levels
    const size_t nzI = outputVar.getLevels();

    for (const auto & inputVarName : row.inputVars.value()) {
      // Get input variable
      const oops::Variable inputVar = balance.balVars()[inputVarName];

      // Get number of input levels
      const size_t nzJ = inputVar.getLevels();

std::cout << "balance.data(): " << balance.data().field_names() << std::endl;
std::cout << "covar.data(): " << covar.data().field_names() << std::endl;
      // Get regression view
      const auto regView = getView3D("reg", outputVar, inputVar, balance.data());

      // Get vv-covariance field
      const auto vvCovView = getView3D("cov", inputVar, covar.data());

      // Balanced covariance name
      const std::string balCovName = "balCov" + inputVar.name();

      // Create balanced covariance field
      createField3D(balCovName, balance.trans()->nw(), outputVar, fset);

      // Get balanced covariance view
      auto balCovView = getView3D(balCovName, outputVar, outputVar, fset);

      // Compute balanced covariance multiplication
      for (size_t jw = 0; jw < balance.trans()->nw(); ++jw) {
        for (size_t jzI1 = 0; jzI1 < nzI; ++jzI1) {
          for (size_t jzI2 = 0; jzI2 < nzI; ++jzI2) {
            for (size_t jzJ1 = 0; jzJ1 < nzJ; ++jzJ1) {
              for (size_t jzJ2 = 0; jzJ2 < nzJ; ++jzJ2) {
                balCovView(jw, jzI1, jzI2) += regView(jw, jzI1, jzJ1)*vvCovView(jw, jzJ1, jzJ2)
                  *regView(jw, jzI2, jzJ2);
              }
            }
          }
        }
      }
    }
  } 

  oops::Log::trace() << "saber::bifourier::computeDiagnostics done" << std::endl;
}

// -----------------------------------------------------------------------------

}  // namespace bifourier
}  // namespace saber
