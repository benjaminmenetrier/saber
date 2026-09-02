/*
 * (C) Copyright 2025 Meteorologisk Institutt
 *
 */

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "atlas/field.h"

#include "oops/base/GeometryData.h"
#include "oops/util/parameters/Parameters.h"

#include "saber/bifourier/BifourierCovarianceImpl.h"
#include "saber/bifourier/BifourierTransformBase.h"
#include "saber/bifourier/BifourierTransformStore.h"
#include "saber/blocks/SaberBlockParametersBase.h"
#include "saber/blocks/SaberOuterBlockBase.h"

namespace saber {
namespace bifourier {

// -----------------------------------------------------------------------------

class BifourierBalanceImplReadParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(BifourierBalanceImplReadParameters, oops::Parameters)

 public:
  // Input file
  oops::RequiredParameter<std::string> inputFile{"input file", this};
};

// -----------------------------------------------------------------------------

class BifourierBalanceImplCalibrationParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(BifourierBalanceImplCalibrationParameters, oops::Parameters)

 public:
  // Use full recursive inverse formula to compute the regression
  oops::Parameter<bool> fullRecursiveInverse{"full recursive inverse", false, this};

  // Filtering scale (in total wavenumber unit)
  oops::Parameter<double> filteringScale{"filtering scale", 0.0, this};

  // Remaining variance fraction (between 0 and 1) in the auto-covariance inversion
  oops::Parameter<double> remainingVar{"remaining variance fraction", 1.0, this};

  // Old covariance input file
  oops::OptionalParameter<std::string> oldCovInputFile{"old covariance input file", this};

  // Half life
  oops::OptionalParameter<double> halfLife{"half life", this};

  // Cycle index
  oops::OptionalParameter<size_t> cycleIndex{"cycle index", this};

  // Sub-ensembles size
  oops::Parameter<size_t> subEnsSize{"sub-ensembles size", 0, this};
};

// -----------------------------------------------------------------------------

class BifourierBalanceImplWriteParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(BifourierBalanceImplWriteParameters, oops::Parameters)

 public:
  // Output file
  oops::RequiredParameter<std::string> outputFile{"output file", this};

  // Write covariance flag
  oops::Parameter<bool> writeCovariance{"write covariance", false, this};

  // Write diagnostics flag
  oops::Parameter<bool> writeDiagnostics{"write diagnostics", false, this};
};

// -----------------------------------------------------------------------------

class BifourierBalanceImplRowParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(BifourierBalanceImplRowParameters, oops::Parameters)

 public:
  // Output variable
  oops::RequiredParameter<std::string> outputVar{"output variable", this};

  // Input variables
  oops::Parameter<std::vector<std::string>> inputVars{"input variables", {}, this};
};

// -----------------------------------------------------------------------------

class BifourierBalanceImplParameters : public SaberBlockParametersBase {
  OOPS_CONCRETE_PARAMETERS(BifourierBalanceImplParameters, SaberBlockParametersBase)

 public:
  // Read parameters
  oops::OptionalParameter<BifourierBalanceImplReadParameters> read{"read", this};

  // Calibration parameters
  oops::OptionalParameter<BifourierBalanceImplCalibrationParameters> calibration{"calibration", this};

  // Write parameters
  oops::OptionalParameter<BifourierBalanceImplWriteParameters> write{"write", this};

  // Rows
  oops::RequiredParameter<std::vector<BifourierBalanceImplRowParameters>>
    rows{"rows", this};

  // Extra variable for auto-covariances
  oops::OptionalParameter<std::string> extraVar{"extra variable for auto-covariances", this};

  oops::Variables mandatoryActiveVars() const
    {return oops::Variables();}
};

// -----------------------------------------------------------------------------

class BifourierBalanceImpl {
 public:
  static const std::string classname()
    {return "saber::bifourier::BifourierBalanceImpl";}

  typedef BifourierBalanceImplParameters Parameters_;

  BifourierBalanceImpl(const oops::GeometryData &,
                       const oops::Variables &,
                       const eckit::Configuration &,
                       const Parameters_ &,
                       const oops::FieldSet3D &,
                       const oops::FieldSet3D &);
  virtual ~BifourierBalanceImpl() = default;

  const oops::GeometryData & innerGeometryData() const
    {return innerGeometryData_;}
  const oops::Variables & innerVars() const
    {return innerVars_;}

  void multiply(oops::FieldSet3D &) const;
  void multiplyAD(oops::FieldSet3D &) const;
  void leftInverseMultiply(oops::FieldSet3D &) const;

  void read();

  void directCalibration(const oops::FieldSets &);

  void iterativeCalibrationInit();
  void iterativeCalibrationUpdate(const oops::FieldSet3D &);
  void iterativeCalibrationFinal();

  void write() const;

  void print(std::ostream &) const;

  // Specific accessors
  const eckit::mpi::Comm & comm() const
    {return comm_;}
  atlas::FieldSet & data()
    {return data_;}
  const std::shared_ptr<BifourierTransformBase> & trans() const
    {return trans_;}
  oops::Variables & balVars()
    {return balVars_;}

 protected:
  // Inner geometry data
  const oops::GeometryData & innerGeometryData_;

  // Communicator
  const eckit::mpi::Comm & comm_;

  // Inner variables
  oops::Variables innerVars_;

  // Parameters
  Parameters_ params_;

  // Filtering length-scale
  const double Lf_;

  // Spectral transform
  const BifourierTransformStore transStore_;
  const std::shared_ptr<BifourierTransformBase> trans_;

  // Ordered variables
  oops::Variables balVars_;

  // Ordered variables, with a possible extra variable in first position for auto-covariances
  oops::Variables balVarsExt_;

  // Number of active regression components
  size_t nCmp_;

  // Data
  atlas::FieldSet data_;

  // Interative counter
  size_t iterativeN_;

  // Covariance implementation (for diagnostics)
  std::unique_ptr<BifourierCovarianceImpl> covar_;

  // Private methods

  // Read covariance
  void readCovariance();

  // Compute regression
  void computeRegression(const std::vector<std::string> &,
                         const oops::Variable &);

  // Compute regressions from covariances
  void computeRegressionsFromCovariances();

  // Get variables to compute full covariances with
  oops::Variables xxCovVars(const oops::Variable &) const;

  // Compute diagnostics
  void computeDiagnostics() const;
};

// -----------------------------------------------------------------------------

}  // namespace bifourier
}  // namespace saber
