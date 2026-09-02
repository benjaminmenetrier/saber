/*
 * (C) Copyright 2025 Meteorologisk Institutt
 *
 */

#pragma once

#include <string>

#include "saber/bifourier/BifourierCovarianceImpl.h"

namespace saber {
namespace bifourier {

// -----------------------------------------------------------------------------

class BifourierAromeCovarianceImplReadParameters : public BifourierCovarianceImplReadParameters {
  OOPS_CONCRETE_PARAMETERS(BifourierAromeCovarianceImplReadParameters,
    BifourierCovarianceImplReadParameters)

 public:
  // Input file format ("netcdf", "arome legacy binary" or "arome legacy netcdf")
  oops::Parameter<std::string> inputFileFormat{"input file format", "netcdf", this};
};

// -----------------------------------------------------------------------------

class BifourierAromeCovarianceImplWriteParameters : public BifourierCovarianceImplWriteParameters {
  OOPS_CONCRETE_PARAMETERS(BifourierAromeCovarianceImplWriteParameters,
    BifourierCovarianceImplWriteParameters)

 public:
  // Output file format ("netcdf", "arome legacy binary" or "arome legacy netcdf")
  oops::Parameter<std::string> outputFileFormat{"output file format", "netcdf", this};
};

// -----------------------------------------------------------------------------

class BifourierAromeCovarianceImplParameters : public BifourierCovarianceImplParameters {
  OOPS_CONCRETE_PARAMETERS(BifourierAromeCovarianceImplParameters, BifourierCovarianceImplParameters)

 public:
  // Read parameters
  oops::OptionalParameter<BifourierAromeCovarianceImplReadParameters> read{"read", this};

  // Write parameters
  oops::OptionalParameter<BifourierAromeCovarianceImplWriteParameters> write{"write", this};

  // REDNMC factor
  oops::Parameter<double> rednmc{"rednmc", std::sqrt(0.5), this};

  oops::Variables mandatoryActiveVars() const override
    {return oops::Variables();}
};


// -----------------------------------------------------------------------------

class BifourierAromeCovarianceImpl : public BifourierCovarianceImpl {
 public:
  static const std::string classname()
    {return "saber::bifourier::BifourierAromeCovarianceImpl";}

  typedef BifourierAromeCovarianceImplParameters Parameters_;

  BifourierAromeCovarianceImpl(const oops::GeometryData & geometryData,
                               const oops::Variables & centralVars,
                               const eckit::Configuration & covarConf,
                               const Parameters_ & params,
                               const oops::FieldSet3D & xb,
                               const oops::FieldSet3D & fg)
  : BifourierCovarianceImpl(geometryData, centralVars, covarConf, params, xb, fg), params_(params)
  {}

  void read();

  void write() const;

 private:
  // Parameters
  Parameters_ params_;

  // Private methods

  // Define AROME weights
  double aromeWeight(const size_t &) const;
};

// -----------------------------------------------------------------------------

}  // namespace bifourier
}  // namespace saber
