/*
 * (C) Copyright 2025 Meteorologisk Institutt
 *
 */

#pragma once

#include <string>
#include <vector>

#include "saber/bifourier/BifourierBalanceImpl.h"

namespace saber {
namespace bifourier {

// -----------------------------------------------------------------------------

class BifourierAromeBalanceImplReadParameters : public BifourierBalanceImplReadParameters {
  OOPS_CONCRETE_PARAMETERS(BifourierAromeBalanceImplReadParameters, BifourierBalanceImplReadParameters)

 public:
  // Input file format ("netcdf", "arome legacy binary" or "arome legacy netcdf")
  oops::Parameter<std::string> inputFileFormat{"input file format", "netcdf", this};
};

// -----------------------------------------------------------------------------

class BifourierAromeBalanceImplWriteParameters : public BifourierBalanceImplWriteParameters {
  OOPS_CONCRETE_PARAMETERS(BifourierAromeBalanceImplWriteParameters, BifourierBalanceImplWriteParameters)

 public:
  // Output file
  oops::RequiredParameter<std::string> outputFile{"output file", this};

  // Output file format ("netcdf", "arome legacy binary" or "arome legacy netcdf")
  oops::Parameter<std::string> outputFileFormat{"output file format", "netcdf", this};
};

// -----------------------------------------------------------------------------

class BalancedAirPressureParameters : public oops::Parameters {
  OOPS_CONCRETE_PARAMETERS(BalancedAirPressureParameters, oops::Parameters)

 public:
  // Zonal wavenumbers size
  oops::RequiredParameter<size_t> M{"zonal truncation", this};

  // Meridional wavenumbers size
  oops::RequiredParameter<size_t> N{"meridional truncation", this};

  // Mean latitude
  oops::RequiredParameter<double> meanLat{"mean latitude", this};
};

// -----------------------------------------------------------------------------

class BifourierAromeBalanceImplParameters : public BifourierBalanceImplParameters {
  OOPS_CONCRETE_PARAMETERS(BifourierAromeBalanceImplParameters, BifourierBalanceImplParameters)

 public:
  // Read parameters
  oops::OptionalParameter<BifourierAromeBalanceImplReadParameters> read{"read", this};

  // Write parameters
  oops::OptionalParameter<BifourierAromeBalanceImplWriteParameters> write{"write", this};

  // Explicit balanced air pressure parameters
  oops::OptionalParameter<BalancedAirPressureParameters>
    explicitPb{"explicit balanced air pressure parameters", this};

  // Balanced air pressure parameters from grid
  oops::Parameter<bool> pbFromTrans{"balanced air pressure parameters from grid", false, this};

  oops::Variables mandatoryActiveVars() const {return oops::Variables(
    std::vector<std::string>({
    "air_upward_absolute_vorticity",
    "air_temperature",
    "log_of_air_pressure_at_surface",
    "air_temperature_and_log_of_air_pressure_at_surface"}));}
};

// -----------------------------------------------------------------------------

class BifourierAromeBalanceImpl : public BifourierBalanceImpl {
 public:
  static const std::string classname()
    {return "saber::bifourier::BifourierAromeBalanceImpl";}

  typedef BifourierAromeBalanceImplParameters Parameters_;

  BifourierAromeBalanceImpl(const oops::GeometryData &,
                            const oops::Variables &,
                            const eckit::Configuration &,
                            const Parameters_ &,
                            const oops::FieldSet3D &,
                            const oops::FieldSet3D &);
  virtual ~BifourierAromeBalanceImpl() = default;

  void multiply(oops::FieldSet3D &) const;
  void multiplyAD(oops::FieldSet3D &) const;
  void leftInverseMultiply(oops::FieldSet3D &) const;

  void read();

  void directCalibration(const oops::FieldSets &);

  void iterativeCalibrationUpdate(const oops::FieldSet3D & fset);

  void write() const;

 private:
  // Parameters
  BifourierAromeBalanceImplParameters params_;

  // Number of levels
  size_t nz_;

  // Vorticity to balanced pressure factor
  std::vector<double> fact1_;

  // Private methods

  // AROME inner variables
  oops::Variables aromeInnerVars(const oops::Variables &);

  // Vorticity to balanced pressure
  void vorToPb(oops::FieldSet3D &) const;

  // Vorticity to balanced pressure, adjoint
  void vorToPbAD(oops::FieldSet3D &) const;

  // Vorticity to balanced pressure, left inverse
  void vorToPbLeftInverse(oops::FieldSet3D &) const;

  // Remove balanced pressure
  void removePb(oops::FieldSet3D &) const;

  // Remove balanced pressure, adjoint
  void removePbAD(oops::FieldSet3D &) const;

  // Remove balanced pressure, left inverse
  void removePbLeftInverse(oops::FieldSet3D &) const;

  // Split TPs
  void splitTPs(oops::FieldSet3D &) const;

  // Gather TPs
  void gatherTPs(oops::FieldSet3D &) const;
};

// -----------------------------------------------------------------------------

}  // namespace bifourier
}  // namespace saber
