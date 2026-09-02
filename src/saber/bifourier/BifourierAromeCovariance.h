/*
 * (C) Copyright 2025 Meteorologisk Institutt
 *
 */

#pragma once

#include <string>

#include "saber/bifourier/BifourierCovarianceImpl.h"
#include "saber/blocks/SaberCentralBlockBase.h"

namespace saber {
namespace bifourier {

// -----------------------------------------------------------------------------

class BifourierAromeCovarianceReadParameters : public BifourierCovarianceImplReadParameters {
  OOPS_CONCRETE_PARAMETERS(BifourierAromeCovarianceReadParameters,
    BifourierCovarianceImplReadParameters)

 public:
  // Input file format ("netcdf", "arome legacy binary" or "arome legacy netcdf")
  oops::Parameter<std::string> inputFileFormat{"input file format", "netcdf", this};
};

// -----------------------------------------------------------------------------

class BifourierAromeCovarianceWriteParameters : public BifourierCovarianceImplWriteParameters {
  OOPS_CONCRETE_PARAMETERS(BifourierAromeCovarianceWriteParameters,
    BifourierCovarianceImplWriteParameters)

 public:
  // Output file format ("netcdf", "arome legacy binary" or "arome legacy netcdf")
  oops::Parameter<std::string> outputFileFormat{"output file format", "netcdf", this};
};

// -----------------------------------------------------------------------------

class BifourierAromeCovarianceParameters : public BifourierCovarianceImplParameters {
  OOPS_CONCRETE_PARAMETERS(BifourierAromeCovarianceParameters, BifourierCovarianceImplParameters)

 public:
  // Read parameters
  oops::OptionalParameter<BifourierAromeCovarianceReadParameters> read{"read", this};

  // Write parameters
  oops::OptionalParameter<BifourierAromeCovarianceWriteParameters> write{"write", this};

  // REDNMC factor
  oops::Parameter<double> rednmc{"rednmc", std::sqrt(0.5), this};

  oops::Variables mandatoryActiveVars() const override
    {return oops::Variables();}
};


// -----------------------------------------------------------------------------

class BifourierAromeCovariance : public SaberCentralBlockBase {
 public:
  static const std::string classname()
    {return "saber::bifourier::BifourierAromeCovariance";}

  typedef BifourierAromeCovarianceParameters Parameters_;

  BifourierAromeCovariance(const oops::GeometryData &,
                      const oops::Variables &,
                      const eckit::Configuration &,
                      const Parameters_ &,
                      const oops::FieldSet3D &,
                      const oops::FieldSet3D &);
  virtual ~BifourierAromeCovariance();

  size_t ctlVecSize() const override
    {return covar_->ctlVecSize();}
  void randomCtlVec(atlas::Field & cv,
                    const size_t & offset) const override
    {covar_->randomCtlVec(cv, offset);}
  void multiplySqrt(const atlas::Field & cv,
                    oops::FieldSet3D & fset,
                    const size_t & offset) const override
    {covar_->multiplySqrt(cv, fset, offset);}
  void multiplySqrtAD(const oops::FieldSet3D & fset,
                      atlas::Field & cv,
                      const size_t & offset) const override
    {covar_->multiplySqrtAD(fset, cv, offset);}

  void read() override;

  void directCalibration(const oops::FieldSets & fsetEns) override
    {covar_->directCalibration(fsetEns);}

  void iterativeCalibrationInit() override
    {covar_->iterativeCalibrationInit();}
  void iterativeCalibrationUpdate(const oops::FieldSet3D & fset) override
    {covar_->iterativeCalibrationUpdate(fset);}
  void iterativeCalibrationFinal() override
    {covar_->iterativeCalibrationFinal();}

  void write() const override;

 private:
  // Parameters
  Parameters_ params_;

  // Covariance implementation
  std::unique_ptr<BifourierCovarianceImpl> covar_;

  // Private methods

  // Define AROME weights
  double aromeWeight(const size_t &) const;

  // Print
  void print(std::ostream & os) const override
    {covar_->print(os);}
};

// -----------------------------------------------------------------------------

}  // namespace bifourier
}  // namespace saber
