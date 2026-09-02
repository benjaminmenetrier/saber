/*
 * (C) Copyright 2026 Meteorologisk Institutt
 *
 */

#pragma once

#include <memory>
#include <string>

#include "atlas/field.h"

#include "oops/base/FieldSets.h"
#include "oops/base/GeometryData.h"
#include "oops/base/Variable.h"

#include "saber/bifourier/BifourierAromeBalanceImpl.h"
#include "saber/bifourier/BifourierAromeCovarianceImpl.h"
#include "saber/bifourier/BifourierCombined.h"
#include "saber/blocks/SaberCentralBlockBase.h"

namespace saber {
namespace bifourier {

// -----------------------------------------------------------------------------

class BifourierCombinedParameters : public SaberBlockParametersBase {
  OOPS_CONCRETE_PARAMETERS(BifourierCombinedParameters, SaberBlockParametersBase)

 public:
  // Covariance parameters
  oops::RequiredParameter<BifourierAromeCovarianceImplParameters> covariance{"covariance", this};

  // Balance parameters
  oops::RequiredParameter<BifourierAromeBalanceImplParameters> balance{"balance", this};

  oops::Variables mandatoryActiveVars() const
    {return oops::Variables();}
};

// -----------------------------------------------------------------------------

class BifourierCombined : public SaberCentralBlockBase {
 public:
  static const std::string classname()
    {return "saber::bifourier::BifourierCombined";}

  typedef BifourierCombinedParameters Parameters_;

  BifourierCombined(const oops::GeometryData &,
                    const oops::Variables &,
                    const eckit::Configuration &,
                    const Parameters_ &,
                    const oops::FieldSet3D &,
                    const oops::FieldSet3D &);
  virtual ~BifourierCombined();

  size_t ctlVecSize() const override
    {return covar_->ctlVecSize();}
  void randomCtlVec(atlas::Field & cv,
                    const size_t & offset) const override
    {covar_->randomCtlVec(cv, offset);}
  void multiplySqrt(const atlas::Field & cv,
                    oops::FieldSet3D & fset,
                    const size_t & offset) const override
    {covar_->multiplySqrt(cv, fset, offset);
     balance_->multiply(fset);}
  void multiplySqrtAD(const oops::FieldSet3D & fset,
                      atlas::Field & cv,
                      const size_t & offset) const override
    {oops::FieldSet3D fsetTmp(fset);
     balance_->multiplyAD(fsetTmp);
     covar_->multiplySqrtAD(fsetTmp, cv, offset);}

  void read() override;

  void write() const override
    {balance_->write();
     covar_->write();}

 protected:
   // Balance implementation
  std::unique_ptr<BifourierAromeBalanceImpl> balance_;

  // Covariance implementation
  std::unique_ptr<BifourierAromeCovarianceImpl> covar_;

  // Diagnostics data
  atlas::FieldSet diagsData_;

  // Print
  void print(std::ostream & os) const override
    {covar_->print(os);}
};

// -----------------------------------------------------------------------------

}  // namespace bifourier
}  // namespace saber
