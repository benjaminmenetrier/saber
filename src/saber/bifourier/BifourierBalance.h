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

#include "saber/bifourier/BifourierBalanceImpl.h"
#include "saber/blocks/SaberOuterBlockBase.h"

namespace saber {
namespace bifourier {

// -----------------------------------------------------------------------------

class BifourierBalance : public SaberOuterBlockBase {
 public:
  static const std::string classname()
    {return "saber::bifourier::BifourierBalance";}

  typedef BifourierBalanceImplParameters Parameters_;

  BifourierBalance(const oops::GeometryData &,
                   const oops::Variables &,
                   const eckit::Configuration &,
                   const Parameters_ &,
                   const oops::FieldSet3D &,
                   const oops::FieldSet3D &);
  virtual ~BifourierBalance();

  const oops::GeometryData & innerGeometryData() const override
    {return balance_->innerGeometryData();}
  const oops::Variables & innerVars() const override
    {return balance_->innerVars();}

  void multiply(oops::FieldSet3D & fset) const override
    {balance_->multiply(fset);}
  void multiplyAD(oops::FieldSet3D & fset) const override
    {balance_->multiplyAD(fset);}
  void leftInverseMultiply(oops::FieldSet3D & fset) const override
    {balance_->leftInverseMultiply(fset);}

  void read() override
    {balance_->read();}

  void directCalibration(const oops::FieldSets & fsetEns) override
    {balance_->directCalibration(fsetEns);}

  void iterativeCalibrationInit() override
    {balance_->iterativeCalibrationInit();}
  void iterativeCalibrationUpdate(const oops::FieldSet3D & fset) override
    {balance_->iterativeCalibrationUpdate(fset);}
  void iterativeCalibrationFinal() override
    {balance_->iterativeCalibrationFinal();}

  void write() const override
    {balance_->write();}

 private:
   // Balance implementation
  std::unique_ptr<BifourierBalanceImpl> balance_;

  // Print
  void print(std::ostream & os) const override
    {balance_->print(os);}
};

// -----------------------------------------------------------------------------

}  // namespace bifourier
}  // namespace saber
