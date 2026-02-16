/*
 * (C) Copyright 2025 Meteorologisk Institutt
 *
 */

#include "saber/bifourier/BifourierSpectralConverter.h"

#include "atlas/field.h"
#include "atlas/functionspace.h"
#include "atlas/grid.h"

#include "oops/util/FloatCompare.h"
#include "oops/util/FunctionSpaceHelpers.h"

using atlas::array::make_view;

namespace saber {
namespace bifourier {

// -----------------------------------------------------------------------------

static SaberOuterBlockMaker<BifourierSpectralConverter>
  makerBifourierSpectralConverter_("BifourierSpectralConverter");

// -----------------------------------------------------------------------------

BifourierSpectralConverter::BifourierSpectralConverter(const oops::GeometryData & outerGeometryData,
                                                       const oops::Variables & outerVars,
                                                       const eckit::Configuration & covarConfig,
                                                       const Parameters_ & params,
                                                       const oops::FieldSet3D & xb,
                                                       const oops::FieldSet3D & fg)
  : SaberOuterBlockBase(params, xb.validTime(), outerGeometryData, outerVars),
    comm_(outerGeometryData.comm()),
    myrank_(comm_.rank()),
    innerVars_(outerVars)
{
  oops::Log::trace() << classname() << "::BifourierSpectralConverter starting" << std::endl;

  // Inner geometry data
  atlas::Grid grid;
  atlas::grid::Partitioner partitioner;
  atlas::Mesh mesh;
  atlas::FunctionSpace functionSpace;
  atlas::FieldSet fset;
  util::setupFunctionSpace(comm_,
                           params.geomConf.value(),
                           grid,
                           partitioner,
                           mesh,
                           functionSpace,
                           fset);
  innerGpGeometryData_ = std::make_unique<oops::GeometryData>(functionSpace, fset,
    outerGeometryData.levelsAreTopDown(), comm_);

  // Create inner spectral transform
  innerTrans_ = transStore_.setupTransform(*innerGpGeometryData_, innerVars_,
    params.transform.value());

  // Create inner spectral GeometryData
  innerGeometryData_ = std::make_unique<oops::GeometryData>(innerTrans_->spFspace(),
    outerGeometryData.fieldSet(), outerGeometryData.levelsAreTopDown(), comm_);

  // Retrieve outer spectral transform
  outerTrans_ = transStore_.retrieveTransform(outerGeometryData, outerVars);

  // Check domain size (assuming that the projection is the same)
  ASSERT(oops::is_close_relative(static_cast<double>(innerTrans_->nx()-1)*innerTrans_->dx(),
    static_cast<double>(outerTrans_->nx()-1)*outerTrans_->dx(), 1.0e-12));
  ASSERT(oops::is_close_relative(static_cast<double>(innerTrans_->ny()-1)*innerTrans_->dy(),
    static_cast<double>(outerTrans_->ny()-1)*outerTrans_->dy(), 1.0e-12));

  // Prepare spectral converter mapping
  // TODO(Benjamin): optimize the double loop
  std::vector<int> outerToInnerJsGlb(outerTrans_->nsGlb(), -1);
  const auto & innerSpVec = innerTrans_->spVec();
  const auto & outerSpVec = outerTrans_->spVec();
  for (size_t outerJsGlb = 0; outerJsGlb < outerTrans_->nsGlb(); ++outerJsGlb) {
    for (size_t innerJsGlb = 0; innerJsGlb < innerTrans_->nsGlb(); ++innerJsGlb) {
      if ((innerSpVec[innerJsGlb].jk == outerSpVec[outerJsGlb].jk) &&
        (innerSpVec[innerJsGlb].jl == outerSpVec[outerJsGlb].jl) &&
        (innerSpVec[innerJsGlb].jq == outerSpVec[outerJsGlb].jq)) {
        ASSERT(outerToInnerJsGlb[outerJsGlb] == -1);
        outerToInnerJsGlb[outerJsGlb] = innerJsGlb;
      }
    }
  }

  // Order outer indices by task
  std::vector<int> outerTask;
  for (size_t outerJsGlb = 0; outerJsGlb < outerTrans_->nsGlb(); ++outerJsGlb) {
    outerTask.push_back(outerTrans_->sGlbToTask(outerJsGlb));
  }
  std::vector<size_t> outerOrder(outerTrans_->nsGlb());
  std::iota(outerOrder.begin(), outerOrder.end(), 0);
  std::stable_sort(outerOrder.begin(), outerOrder.end(),
    [&](size_t i, size_t j){return outerTask[i] < outerTask[j];});

  // Prepare spectral converter communications
  std::vector<int> sendOuterIndexGlb;
  sendCounts_.resize(comm_.size());
  recvCounts_.resize(comm_.size());
  std::fill(sendCounts_.begin(), sendCounts_.end(), 0);
  std::fill(recvCounts_.begin(), recvCounts_.end(), 0);
  for (size_t outerJsGlb = 0; outerJsGlb < outerTrans_->nsGlb(); ++outerJsGlb) {
    // Get ordered index
    const size_t orderedOuterJsGlb = outerOrder[outerJsGlb];

    if (outerToInnerJsGlb[orderedOuterJsGlb] >= 0) {
      // Get inner jsGlb
      const size_t innerJsGlb = outerToInnerJsGlb[orderedOuterJsGlb];

      // Get inner and outer tasks
      const size_t innerTask = innerTrans_->sGlbToTask(innerJsGlb);
      const size_t outerTask = outerTrans_->sGlbToTask(orderedOuterJsGlb);

      if (innerTask == myrank_) {
        // Save coefficients to send
        const size_t innerJs = innerTrans_->sGlbToS(innerJsGlb);
        sendIndex_.push_back(innerJs);
        sendOuterIndexGlb.push_back(orderedOuterJsGlb);

        // Update send counts
        ++sendCounts_[outerTask];
      }

      if (outerTask == myrank_) {
        // Update receive counts
        ++recvCounts_[innerTask];
      }
    }
  }

  // Sizes
  sendSize_ = 0;
  for (const auto & n : sendCounts_) sendSize_ += n;
  recvSize_ = 0;
  for (const auto & n : recvCounts_) recvSize_ += n;

  // Displacements
  sendDispls_.resize(comm_.size());
  recvDispls_.resize(comm_.size());
  for (size_t jt = 0; jt < comm_.size(); ++jt) {
    sendDispls_[jt] = static_cast<int>(jt ? sendDispls_[jt-1] + sendCounts_[jt-1] : 0);
    recvDispls_[jt] = static_cast<int>(jt ? recvDispls_[jt-1] + recvCounts_[jt-1] : 0);
  }

  // Communicate receive global index
  std::vector<int> recvOuterIndexGlb(recvSize_);
  comm_.allToAllv(sendOuterIndexGlb.data(), sendCounts_.data(), sendDispls_.data(),
    recvOuterIndexGlb.data(), recvCounts_.data(), recvDispls_.data());

  // Retrieve local receive index
  recvIndex_.resize(recvSize_);
  for (size_t jj = 0; jj < recvSize_; ++jj) {
    recvIndex_[jj] = outerTrans_->sGlbToS(recvOuterIndexGlb[jj]);
  }

  // Multiply by nvz
  for (size_t jt = 0; jt < comm_.size(); ++jt) {
    sendCounts_[jt] *= innerTrans_->nvz();
    recvCounts_[jt] *= outerTrans_->nvz();
    sendDispls_[jt] *= innerTrans_->nvz();
    recvDispls_[jt] *= outerTrans_->nvz();
  }

  oops::Log::trace() << classname() << "::BifourierSpectralConverter done" << std::endl;
}

// -----------------------------------------------------------------------------

void BifourierSpectralConverter::multiply(oops::FieldSet3D & fset) const {
  oops::Log::trace() << classname() << "::multiply starting" << std::endl;

  // Create buffers
  std::vector<double> sendVec(sendSize_*innerTrans_->nvz(), 0.0);
  std::vector<double> recvVec(recvSize_*outerTrans_->nvz());

  // Reserialize from spectral FieldSet
  size_t zOffset = 0;
  for (const auto & var : outerVars_) {
    // Get number of vertical levels
    const size_t nz = var.getLevels();

    // Check field
    const auto spField = fset[var.name()];
    ASSERT(spField.shape(0) == static_cast<int>(innerTrans_->ns()));
    ASSERT(spField.shape(1) == static_cast<int>(nz));

    // Get field view
    const auto spView = make_view<double, 2>(spField);

    for (size_t jj = 0; jj < sendSize_; ++jj) {
      for (size_t jz = 0; jz < nz; ++jz) {
        // Total level index
        const size_t jvz = zOffset + jz;

        // Spectral index
        const size_t js = sendIndex_[jj];

        // Communication vector index
        const size_t jjv = jj*innerTrans_->nvz()+jvz;

        // Copy data
        sendVec[jjv] = spView(js, jz);
      }
    }

    // Update total number of levels
    zOffset += nz;
  }

  // Communication
  comm_.allToAllv(sendVec.data(), sendCounts_.data(), sendDispls_.data(),
    recvVec.data(), recvCounts_.data(), recvDispls_.data());

  // Remove outer variables
  util::removeFieldsFromFieldSet(fset.fieldSet(), outerVars_.variables());

  // Deserialize into grid-point FieldSet
  zOffset = 0;
  for (const auto & var : outerVars_) {
    // Get number of vertical levels
    const size_t nz = var.getLevels();

    // Create field
    atlas::Field spField = outerGeometryData_.functionSpace().createField<double>(
      atlas::option::name(var.name()) | atlas::option::levels(nz));
    fset.add(spField);

    // Get field view
    auto spView = make_view<double, 2>(spField);

    // Set to zero
    spView.assign(0.0);

    for (size_t jj = 0; jj < recvSize_; ++jj) {
      for (size_t jz = 0; jz < nz; ++jz) {
        // Total level index
        const size_t jvz = zOffset + jz;

        // Spectral index
        const size_t js = recvIndex_[jj];

        // Communication vector index
        const size_t jjv = jj*outerTrans_->nvz()+jvz;

        // Copy data
        spView(js, jz) = recvVec[jjv];
      }
    }

    // Update total number of levels
    zOffset += nz;
  }

  oops::Log::trace() << classname() << "::multiply done" << std::endl;
}

// -----------------------------------------------------------------------------

void BifourierSpectralConverter::multiplyAD(oops::FieldSet3D & fset) const {
  oops::Log::trace() << classname() << "::multiplyAD starting" << std::endl;

  // Create buffers
  std::vector<double> recvVec(recvSize_*outerTrans_->nvz());
  std::vector<double> sendVec(sendSize_*innerTrans_->nvz());

  // Reserialize from spectral FieldSet
  size_t zOffset = 0;
  for (const auto & var : outerVars_) {
    // Get number of vertical levels
    const size_t nz = var.getLevels();

    // Check field
    const auto spField = fset[var.name()];
    ASSERT(spField.shape(0) == static_cast<int>(outerTrans_->ns()));
    ASSERT(spField.shape(1) == static_cast<int>(nz));

    // Get field view
    const auto spView = make_view<double, 2>(spField);

    for (size_t jj = 0; jj < recvSize_; ++jj) {
      for (size_t jz = 0; jz < nz; ++jz) {
        // Total level index
        const size_t jvz = zOffset + jz;

        // Spectral index
        const size_t js = recvIndex_[jj];

        // Communication vector index
        const size_t jjv = jj*outerTrans_->nvz()+jvz;

        // Copy data
        recvVec[jjv] = spView(js, jz);
      }
    }

    // Update total number of levels
    zOffset += nz;
  }

  // Communication
  comm_.allToAllv(recvVec.data(), recvCounts_.data(), recvDispls_.data(),
    sendVec.data(), sendCounts_.data(), sendDispls_.data());

  // Remove inner variables
  util::removeFieldsFromFieldSet(fset.fieldSet(), innerVars_.variables());

  // Deserialize into grid-point FieldSet
  zOffset = 0;
  for (const auto & var : outerVars_) {
    // Get number of vertical levels
    const size_t nz = var.getLevels();

    // Create field
    atlas::Field spField = innerGeometryData_->functionSpace().createField<double>(
      atlas::option::name(var.name()) | atlas::option::levels(nz));
    fset.add(spField);

    // Get field view
    auto spView = make_view<double, 2>(spField);

    // Set to zero
    spView.assign(0.0);

    for (size_t jj = 0; jj < sendSize_; ++jj) {
      for (size_t jz = 0; jz < nz; ++jz) {
        // Total level index
        const size_t jvz = zOffset + jz;

        // Spectral index
        const size_t js = sendIndex_[jj];

        // Communication vector index
        const size_t jjv = jj*innerTrans_->nvz()+jvz;

        // Copy data
        spView(js, jz) = sendVec[jjv];
      }
    }

    // Update total number of levels
    zOffset += nz;
  }

  oops::Log::trace() << classname() << "::multiplyAD done" << std::endl;
}

// -----------------------------------------------------------------------------

void BifourierSpectralConverter::leftInverseMultiply(oops::FieldSet3D & fset) const {
  oops::Log::trace() << classname() << "::leftInverseMultiply starting" << std::endl;

  ASSERT(false);

  oops::Log::trace() << classname() << "::leftInverseMultiply done" << std::endl;
}

// -----------------------------------------------------------------------------

void BifourierSpectralConverter::print(std::ostream & os) const {
  os << classname();
}

// -----------------------------------------------------------------------------

}  // namespace bifourier
}  // namespace saber
