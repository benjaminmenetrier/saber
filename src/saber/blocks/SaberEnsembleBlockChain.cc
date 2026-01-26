/*
 * (C) Copyright 2023- UCAR
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include "saber/blocks/SaberEnsembleBlockChain.h"

#include "oops/util/RandomField.h"
#include "saber/oops/Utilities.h"

namespace saber {

// -----------------------------------------------------------------------------

void SaberEnsembleBlockChain::multiply(oops::FieldSet4D & fset4d) const {
  oops::Log::trace() << "saber::SaberEnsembleBlockChain::multiply starting" << std::endl;

  if (strategy_ == "univariate") {
    // Outer blocks adjoint multiplication
    if (outerBlockChain_) {
      outerBlockChain_->applyOuterBlocksAD(fset4d);
    }

    // Central block: ensemble covariance
    // Initialization
    const oops::FieldSet4D fset4dInit = oops::copyFieldSet4D(fset4d);
    fset4d.zero();

    for (const auto & scaleData : scaleDataVec_) {
      // Copy initial FieldSet4D
      oops::FieldSet4D fset4dScaleInit = oops::copyFieldSet4D(fset4dInit);

      // Apply interpolator adjoint
      if (scaleData.interpolator()) {
        scaleData.interpolator()->applyOuterBlocksAD(fset4dScaleInit);
      }

      // Create scale FieldSet4D
      oops::FieldSet4D fset4dScale = oops::copyFieldSet4D(fset4dScaleInit);
      fset4dScale.zero();

      for (size_t ie = 0; ie < scaleData.ensemble()->ens_size(); ++ie) {
        // Copy initial FieldSet4D
        oops::FieldSet4D fset4dMem = oops::copyFieldSet4D(fset4dScaleInit);

        if (scaleData.localization()) {
          // With localization

          // First schur product
          for (size_t it = 0; it < fset4dMem.size(); ++it) {
            fset4dMem[it] *= (*scaleData.ensemble())(it, ie);
          }

          // Apply localization
          scaleData.localization()->multiply(fset4dMem);

          // Second schur product
          for (size_t it = 0; it < fset4dMem.size(); ++it) {
            fset4dMem[it] *= (*scaleData.ensemble())(it, ie);
          }

          // Add up member contribution
          fset4dScale += fset4dMem;
        } else {
          // No localization

          // Compute weight
          const double wgt = fset4dScaleInit.dot_product_with(*scaleData.ensemble(), ie, vars_);

          // Copy ensemble member
          fset4dMem.deepCopy(*scaleData.ensemble(), ie);

          // Apply weight
          fset4dMem *= wgt;

          // Add up member contribution
          fset4dScale += fset4dMem;
        }
        // TODO(Algo): Add communication here when the code starts supporting
        // ensemble members distributed across MPI tasks.
      }

      // Normalize result
      const double rk = 1.0/static_cast<double>(scaleData.ensemble()->ens_size()-1);
      fset4dScale *= rk;

      // Apply interpolator
      if (scaleData.interpolator()) {
        scaleData.interpolator()->applyOuterBlocks(fset4dScale);
      }

      // Add up scale contribution
      fset4d += fset4dScale;
    }

    // Outer blocks forward multiplication
    if (outerBlockChain_) {
      outerBlockChain_->applyOuterBlocks(fset4d);
    }
  } else if (strategy_ == "crossed") {
    // Create control vector
    atlas::Field cv = atlas::Field("genericCtlVec",
                                       atlas::array::make_datatype<double>(),
                                       atlas::array::make_shape(ctlVecSize()));

    // Adjoint square-root multiply
    multiplySqrtAD(fset4d, cv, 0);

    // Square-root multiply
    multiplySqrt(cv, fset4d, 0);
  }

  oops::Log::trace() << "saber::SaberEnsembleBlockChain::multiply done" << std::endl;
}

// -----------------------------------------------------------------------------

void SaberEnsembleBlockChain::randomize(oops::FieldSet4D & fset4d) const {
  oops::Log::trace() << "saber::SaberEnsembleBlockChain::randomize starting" << std::endl;

  // Central block: randomization with ensemble covariance
  const auto & scaleData = scaleDataVec_[0];
  fset4d.deepCopy(*scaleData.ensemble(), 0);
  if (scaleData.interpolator()) {
    scaleData.interpolator()->applyOuterBlocks(fset4d);
  }
  fset4d.zero();
  std::unique_ptr<util::NormalDistribution<double>> normalDist;

  if (strategy_ == "univariate") {
    for (const auto & scaleData : scaleDataVec_) {
      // Create scale FieldSet4D
      oops::FieldSet4D fset4dScale(fset4d.times(), fset4d.commTime(), fset4d[0].commGeom());

      // Copy ensemble member
      fset4dScale.deepCopy(*scaleData.ensemble(), 0);
      fset4dScale.zero();

      for (unsigned int ie = 0; ie < scaleData.ensemble()->ens_size(); ++ie) {
        // Create empty FieldSet4D
        // TODO(Benjamin): could be a oops::copyFieldSet4D(fset4dScale);
        oops::FieldSet4D fset4dMem(fset4d.times(), fset4d.commTime(), fset4d[0].commGeom());

        // Copy ensemble member
        fset4dMem.deepCopy(*scaleData.ensemble(), ie);

        if (scaleData.localization()) {
          // With localization

          // Randomize localization
          scaleData.localization()->randomize(fset4dMem);

          // Schur product
          for (size_t it = 0; it < fset4dMem.size(); ++it) {
            fset4dMem[it] *= (*scaleData.ensemble())(it, ie);
          }
        } else {
          // No localization
          if (!normalDist) {
            normalDist = std::make_unique<util::NormalDistribution<double>>(
              scaleData.ensemble()->ens_size(), 0.0, 1.0, seed_);
          }

          // Apply weight
          fset4dMem *= (*normalDist)[ie];
        }

        // Add up member contribution
        fset4dScale += fset4dMem;
      }

      // Normalize result
      const double rk = 1.0/sqrt(static_cast<double>(scaleData.ensemble()->ens_size()-1));
      fset4dScale *= rk;

      // Apply interpolator
      if (scaleData.interpolator()) {
        scaleData.interpolator()->applyOuterBlocks(fset4dScale);
      }

      // Add up scale contribution
      fset4d += fset4dScale;
    }

    // Outer blocks forward multiplication
    if (outerBlockChain_) {
       outerBlockChain_->applyOuterBlocks(fset4d);
    }
  } else if (strategy_ == "crossed") {
    // Create control vector
    atlas::Field cv("genericCtlVec", atlas::array::make_datatype<double>(),
      atlas::array::make_shape(ctlVecSize()));

    // Sizes, sendcounts and displs
    std::vector<int> sendcounts(comm_.size());
    comm_.allGather(static_cast<int>(ctlVecSize()), sendcounts.begin(), sendcounts.end());
    size_t ctlVecSizeGlb = 0;
    for (const auto ctlVecSize : sendcounts) {
      ctlVecSizeGlb += ctlVecSize;
    }
    std::vector<int> displs(comm_.size());
    displs[0] = 0;
    for (size_t jt = 0; jt < comm_.size()-1; ++jt) {
      displs[jt+1] = displs[jt]+sendcounts[jt];
    }

    // Generate global random vector
    std::vector<double> rand_vec_glb;
    if (comm_.rank() == 0) {
      util::NormalDistributionField dist(ctlVecSizeGlb, 0.0, 1.0);
      rand_vec_glb.resize(ctlVecSizeGlb);
      for (size_t jcv = 0; jcv < ctlVecSizeGlb; ++jcv) {
        rand_vec_glb[jcv] = dist[jcv];
      }
    }

    // Scatter random vector
    std::vector<double> rand_vec(ctlVecSize());
    comm_.scatterv(rand_vec_glb.begin(), rand_vec_glb.end(), sendcounts, displs,
      rand_vec.begin(), rand_vec.end(), 0);

    // Fill control vector
    auto cvView = atlas::array::make_view<double, 1>(cv);
    for (size_t jcv = 0; jcv < ctlVecSize(); ++jcv) {
      cvView(jcv) = rand_vec[jcv];
    }

    // Square-root multiply
    multiplySqrt(cv, fset4d, 0);
  }

  oops::Log::trace() << "saber::SaberEnsembleBlockChain::done starting" << std::endl;
}

// -----------------------------------------------------------------------------

void SaberEnsembleBlockChain::multiplySqrt(const atlas::Field & cv,
                                           oops::FieldSet4D & fset4d,
                                           const size_t & offset) const {
  oops::Log::trace() << "saber::SaberEnsembleBlockChain::multiplySqrt starting" << std::endl;

  // Initialization
  fset4d.zero();
  size_t index = offset;

  for (const auto & scaleData : scaleDataVec_) {
    if (strategy_ == "crossed") {
      // Restart index (save control vector)
      index = offset;
    }

    // Create scale FieldSet4D
    oops::FieldSet4D fset4dScale(fset4d.times(), fset4d.commTime(), fset4d[0].commGeom());

    // Copy ensemble member
    fset4dScale.deepCopy(*scaleData.ensemble(), 0);
    fset4dScale.zero();

    // Central block: ensemble covariance square-root
    for (unsigned int ie = 0; ie < scaleData.ensemble()->ens_size(); ++ie) {
      // Create empty FieldSet4D
      oops::FieldSet4D fset4dMem(fset4d.times(), fset4d.commTime(), fset4d[0].commGeom());

      if (scaleData.localization()) {
        // With localization
        scaleData.localization()->multiplySqrt(cv, fset4dMem, index);
        index += scaleData.localization()->ctlVecSize();

        // Schur product
        for (size_t it = 0; it < fset4dMem.size(); ++it) {
          fset4dMem[it] *= (*scaleData.ensemble())(it, ie);
        }
      } else {
        // No localization
        const auto cvView = atlas::array::make_view<double, 1>(cv);
        fset4dMem.deepCopy(*scaleData.ensemble(), ie);

        // Apply weight
        fset4dMem *= cvView(index);
        ++index;
      }

      // Add up member contribution
      fset4dScale += fset4dMem;
    }

    // Normalize result
    const double rk = 1.0/std::sqrt(static_cast<double>(scaleData.ensemble()->ens_size()-1));
    fset4dScale *= rk;

    // Apply interpolator
    if (scaleData.interpolator()) {
      scaleData.interpolator()->applyOuterBlocks(fset4dScale);
    }

    // Add up scale contribution
    fset4d += fset4dScale;
  }

  // Outer blocks forward multiplication
  if (outerBlockChain_) {
    outerBlockChain_->applyOuterBlocks(fset4d);
  }

  oops::Log::trace() << "saber::SaberEnsembleBlockChain::multiplySqrt done" << std::endl;
}

// -----------------------------------------------------------------------------

void SaberEnsembleBlockChain::multiplySqrtAD(const oops::FieldSet4D & fset4d,
                                             atlas::Field & cv,
                                             const size_t & offset) const {
  oops::Log::trace() << "saber::SaberEnsembleBlockChain::multiplySqrtAD starting" << std::endl;

  // Copy input FieldSet
  oops::FieldSet4D fset4dInit = oops::copyFieldSet4D(fset4d);

  // Outer blocks adjoint multiplication
  if (outerBlockChain_) {
    outerBlockChain_->applyOuterBlocksAD(fset4dInit);
  }

  // Initialization
  size_t index = offset;

  // Get control vector view
  auto cvView = atlas::array::make_view<double, 1>(cv);

  // Initialize control vector
  cvView.assign(0.0);

  for (const auto & scaleData : scaleDataVec_) {
    if (strategy_ == "crossed") {
      // Restart index (save control vector)
      index = offset;
    }

    // Copy initial FieldSet4D
    oops::FieldSet4D fset4dScaleInit = oops::copyFieldSet4D(fset4dInit);

    // Apply interpolator adjoint
    if (scaleData.interpolator()) {
      scaleData.interpolator()->applyOuterBlocksAD(fset4dScaleInit);
    }

    // Normalize initial fieldset
    const double rk = 1.0/std::sqrt(static_cast<double>(scaleData.ensemble()->ens_size()-1));
    fset4dScaleInit *= rk;

    // Central block: ensemble covariance square-root adjoint
    if (scaleData.localization()) {
      // Create scale control vector
      atlas::Field cvScale("genericCtlVec", atlas::array::make_datatype<double>(),
        atlas::array::make_shape(scaleData.localization()->ctlVecSize()));

      // Get scale control vector view
      auto cvScaleView = atlas::array::make_view<double, 1>(cvScale);

      for (unsigned int ie = 0; ie < scaleData.ensemble()->ens_size(); ++ie) {
        // Copy initial fieldset
        oops::FieldSet4D fset4dMem = oops::copyFieldSet4D(fset4dScaleInit);

        // First schur product
        for (size_t it = 0; it < fset4dMem.size(); ++it) {
          fset4dMem[it] *= (*scaleData.ensemble())(it, ie);
        }

        // Apply localization square-root adjoint
        scaleData.localization()->multiplySqrtAD(fset4dMem, cvScale, 0);
        for (size_t jcv = 0; jcv < scaleData.localization()->ctlVecSize(); ++jcv) {
          cvView(index+jcv) += cvScaleView(jcv);
        }
        index += scaleData.localization()->ctlVecSize();
      }
    } else {
      for (unsigned int ie = 0; ie < scaleData.ensemble()->ens_size(); ++ie) {
        // Compute weight
        cvView(index) = fset4dScaleInit.dot_product_with(*scaleData.ensemble(), ie, vars_);
        ++index;
      }
    }
  }

  oops::Log::trace() << "saber::SaberEnsembleBlockChain::multiplySqrtAD done" << std::endl;
}

// -----------------------------------------------------------------------------

}  // namespace saber
