!----------------------------------------------------------------------
! Module: type_ens
!> Ensemble derived type
! Author: Benjamin Menetrier
! Licensing: this code is distributed under the CeCILL-C license
! Copyright © 2015-... UCAR, CERFACS, METEO-FRANCE and IRIT
!----------------------------------------------------------------------
module type_ens

use atlas_module, only: atlas_fieldset
use fckit_mpi_module, only: fckit_mpi_sum,fckit_mpi_max
use netcdf
use tools_const, only: deg2rad,rad2deg,req
use tools_func, only: sphere_dist,lonlat2xyz,xyz2lonlat
use tools_kinds, only: kind_real,nc_kind_real
use tools_qsort, only: qsort
use type_fieldset, only: fieldset_type
use type_geom, only: geom_type
use type_io, only: io_type
use type_linop, only: linop_type
use type_mpl, only: mpl_type
use type_nam, only: nam_type
use type_rng, only: rng_type

implicit none

! Ensemble derived type
type ens_type
   ! Attributes
   integer :: ne                              !< Ensemble size
   integer :: nsub                            !< Number of sub-ensembles

   ! Data
   type(fieldset_type),allocatable :: mem(:)  !< Members
   type(fieldset_type),allocatable :: mean(:) !< Ensemble mean
   type(fieldset_type) :: m2                  !< Variance
   type(fieldset_type) :: m4                  !< Fourth-order centered moment
contains
   procedure :: set_att => ens_set_att
   procedure :: alloc => ens_alloc
   procedure :: dealloc => ens_dealloc
   procedure :: copy => ens_copy
   procedure :: compute_mean => ens_compute_mean
   procedure :: compute_moments => ens_compute_moments
   procedure :: normalize => ens_normalize
   procedure :: ens_get_c0_single
   procedure :: ens_get_c0_all
   generic :: get_c0 => ens_get_c0_single,ens_get_c0_all
   procedure :: ens_set_c0_single
   procedure :: ens_set_c0_all
   generic :: set_c0 => ens_set_c0_single,ens_set_c0_all
   procedure :: apply_bens => ens_apply_bens
   procedure :: apply_bens_dirac => ens_apply_bens_dirac
   procedure :: normality => ens_normality
end type ens_type

private
public :: ens_type

contains

!----------------------------------------------------------------------
! Subroutine: ens_set_att
!> Set attributes
!----------------------------------------------------------------------
subroutine ens_set_att(ens,ne,nsub)

implicit none

! Passed variables
class(ens_type),intent(inout) :: ens !< Ensemble
integer,intent(in) :: ne             !< Ensemble size
integer,intent(in) :: nsub           !< Number of sub-ensembles

! Copy attributes
ens%ne = ne
ens%nsub = nsub

end subroutine ens_set_att

!----------------------------------------------------------------------
! Subroutine: ens_alloc
!> Allocation
!----------------------------------------------------------------------
subroutine ens_alloc(ens,ne,nsub)

implicit none

! Passed variables
class(ens_type),intent(inout) :: ens !< Ensemble
integer,intent(in) :: ne             !< Ensemble size
integer,intent(in) :: nsub           !< Number of sub-ensembles

! Copy attributes
call ens%set_att(ne,nsub)

! Allocation
if (ne>0) then
   allocate(ens%mem(ne))
   allocate(ens%mean(nsub))
end if

end subroutine ens_alloc

!----------------------------------------------------------------------
! Subroutine: ens_dealloc
!> Release memory
!----------------------------------------------------------------------
subroutine ens_dealloc(ens)

implicit none

! Passed variables
class(ens_type),intent(inout) :: ens !< Ensemble

! Local variables
integer :: ie,isub

! Release memory
if (allocated(ens%mem)) then
   do ie=1,ens%ne
      call ens%mem(ie)%final()
   end do
   deallocate(ens%mem)
end if
if (allocated(ens%mean)) then
   do isub=1,ens%nsub
      call ens%mean(isub)%final()
   end do
   deallocate(ens%mean)
end if
call ens%m2%final()
call ens%m4%final()

end subroutine ens_dealloc

!----------------------------------------------------------------------
! Subroutine: ens_copy
!> Copy
!----------------------------------------------------------------------
subroutine ens_copy(ens_out,mpl,nam,geom,ens_in)

implicit none

! Passed variables
class(ens_type),intent(inout) :: ens_out !< Output ensemble
type(mpl_type),intent(inout) :: mpl      !< MPI data
type(nam_type),intent(in) :: nam         !< Namelist
type(geom_type),intent(in) :: geom       !< Geometry
type(ens_type),intent(in) :: ens_in      !< Input ensemble

! Local variables
integer :: ie,isub

! Copy data
if (allocated(ens_in%mem)) then
   do ie=1,ens_in%ne
      if (.not.ens_in%mem(ie)%is_null()) then
         call ens_out%mem(ie)%init(mpl,geom%nmga,geom%nl0,geom%gmask_mga,nam%variables(1:nam%nv),nam%lev2d,geom%afunctionspace_mg)
         call ens_out%mem(ie)%copy_fields(ens_in%mem(ie))
      end if
   end do
end if
if (allocated(ens_in%mean)) then
   do isub=1,ens_in%nsub
      if (.not.ens_in%mean(isub)%is_null()) then
         ens_out%mean(isub) = atlas_fieldset()
         call ens_out%mean(isub)%init(mpl,geom%nmga,geom%nl0,geom%gmask_mga,nam%variables(1:nam%nv),nam%lev2d, &
 & geom%afunctionspace_mg)
         call ens_out%mean(isub)%copy_fields(ens_in%mean(isub))
      end if
   end do
end if
if (.not.ens_in%m2%is_null()) then
   call ens_out%m2%init(mpl,geom%nmga,geom%nl0,geom%gmask_mga,nam%variables(1:nam%nv),nam%lev2d,geom%afunctionspace_mg)
   call ens_out%m2%copy_fields(ens_in%m2)
end if
if (.not.ens_in%m4%is_null()) then
   call ens_out%m4%init(mpl,geom%nmga,geom%nl0,geom%gmask_mga,nam%variables(1:nam%nv),nam%lev2d,geom%afunctionspace_mg)
   call ens_out%m4%copy_fields(ens_in%m4)
end if

end subroutine ens_copy

!----------------------------------------------------------------------
! Subroutine: ens_compute_mean
!> Compute ensemble mean(s)
!----------------------------------------------------------------------
subroutine ens_compute_mean(ens,mpl,nam,geom)

implicit none

! Passed variables
class(ens_type),intent(inout) :: ens !< Ensemble
type(mpl_type),intent(inout) :: mpl  !< MPI data
type(nam_type),intent(in) :: nam     !< Namelist
type(geom_type),intent(in) :: geom   !< Geometry

! Local variables
integer :: isub,ie_sub,ie

do isub=1,ens%nsub
   ! Initialization
   call ens%mean(isub)%init(mpl,geom%nmga,geom%nl0,geom%gmask_mga,nam%variables(1:nam%nv),nam%lev2d,geom%afunctionspace_mg)

   ! Set fields at zero
   call ens%mean(isub)%zero_fields

   ! Compute mean
   do ie_sub=1,ens%ne/ens%nsub
      ie = ie_sub+(isub-1)*ens%ne/ens%nsub
      call ens%mean(isub)%add_fields(ens%mem(ie))
   end do
   call ens%mean(isub)%mult_fields(1.0/real(ens%ne/ens%nsub,kind_real))
end do

end subroutine ens_compute_mean

!----------------------------------------------------------------------
! Subroutine: ens_compute_moments
!> Compute 2nd- and 4th-order centered moments
!----------------------------------------------------------------------
subroutine ens_compute_moments(ens,mpl,nam,geom)

implicit none

! Passed variables
class(ens_type),intent(inout) :: ens !< Ensemble
type(mpl_type),intent(inout) :: mpl  !< MPI data
type(nam_type),intent(in) :: nam     !< Namelist
type(geom_type),intent(in) :: geom   !< Geometry

! Local variables
integer :: isub,ie_sub,ie
type(fieldset_type) :: pert

! Initialization
call pert%init(mpl,geom%nmga,geom%nl0,geom%gmask_mga,nam%variables(1:nam%nv),nam%lev2d,geom%afunctionspace_mg)
call ens%m2%init(mpl,geom%nmga,geom%nl0,geom%gmask_mga,nam%variables(1:nam%nv),nam%lev2d,geom%afunctionspace_mg)
call ens%m4%init(mpl,geom%nmga,geom%nl0,geom%gmask_mga,nam%variables(1:nam%nv),nam%lev2d,geom%afunctionspace_mg)

! Set fields at zero
call ens%m2%zero_fields
call ens%m4%zero_fields

do isub=1,ens%nsub
   do ie_sub=1,ens%ne/ens%nsub
      ! Compute perturbation
      ie = ie_sub+(isub-1)*ens%ne/ens%nsub
      call pert%copy_fields(ens%mem(ie))
      call pert%sub_fields(ens%mean(isub))

      ! Square
      call pert%square_fields
      call ens%m2%add_fields(pert)

      ! Square again
      call pert%square_fields
      call ens%m4%add_fields(pert)
   end do
end do

! Normalize
call ens%m2%mult_fields(1.0/real(ens%ne-ens%nsub,kind_real))
call ens%m4%mult_fields(1.0/real(ens%ne,kind_real))

end subroutine ens_compute_moments

!----------------------------------------------------------------------
! Subroutine: ens_normalize
!> Normalize ensemble members as perturbations (zero mean) with unit variance
!----------------------------------------------------------------------
subroutine ens_normalize(ens,mpl,nam,geom)

implicit none

! Passed variables
class(ens_type),intent(inout) :: ens !< Ensemble
type(mpl_type),intent(inout) :: mpl  !< MPI data
type(nam_type),intent(in) :: nam     !< Namelist
type(geom_type),intent(in) :: geom   !< Geometry

! Local variables
integer :: isub,ie_sub,ie
type(fieldset_type) :: std

! Compute ensemble mean
call ens%compute_mean(mpl,nam,geom)

! Remove mean
do isub=1,ens%nsub
   do ie_sub=1,ens%ne/ens%nsub
      ie = ie_sub+(isub-1)*ens%ne/ens%nsub
      call ens%mem(ie)%sub_fields(ens%mean(isub))
   end do
   call ens%mean(isub)%zero_fields
end do

! Compute moments
call ens%compute_moments(mpl,nam,geom)

! Compute standard deviation
call std%init(mpl,geom%nmga,geom%nl0,geom%gmask_mga,nam%variables(1:nam%nv),nam%lev2d,geom%afunctionspace_mg)
call std%copy_fields(ens%m2)
call std%sqrt_fields

! Normalize members
do isub=1,ens%nsub
   do ie_sub=1,ens%ne/ens%nsub
      ie = ie_sub+(isub-1)*ens%ne/ens%nsub
      call ens%mem(ie)%div_fields(std)
   end do
end do

! Recompute moments
call ens%compute_moments(mpl,nam,geom)

end subroutine ens_normalize

!----------------------------------------------------------------------
! Subroutine: ens_get_c0_single
!> Get ensemble field on subset Sc0, single field
!----------------------------------------------------------------------
subroutine ens_get_c0_single(ens,mpl,iv,geom,fieldtype,i,fld_c0a)

implicit none

! Passed variables
class(ens_type),intent(in) :: ens                          !< Ensemble
type(mpl_type),intent(inout) :: mpl                        !< MPI data
integer,intent(in) :: iv                                   !< Variable index
type(geom_type),intent(in) :: geom                         !< Geometry
character(len=*),intent(in) :: fieldtype                   !< Field type ('member', 'pert', 'mean', 'm2' or 'm4')
integer,intent(in) :: i                                    !< Index (member or subset)
real(kind_real),intent(out) :: fld_c0a(geom%nc0a,geom%nl0) !< Field on Sc0 subset, halo A

! Local variables
integer :: isub
real(kind_real),allocatable :: fld_mga(:,:),mean(:,:)
character(len=1024),parameter :: subr = 'ens_get_c0_single'

! Allocation
if (.not.geom%same_grid) allocate(fld_mga(geom%nmga,geom%nl0))

select case (trim(fieldtype))
case ('member')
   ! Fieldset to Fortran array
   if (geom%same_grid) then
      call ens%mem(i)%to_array(mpl,iv,fld_c0a)
   else
      call ens%mem(i)%to_array(mpl,iv,fld_mga)
   end if
case ('pert')
   ! Allocation
   allocate(mean(geom%nmga,geom%nl0))

   ! Get sub-ensemble
   isub = (i-1)/(ens%ne/ens%nsub)+1

   ! Fieldset to Fortran array
   if (geom%same_grid) then
      call ens%mem(i)%to_array(mpl,iv,fld_c0a)
   else
      call ens%mem(i)%to_array(mpl,iv,fld_mga)
   end if
   call ens%mean(isub)%to_array(mpl,iv,mean)

   ! Member to perturbation
   if (geom%same_grid) then
      fld_c0a = fld_c0a-mean
   else
      fld_mga = fld_mga-mean
   end if

   ! Release memory
   deallocate(mean)
case ('mean')
   ! Fieldset to Fortran array
   if (geom%same_grid) then
      call ens%mean(i)%to_array(mpl,iv,fld_c0a)
   else
      call ens%mean(i)%to_array(mpl,iv,fld_mga)
   end if
case ('m2')
   ! Fieldset to Fortran array
   if (geom%same_grid) then
      call ens%m2%to_array(mpl,iv,fld_c0a)
   else
      call ens%m2%to_array(mpl,iv,fld_mga)
   end if
case ('m4')
   ! Fieldset to Fortran array
   if (geom%same_grid) then
      call ens%m4%to_array(mpl,iv,fld_c0a)
   else
      call ens%m4%to_array(mpl,iv,fld_mga)
   end if
case default
   call mpl%abort(subr,'wrong field type')
end select

if (.not.geom%same_grid) then
   ! Model grid to subset Sc0
   call geom%copy_mga_to_c0a(mpl,fld_mga,fld_c0a)

   ! Release memory
   deallocate(fld_mga)
end if

end subroutine ens_get_c0_single

!----------------------------------------------------------------------
! Subroutine: ens_get_c0_all
!> Get ensemble field on subset Sc0, all field
!----------------------------------------------------------------------
subroutine ens_get_c0_all(ens,mpl,nam,geom,fieldtype,i,fld_c0a)

implicit none

! Passed variables
class(ens_type),intent(in) :: ens                                 !< Ensemble
type(mpl_type),intent(inout) :: mpl                               !< MPI data
type(nam_type),intent(in) :: nam                                  !< Namelist
type(geom_type),intent(in) :: geom                                !< Geometry
character(len=*),intent(in) :: fieldtype                          !< Field type ('member', 'pert', 'mean', 'm2' or 'm4')
integer,intent(in) :: i                                           !< Index (member or subset)
real(kind_real),intent(out) :: fld_c0a(geom%nc0a,geom%nl0,nam%nv) !< Field on Sc0 subset, halo A

! Local variables
integer :: iv

! Loop over fields
do iv=1,nam%nv
   call ens%get_c0(mpl,iv,geom,fieldtype,i,fld_c0a(:,:,iv))
end do

end subroutine ens_get_c0_all

!----------------------------------------------------------------------
! Subroutine: ens_set_c0_single
!> Set ensemble member on subset Sc0, single field
!----------------------------------------------------------------------
subroutine ens_set_c0_single(ens,mpl,iv,geom,fieldtype,i,fld_c0a)

implicit none

! Passed variables
class(ens_type),intent(inout) :: ens                      !< Ensemble
type(mpl_type),intent(inout) :: mpl                       !< MPI data
integer,intent(in) :: iv                                  !< Variable index
type(geom_type),intent(in) :: geom                        !< Geometry
character(len=*),intent(in) :: fieldtype                  !< Field type ('member', 'pert', 'mean', 'm2' or 'm4')
integer,intent(in) :: i                                   !< Index (member or subset)
real(kind_real),intent(in) :: fld_c0a(geom%nc0a,geom%nl0) !< Field on Sc0 subset, halo A

! Local variables
integer :: isub
real(kind_real),allocatable :: fld_mga(:,:),mean(:,:),fld_c0a_tmp(:,:)
character(len=1024),parameter :: subr = 'ens_set_c0_single'

if (.not.geom%same_grid) then
   ! Allocation
   allocate(fld_mga(geom%nmga,geom%nl0))

   ! Subset Sc0 to model grid
   call geom%copy_c0a_to_mga(mpl,fld_c0a,fld_mga)
end if

select case (trim(fieldtype))
case ('member')
   ! Fortran array to fieldset
   if (geom%same_grid) then
      call ens%mem(i)%from_array(mpl,iv,fld_c0a)
   else
      call ens%mem(i)%from_array(mpl,iv,fld_mga)
   end if
case ('pert')
   ! Allocation
   allocate(mean(geom%nmga,geom%nl0))
   if (geom%same_grid) allocate(fld_c0a_tmp(geom%nc0a,geom%nl0))

   ! Get sub-ensemble
   isub = (i-1)/(ens%ne/ens%nsub)+1

   ! Fieldset to Fortran array
   call ens%mean(isub)%to_array(mpl,iv,mean)

   ! Perturbation to member
   if (geom%same_grid) then
      fld_c0a_tmp = fld_c0a+mean
   else
      fld_mga = fld_mga+mean
   end if

   ! Fortran array to fieldset
   if (geom%same_grid) then
      call ens%mem(i)%from_array(mpl,iv,fld_c0a_tmp)
   else
      call ens%mem(i)%from_array(mpl,iv,fld_mga)
   end if

   ! Release memory
   deallocate(mean)
   if (geom%same_grid) deallocate(fld_c0a_tmp)
case ('mean')
   ! Fortran array to fieldset
   if (geom%same_grid) then
      call ens%mean(i)%from_array(mpl,iv,fld_c0a)
   else
      call ens%mean(i)%from_array(mpl,iv,fld_mga)
   end if
case ('m2')
   ! Fortran array to fieldset
   if (geom%same_grid) then
      call ens%m2%from_array(mpl,iv,fld_c0a)
   else
      call ens%m2%from_array(mpl,iv,fld_mga)
   end if
case ('m4')
   ! Fortran array to fieldset
   if (geom%same_grid) then
      call ens%m4%from_array(mpl,iv,fld_c0a)
   else
      call ens%m4%from_array(mpl,iv,fld_mga)
   end if
case default
   call mpl%abort(subr,'wrong field type')
end select

end subroutine ens_set_c0_single

!----------------------------------------------------------------------
! Subroutine: ens_set_c0_all
!> Get ensemble member or perturbation on subset Sc0, all field
!----------------------------------------------------------------------
subroutine ens_set_c0_all(ens,mpl,nam,geom,fieldtype,i,fld_c0a)

implicit none

! Passed variables
class(ens_type),intent(inout) :: ens                             !< Ensemble
type(mpl_type),intent(inout) :: mpl                              !< MPI data
type(nam_type),intent(in) :: nam                                 !< Namelist
type(geom_type),intent(in) :: geom                               !< Geometry
character(len=*),intent(in) :: fieldtype                         !< Field type ('member', 'pert', 'mean', 'm2' or 'm4')
integer,intent(in) :: i                                          !< Index (member or subset)
real(kind_real),intent(in) :: fld_c0a(geom%nc0a,geom%nl0,nam%nv) !< Field on Sc0 subset, halo A

! Local variables
integer :: iv

! Loop over fields
do iv=1,nam%nv
   call ens%set_c0(mpl,iv,geom,fieldtype,i,fld_c0a(:,:,iv))
end do

end subroutine ens_set_c0_all

!----------------------------------------------------------------------
! Subroutine: ens_apply_bens
!> Apply raw ensemble covariance
!----------------------------------------------------------------------
subroutine ens_apply_bens(ens,mpl,nam,geom,fld)

implicit none

! Passed variables
class(ens_type),intent(in) :: ens                               !< Ensemble
type(mpl_type),intent(inout) :: mpl                             !< MPI data
type(nam_type),intent(in) :: nam                                !< Namelist
type(geom_type),intent(in) :: geom                              !< Geometry
real(kind_real),intent(inout) :: fld(geom%nc0a,geom%nl0,nam%nv) !< Field

! Local variable
integer :: ie,ic0a,il0,iv
real(kind_real) :: alpha,norm
real(kind_real) :: fld_copy(geom%nc0a,geom%nl0,nam%nv)
real(kind_real) :: pert(geom%nc0a,geom%nl0,nam%nv)

! Initialization
fld_copy = fld

! Apply ensemble covariance formula
fld = 0.0
norm = 1.0/real(ens%ne-1,kind_real)
do ie=1,ens%ne
   ! Get perturbation on subset Sc0
   call ens%get_c0(mpl,nam,geom,'pert',ie,pert)

   ! Copy value or set at missing value
   do il0=1,geom%nl0
      do ic0a=1,geom%nc0a
         if (.not.geom%gmask_c0a(ic0a,il0)) pert(ic0a,il0,:) = mpl%msv%valr
      end do
   end do

   ! Dot product
   call mpl%dot_prod(pert,fld_copy,alpha)

   ! Schur product
   !$omp parallel do schedule(static) private(iv,il0,ic0a)
   do iv=1,nam%nv
      do il0=1,geom%nl0
         do ic0a=1,geom%nc0a
            if (geom%gmask_c0a(ic0a,il0)) fld(ic0a,il0,iv) = fld(ic0a,il0,iv)+alpha*pert(ic0a,il0,iv)*norm
         end do
      end do
   end do
   !$omp end parallel do
end do

end subroutine ens_apply_bens

!----------------------------------------------------------------------
! Subroutine: ens_apply_bens_dirac
!> Apply raw ensemble covariance to a Dirac (faster formulation)
!----------------------------------------------------------------------
subroutine ens_apply_bens_dirac(ens,mpl,nam,geom,iprocdir,ic0adir,il0dir,ivdir,fld)

implicit none

! Passed variables
class(ens_type),intent(in) :: ens                             !< Ensemble
type(mpl_type),intent(inout) :: mpl                           !< MPI data
type(nam_type),intent(in) :: nam                              !< Namelist
type(geom_type),intent(in) :: geom                            !< Geometry
integer,intent(in) :: iprocdir                                !< Processor index for dirac function
integer,intent(in) :: ic0adir                                 !< Subset Sc0, halo A index for dirac function
integer,intent(in) :: il0dir                                  !< Subset Sl0 index for dirac function
integer,intent(in) :: ivdir                                   !< Variable index for dirac function
real(kind_real),intent(out) :: fld(geom%nc0a,geom%nl0,nam%nv) !< Field

! Local variable
integer :: ie,ic0a,il0,iv
real(kind_real) :: fld_c0a(geom%nc0a,geom%nl0,nam%nv)
real(kind_real) :: alpha(ens%ne),norm

! Apply ensemble covariance formula for a Dirac function
norm = 1.0/real(ens%ne-1,kind_real)
do ie=1,ens%ne
   ! Get perturbation on subset Sc0
   call ens%get_c0(mpl,ivdir,geom,'pert',ie,fld_c0a(:,:,1))

   ! Get member value at Dirac point
   if (mpl%myproc==iprocdir) alpha(ie) = fld_c0a(ic0adir,il0dir,1)
end do
call mpl%f_comm%broadcast(alpha,iprocdir-1)
fld = 0.0
do ie=1,ens%ne
   ! Get perturbation on subset Sc0
   call ens%get_c0(mpl,nam,geom,'pert',ie,fld_c0a)

   do iv=1,nam%nv
      ! Apply Dirac-specific formula
      do il0=1,geom%nl0
         do ic0a=1,geom%nc0a
            if (geom%gmask_c0a(ic0a,il0)) then
               fld(ic0a,il0,iv) = fld(ic0a,il0,iv)+alpha(ie)*fld_c0a(ic0a,il0,iv)*norm
            else
               fld(ic0a,il0,iv) = mpl%msv%valr
            end if
         end do
      end do
   end do
end do

end subroutine ens_apply_bens_dirac

!----------------------------------------------------------------------
! Subroutine: ens_normality
!> Perform some normality diagnostics
!----------------------------------------------------------------------
subroutine ens_normality(ens,mpl,nam,geom,io)

implicit none

! Passed variables
class(ens_type),intent(inout) :: ens !< Ensemble
type(mpl_type),intent(inout) :: mpl  !< MPI data
type(nam_type),intent(in) :: nam     !< Namelist
type(geom_type),intent(in) :: geom   !< Geometry
type(io_type),intent(in) :: io       !< I/O

! Local variables
integer :: ncid,nloc_id,ne_id,nem1_id,ic0a_id,il0_id,iv_id,order_id,ens_norm_id,ens_step_id
integer :: iv,il0,ic0a,ie,nloc,iloc,nglb
integer,allocatable :: ic0a_loc(:),il0_loc(:),iv_loc(:),order(:,:)
real(kind_real) :: norm
real(kind_real) :: fld_c0a(geom%nc0a,geom%nl0,ens%ne)
real(kind_real) :: m2(geom%nc0a,geom%nl0,nam%nv)
real(kind_real) :: m4(geom%nc0a,geom%nl0,nam%nv)
real(kind_real) :: kurt(geom%nc0a,geom%nl0,nam%nv)
real(kind_real),allocatable :: ens_loc(:),ens_norm(:,:),ens_step(:,:)
character(len=1024) :: filename
character(len=1024),parameter :: subr = 'ens_normality'

! Set file name
filename = trim(nam%prefix)//'_umf'

! Write vertical unit
call io%fld_write(mpl,nam,geom,filename,'vunit',geom%vunit_c0a)

! Compute variance and kurtosis
write(mpl%info,'(a7,a)') '','Compute variance and kurtosis'
call mpl%flush
call ens%compute_moments(mpl,nam,geom)
call ens%get_c0(mpl,nam,geom,'m2',0,m2)
call ens%get_c0(mpl,nam,geom,'m4',0,m4)
kurt = mpl%msv%valr
do iv=1,nam%nv
   do il0=1,geom%nl0
      do ic0a=1,geom%nc0a
         if (m2(ic0a,il0,iv)>0.0) kurt(ic0a,il0,iv) = m4(ic0a,il0,iv)/m2(ic0a,il0,iv)**2
      end do
   end do
end do

! Write
call io%fld_write(mpl,nam,geom,filename,'m2',m2)
call io%fld_write(mpl,nam,geom,filename,'m4',m4)
call io%fld_write(mpl,nam,geom,filename,'kurt',kurt)

! Allocation
nloc = count(mpl%msv%isnot(kurt).and.(kurt>nam%gen_kurt_th))
allocate(ic0a_loc(nloc))
allocate(il0_loc(nloc))
allocate(iv_loc(nloc))
allocate(order(ens%ne,nloc))
allocate(ens_loc(ens%ne))
allocate(ens_norm(ens%ne,nloc))
allocate(ens_step(ens%ne-1,nloc))
call mpl%f_comm%allreduce(nloc,nglb,fckit_mpi_sum())

! Save ensemble
write(mpl%info,'(a7,a,i6,a,i6,a)') '','Save ensemble for ',nloc,' points (',nglb,' total)'
call mpl%flush
iloc = 0
do iv=1,nam%nv
   do ie=1,ens%ne
      ! Get perturbation on subset Sc0
      call ens%get_c0(mpl,iv,geom,'pert',ie,fld_c0a(:,:,ie))
   end do

   do il0=1,geom%nl0
      do ic0a=1,geom%nc0a
         if (mpl%msv%isnot(kurt(ic0a,il0,iv)).and.(kurt(ic0a,il0,iv)>nam%gen_kurt_th)) then
            ! Update index
            iloc = iloc+1

            ! Copy data
            ic0a_loc(iloc) = ic0a
            il0_loc(iloc) = il0
            iv_loc(iloc) = iv
            do ie=1,ens%ne
               ens_loc(ie) = fld_c0a(ic0a,il0,ie)
            end do

            ! Sort ensemble
            call qsort(ens%ne,ens_loc,order(:,iloc))

            ! Normalize ensemble
            norm = 1.0/(maxval(ens_loc)-minval(ens_loc))
            ens_norm(:,iloc) = (ens_loc-minval(ens_loc))*norm

            ! Compute ensemble steps
            do ie=1,ens%ne-1
               ens_step(ie,iloc) = ens_norm(ie+1,iloc)-ens_norm(ie,iloc)
            end do
         end if
      end do
   end do
end do

! Write normality diagnostics
write(mpl%info,'(a7,a)') '','Write normality diagnostics'
call mpl%flush

! Set file name
write(filename,'(a,a,i6.6,a,i6.6)') trim(nam%prefix),'_normality_',mpl%nproc,'-',mpl%myproc
ncid = mpl%nc_file_create_or_open(subr,trim(nam%datadir)//'/'//trim(filename)//'.nc')

! Define dimensions
nloc_id = mpl%nc_dim_define_or_get(subr,ncid,'nloc',nloc)
ne_id = mpl%nc_dim_define_or_get(subr,ncid,'ne',ens%ne)
nem1_id = mpl%nc_dim_define_or_get(subr,ncid,'nem1',ens%ne-1)

if (nloc>0) then
   ! Define variables
   ic0a_id = mpl%nc_var_define_or_get(subr,ncid,'ic0a',nf90_int,(/nloc_id/))
   il0_id = mpl%nc_var_define_or_get(subr,ncid,'il0',nf90_int,(/nloc_id/))
   iv_id = mpl%nc_var_define_or_get(subr,ncid,'iv',nf90_int,(/nloc_id/))
   order_id = mpl%nc_var_define_or_get(subr,ncid,'order',nf90_int,(/ne_id,nloc_id/))
   ens_norm_id = mpl%nc_var_define_or_get(subr,ncid,'ens_norm',nc_kind_real,(/ne_id,nloc_id/))
   ens_step_id = mpl%nc_var_define_or_get(subr,ncid,'ens_step',nc_kind_real,(/nem1_id,nloc_id/))
end if

if (nloc>0) then
   ! Write variables
   call mpl%ncerr(subr,nf90_put_var(ncid,ic0a_id,ic0a_loc))
   call mpl%ncerr(subr,nf90_put_var(ncid,il0_id,il0_loc))
   call mpl%ncerr(subr,nf90_put_var(ncid,iv_id,iv_loc))
   call mpl%ncerr(subr,nf90_put_var(ncid,order_id,order))
   call mpl%ncerr(subr,nf90_put_var(ncid,ens_norm_id,ens_norm))
   call mpl%ncerr(subr,nf90_put_var(ncid,ens_step_id,ens_step))
end if

! Close file
call mpl%ncerr(subr,nf90_close(ncid))

! Release memory
deallocate(ic0a_loc)
deallocate(il0_loc)
deallocate(iv_loc)
deallocate(order)
deallocate(ens_loc)
deallocate(ens_norm)
deallocate(ens_step)

end subroutine ens_normality

end module type_ens
