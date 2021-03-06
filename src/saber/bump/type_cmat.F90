!----------------------------------------------------------------------
! Module: type_cmat
!> C matrix derived type
! Author: Benjamin Menetrier
! Licensing: this code is distributed under the CeCILL-C license
! Copyright © 2015-... UCAR, CERFACS, METEO-FRANCE and IRIT
!----------------------------------------------------------------------
module type_cmat

use fckit_mpi_module, only: fckit_mpi_sum
use netcdf
use tools_const, only: rad2deg,reqkm,req
use tools_func, only: lct_d2h,lct_h2r
use tools_kinds, only: kind_real,huge_real
use type_bpar, only: bpar_type
use type_cmat_blk, only: cmat_blk_type
use type_diag, only: diag_type
use type_geom, only: geom_type
use type_hdiag, only: hdiag_type
use type_io, only: io_type
use type_lct, only: lct_type
use type_mom, only: mom_type
use type_mpl, only: mpl_type
use type_nam, only: nam_type
use type_rng, only: rng_type
use type_samp, only: samp_type

implicit none

! C matrix derived type
type cmat_type
   character(len=1024) :: prefix             !< Prefix
   type(cmat_blk_type),allocatable :: blk(:) !< C matrix blocks
   logical :: allocated                      !< Allocation flag
contains
   procedure :: cmat_alloc
   procedure :: cmat_alloc_blk
   generic :: alloc => cmat_alloc,cmat_alloc_blk
   procedure :: init => cmat_init
   procedure :: partial_dealloc => cmat_partial_dealloc
   procedure :: dealloc => cmat_dealloc
   procedure :: read => cmat_read
   procedure :: write => cmat_write
   procedure :: from_hdiag => cmat_from_hdiag
   procedure :: from_lct => cmat_from_lct
   procedure :: from_nam => cmat_from_nam
   procedure :: from_bump => cmat_from_bump
   procedure :: setup_sampling => cmat_setup_sampling
end type cmat_type

private
public :: cmat_type

contains

!----------------------------------------------------------------------
! Subroutine: cmat_alloc
!> C matrix allocation
!----------------------------------------------------------------------
subroutine cmat_alloc(cmat,bpar)

implicit none

! Passed variables
class(cmat_type),intent(inout) :: cmat !< C matrix
type(bpar_type),intent(in) :: bpar     !< Block parameters

! Allocation
if (.not.allocated(cmat%blk)) allocate(cmat%blk(bpar%nbe))

end subroutine cmat_alloc

!----------------------------------------------------------------------
! Subroutine: cmat_alloc_blk
!> Allocation
!----------------------------------------------------------------------
subroutine cmat_alloc_blk(cmat,nam,geom,bpar)

implicit none

! Passed variables
class(cmat_type),intent(inout) :: cmat !< C matrix
type(nam_type),intent(in) :: nam       !< Namelist
type(geom_type),intent(in) :: geom     !< Geometry
type(bpar_type),intent(in) :: bpar     !< Block parameters

! Local variables
integer :: ib

! Allocation
do ib=1,bpar%nbe
   if (bpar%B_block(ib).and.bpar%nicas_block(ib)) then
      cmat%blk(ib)%ib = ib
      call cmat%blk(ib)%alloc(nam,geom,bpar)
   end if
end do

! Update allocation flag
cmat%allocated = .true.

end subroutine cmat_alloc_blk

!----------------------------------------------------------------------
! Subroutine: cmat_init
!> C matrix initialization
!----------------------------------------------------------------------
subroutine cmat_init(cmat,mpl,nam,bpar)

implicit none

! Passed variables
class(cmat_type),intent(inout) :: cmat !< C matrix
type(mpl_type),intent(inout) :: mpl    !< MPI data
type(nam_type),intent(in) :: nam       !< Namelist
type(bpar_type),intent(in) :: bpar     !< Block parameters

! Local variables
integer :: ib

! Initialize blocks
do ib=1,bpar%nbe
   if (bpar%B_block(ib).and.bpar%nicas_block(ib)) call cmat%blk(ib)%init(mpl,nam,bpar)
end do

end subroutine cmat_init

!----------------------------------------------------------------------
! Subroutine: cmat_partial_dealloc
!> Release memory (partial)
!----------------------------------------------------------------------
subroutine cmat_partial_dealloc(cmat)

implicit none

! Passed variables
class(cmat_type),intent(inout) :: cmat !< C matrix

! Local variables
integer :: ib

! Release memory
if (allocated(cmat%blk)) then
   do ib=1,size(cmat%blk)
      call cmat%blk(ib)%partial_bump_dealloc
      call cmat%blk(ib)%partial_dealloc
   end do
end if

! Update allocation flag
cmat%allocated = .false.

end subroutine cmat_partial_dealloc

!----------------------------------------------------------------------
! Subroutine: cmat_dealloc
!> Release memory
!----------------------------------------------------------------------
subroutine cmat_dealloc(cmat)

implicit none

! Passed variables
class(cmat_type),intent(inout) :: cmat !< C matrix

! Local variables
integer :: ib

! Release memory
if (allocated(cmat%blk)) then
   do ib=1,size(cmat%blk)
      call cmat%blk(ib)%dealloc
   end do
   deallocate(cmat%blk)
end if

! Update allocation flag
cmat%allocated = .false.

end subroutine cmat_dealloc

!----------------------------------------------------------------------
! Subroutine: cmat_read
!> Read
!----------------------------------------------------------------------
subroutine cmat_read(cmat,mpl,nam,geom,bpar,io)

implicit none

! Passed variables
class(cmat_type),intent(inout) :: cmat !< C matrix
type(mpl_type),intent(inout) :: mpl    !< MPI data
type(nam_type),intent(in) :: nam       !< Namelist
type(geom_type),intent(in) :: geom     !< Geometry
type(bpar_type),intent(in) :: bpar     !< Block parameters
type(io_type),intent(in) :: io         !< I/O

! Local variables
integer :: ib,ncid,anisotropic
character(len=1024) :: filename,grpname
character(len=1024),parameter :: subr = 'cmat_read'

! Allocation
call cmat%alloc(bpar)

! Set filename
filename = trim(nam%prefix)//'_cmat'

if (mpl%main) then
   ! Read attribute
   call mpl%ncerr(subr,nf90_open(trim(nam%datadir)//'/'//trim(filename)//'.nc',nf90_nowrite,ncid))
   call mpl%ncerr(subr,nf90_get_att(ncid,nf90_global,'anisotropic',anisotropic))
   call mpl%ncerr(subr,nf90_close(ncid))
end if

! Broadcast attribute
call mpl%f_comm%broadcast(anisotropic,mpl%rootproc-1)

! Copy attribute
do ib=1,bpar%nbe
   if (bpar%B_block(ib).and.bpar%nicas_block(ib)) then
      if (anisotropic==1) then
         cmat%blk(ib)%anisotropic = .true.
      elseif (anisotropic==0) then
         cmat%blk(ib)%anisotropic = .false.
      else
         call mpl%abort(subr,'wrong anisotropic flag')
      end if
   end if
end do

! Allocation
call cmat%alloc(nam,geom,bpar)

do ib=1,bpar%nbe
   if (bpar%B_block(ib).and.bpar%nicas_block(ib)) then
      ! Read fields
      call nam%io_key_value(bpar%blockname(ib),grpname)
      if (bpar%nicas_block(ib)) then
         call io%fld_read(mpl,nam,geom,filename,'coef_ens',cmat%blk(ib)%coef_ens,grpname)
         call io%fld_read(mpl,nam,geom,filename,'coef_sta',cmat%blk(ib)%coef_sta,grpname)
         call io%fld_read(mpl,nam,geom,filename,'rh',cmat%blk(ib)%rh,grpname)
         call io%fld_read(mpl,nam,geom,filename,'rv',cmat%blk(ib)%rv,grpname)
         call io%fld_read(mpl,nam,geom,filename,'rhs',cmat%blk(ib)%rhs,grpname)
         call io%fld_read(mpl,nam,geom,filename,'rvs',cmat%blk(ib)%rvs,grpname)
         if (cmat%blk(ib)%anisotropic) then
            call io%fld_read(mpl,nam,geom,filename,'H11',cmat%blk(ib)%H11,grpname)
            call io%fld_read(mpl,nam,geom,filename,'H22',cmat%blk(ib)%H22,grpname)
            call io%fld_read(mpl,nam,geom,filename,'H33',cmat%blk(ib)%H33,grpname)
            call io%fld_read(mpl,nam,geom,filename,'H12',cmat%blk(ib)%H12,grpname)
            call io%fld_read(mpl,nam,geom,filename,'Hcoef',cmat%blk(ib)%Hcoef,grpname)
         end if
      end if
   end if
end do

end subroutine cmat_read

!----------------------------------------------------------------------
! Subroutine: cmat_write
!> Write
!----------------------------------------------------------------------
subroutine cmat_write(cmat,mpl,nam,geom,bpar,io)

implicit none

! Passed variables
class(cmat_type),intent(in) :: cmat !< C matrix
type(mpl_type),intent(inout) :: mpl !< MPI data
type(nam_type),intent(in) :: nam    !< Namelist
type(geom_type),intent(in) :: geom  !< Geometry
type(bpar_type),intent(in) :: bpar  !< Block parameters
type(io_type),intent(in) :: io      !< I/O

! Local variables
integer :: ib
integer :: ncid
character(len=1024) :: filename,grpname
character(len=1024),parameter :: subr = 'cmat_write'

! Set filename
filename = trim(nam%prefix)//'_cmat'

! Write vertical unit
call io%fld_write(mpl,nam,geom,filename,'vunit',geom%vunit_c0a)

do ib=1,bpar%nbe
   if (bpar%B_block(ib).and.bpar%nicas_block(ib)) then
      ! Write fields
      call nam%io_key_value(bpar%blockname(ib),grpname)
      if (bpar%nicas_block(ib)) then
         call io%fld_write(mpl,nam,geom,filename,'coef_ens',cmat%blk(ib)%coef_ens,grpname)
         call io%fld_write(mpl,nam,geom,filename,'coef_sta',cmat%blk(ib)%coef_sta,grpname)
         call io%fld_write(mpl,nam,geom,filename,'rh',cmat%blk(ib)%rh,grpname)
         call io%fld_write(mpl,nam,geom,filename,'rv',cmat%blk(ib)%rv,grpname)
         call io%fld_write(mpl,nam,geom,filename,'rhs',cmat%blk(ib)%rhs,grpname)
         call io%fld_write(mpl,nam,geom,filename,'rvs',cmat%blk(ib)%rvs,grpname)
         if (cmat%blk(ib)%anisotropic) then
            call io%fld_write(mpl,nam,geom,filename,'H11',cmat%blk(ib)%H11,grpname)
            call io%fld_write(mpl,nam,geom,filename,'H22',cmat%blk(ib)%H22,grpname)
            call io%fld_write(mpl,nam,geom,filename,'H33',cmat%blk(ib)%H33,grpname)
            call io%fld_write(mpl,nam,geom,filename,'H12',cmat%blk(ib)%H12,grpname)
            call io%fld_write(mpl,nam,geom,filename,'Hcoef',cmat%blk(ib)%Hcoef,grpname)
         end if
      end if

      if (mpl%main) then
         ! Write attributes
         call mpl%ncerr(subr,nf90_open(trim(nam%datadir)//'/'//trim(filename)//'.nc',nf90_write,ncid))
         if (cmat%blk(ib)%anisotropic) then
            call mpl%ncerr(subr,nf90_put_att(ncid,nf90_global,'anisotropic',1))
         else
            call mpl%ncerr(subr,nf90_put_att(ncid,nf90_global,'anisotropic',0))
         end if
         call mpl%ncerr(subr,nf90_close(ncid))
      end if
   end if
end do

end subroutine cmat_write

!----------------------------------------------------------------------
! Subroutine: cmat_from_hdiag
!> Import HDIAG into C matrix
!----------------------------------------------------------------------
subroutine cmat_from_hdiag(cmat,mpl,nam,geom,bpar,hdiag)

implicit none

! Passed variables
class(cmat_type),intent(inout) :: cmat !< C matrix
type(mpl_type),intent(inout) :: mpl    !< MPI data
type(nam_type),intent(in) :: nam       !< Namelist
type(geom_type),intent(in) :: geom     !< Geometry
type(bpar_type),intent(in) :: bpar     !< Block parameters
type(hdiag_type),intent(in) :: hdiag   !< Hybrid diagnostics

! Local variables
integer :: ib,n,i,il0,il0i,ic2a,ic0a
real(kind_real),allocatable :: fld_c2a(:,:,:),fld_c2b(:,:),fld_c0a(:,:,:)
character(len=1024),parameter :: subr = 'cmat_from_hdiag'

! Allocation
call cmat%alloc(bpar)

! Copy attributes
do ib=1,bpar%nbe
   if (bpar%B_block(ib).and.bpar%nicas_block(ib)) cmat%blk(ib)%anisotropic = .false.
end do

! Allocation
call cmat%alloc(nam,geom,bpar)
if (nam%local_diag) then
   allocate(fld_c2a(hdiag%samp%nc2a,geom%nl0,4))
   allocate(fld_c2b(hdiag%samp%nc2b,geom%nl0))
   allocate(fld_c0a(geom%nc0a,geom%nl0,4))
end if

! Initialization
call cmat%init(mpl,nam,bpar)

! Convolution parameters
do ib=1,bpar%nbe
   if (bpar%B_block(ib)) then
      if (bpar%nicas_block(ib)) then
         if (nam%local_diag) then
            ! Initialization
            fld_c2a = mpl%msv%valr

            ! Copy data
            n = 2
            if (bpar%fit_block(ib)) n = n+2
            do ic2a=1,hdiag%samp%nc2a
               select case (trim(nam%method))
               case ('cor')
                  fld_c2a(ic2a,:,1) = hdiag%cov_1%blk(ic2a,ib)%coef_ens
                  fld_c2a(ic2a,:,2) = 0.0
                  if (bpar%fit_block(ib)) then
                     fld_c2a(ic2a,:,3) = hdiag%cor_1%blk(ic2a,ib)%fit_rh
                     fld_c2a(ic2a,:,4) = hdiag%cor_1%blk(ic2a,ib)%fit_rv
                  end if
               case ('loc')
                  fld_c2a(ic2a,:,1) = hdiag%loc_1%blk(ic2a,ib)%coef_ens
                  fld_c2a(ic2a,:,2) = 0.0
                  if (bpar%fit_block(ib)) then
                     fld_c2a(ic2a,:,3) = hdiag%loc_1%blk(ic2a,ib)%fit_rh
                     fld_c2a(ic2a,:,4) = hdiag%loc_1%blk(ic2a,ib)%fit_rv
                  end if
               case ('hyb-avg','hyb-rnd')
                  fld_c2a(ic2a,:,1) = hdiag%loc_2%blk(ic2a,ib)%coef_ens
                  fld_c2a(ic2a,:,2) = hdiag%loc_2%blk(ic2a,ib)%coef_sta
                  if (bpar%fit_block(ib)) then
                     fld_c2a(ic2a,:,3) = hdiag%loc_2%blk(ic2a,ib)%fit_rh
                     fld_c2a(ic2a,:,4) = hdiag%loc_2%blk(ic2a,ib)%fit_rv
                  end if
               case ('dual-ens')
                  call mpl%abort(subr,'dual-ens not ready yet for C matrix')
               case default
                  call mpl%abort(subr,'cmat not implemented yet for this method')
               end select
            end do

            do i=1,n
               ! Fill missing values
               do il0=1,geom%nl0
                  call hdiag%samp%diag_fill(mpl,fld_c2a(:,il0,i))
               end do

               ! Interpolate
               call hdiag%samp%com_AB%ext(mpl,geom%nl0,fld_c2a(:,:,i),fld_c2b)
               do il0=1,geom%nl0
                  il0i = min(il0,geom%nl0i)
                  call hdiag%samp%h(il0i)%apply(mpl,fld_c2b(:,il0),fld_c0a(:,il0,i))
               end do
            end do

            ! Copy to C matrix
            cmat%blk(ib)%coef_ens = fld_c0a(:,:,1)
            call mpl%f_comm%allreduce(sum(cmat%blk(ib)%coef_ens,mask=geom%gmask_c0a),cmat%blk(ib)%wgt,fckit_mpi_sum())
            cmat%blk(ib)%wgt = cmat%blk(ib)%wgt/real(sum(geom%nc0_gmask(1:geom%nl0)),kind_real)
            cmat%blk(ib)%coef_sta = fld_c0a(:,:,2)
            if (bpar%fit_block(ib)) then
               cmat%blk(ib)%rh = fld_c0a(:,:,3)
               cmat%blk(ib)%rv = fld_c0a(:,:,4)
            end if
         else
            ! Initialization
            cmat%blk(ib)%coef_ens = mpl%msv%valr
            cmat%blk(ib)%wgt = mpl%msv%valr
            cmat%blk(ib)%coef_sta = mpl%msv%valr
            if (bpar%fit_block(ib)) then
               cmat%blk(ib)%rh = mpl%msv%valr
               cmat%blk(ib)%rv = mpl%msv%valr
            end if

            ! Copy to C matrix
            do il0=1,geom%nl0
               ! Copy data
               select case (trim(nam%method))
               case ('cor')
                  cmat%blk(ib)%coef_ens(:,il0) = hdiag%cov_1%blk(0,ib)%coef_ens(il0)
                  cmat%blk(ib)%wgt = sum(hdiag%cov_1%blk(0,ib)%coef_ens)/real(geom%nl0,kind_real)
                  cmat%blk(ib)%coef_sta(:,il0) = 0.0
                  if (bpar%fit_block(ib)) then
                     cmat%blk(ib)%rh(:,il0) = hdiag%cor_1%blk(0,ib)%fit_rh(il0)
                     cmat%blk(ib)%rv(:,il0) = hdiag%cor_1%blk(0,ib)%fit_rv(il0)
                  else

                  end if
               case ('loc')
                  cmat%blk(ib)%coef_ens(:,il0) = hdiag%loc_1%blk(0,ib)%coef_ens(il0)
                  cmat%blk(ib)%wgt = sum(hdiag%loc_1%blk(0,ib)%coef_ens)/real(geom%nl0,kind_real)
                  cmat%blk(ib)%coef_sta(:,il0) = 0.0
                  if (bpar%fit_block(ib)) then
                     cmat%blk(ib)%rh(:,il0) = hdiag%loc_1%blk(0,ib)%fit_rh(il0)
                     cmat%blk(ib)%rv(:,il0) = hdiag%loc_1%blk(0,ib)%fit_rv(il0)
                  end if
               case ('hyb-avg','hyb-rnd')
                  cmat%blk(ib)%coef_ens(:,il0) = hdiag%loc_2%blk(0,ib)%coef_ens(il0)
                  cmat%blk(ib)%wgt = sum(hdiag%loc_2%blk(0,ib)%coef_ens)/real(geom%nl0,kind_real)
                  cmat%blk(ib)%coef_sta(:,il0) = hdiag%loc_2%blk(0,ib)%coef_sta
                  if (bpar%fit_block(ib)) then
                     cmat%blk(ib)%rh(:,il0) = hdiag%loc_2%blk(0,ib)%fit_rh(il0)
                     cmat%blk(ib)%rv(:,il0) = hdiag%loc_2%blk(0,ib)%fit_rv(il0)
                  end if
               case ('dual-ens')
                  call mpl%abort(subr,'dual-ens not ready yet for C matrix')
               case default
                  call mpl%abort(subr,'cmat not implemented yet for this method')
               end select
            end do
         end if

         ! Set mask
         do il0=1,geom%nl0
            do ic0a=1,geom%nc0a
               if (.not.geom%gmask_c0a(ic0a,il0)) then
                  cmat%blk(ib)%coef_ens(ic0a,il0) = mpl%msv%valr
                  cmat%blk(ib)%coef_sta(ic0a,il0) = mpl%msv%valr
                  if (bpar%fit_block(ib)) then
                     cmat%blk(ib)%rh(ic0a,il0) = mpl%msv%valr
                     cmat%blk(ib)%rv(ic0a,il0) = mpl%msv%valr
                  end if
               end if
            end do
         end do
      else
         ! Define weight only
         select case (trim(nam%method))
         case ('cor')
            cmat%blk(ib)%wgt = sum(hdiag%cov_1%blk(0,ib)%coef_ens)/real(geom%nl0,kind_real)
         case ('loc')
            cmat%blk(ib)%wgt = sum(hdiag%loc_1%blk(0,ib)%coef_ens)/real(geom%nl0,kind_real)
         case ('hyb-avg','hyb-rnd')
            cmat%blk(ib)%wgt = sum(hdiag%loc_2%blk(0,ib)%coef_ens)/real(geom%nl0,kind_real)
         case ('dual-ens')
            call mpl%abort(subr,'dual-ens not ready yet for C matrix')
         case default
            call mpl%abort(subr,'cmat not implemented yet for this method')
         end select
      end if
   end if
end do

! Release memory
if (nam%local_diag) then
   deallocate(fld_c2a)
   deallocate(fld_c2b)
   deallocate(fld_c0a)
end if

end subroutine cmat_from_hdiag

!----------------------------------------------------------------------
! Subroutine: cmat_from_lct
!> Import LCT into C matrix
!----------------------------------------------------------------------
subroutine cmat_from_lct(cmat,mpl,nam,geom,bpar,lct)

implicit none

! Passed variables
class(cmat_type),intent(inout) :: cmat !< C matrix
type(mpl_type),intent(inout) :: mpl    !< MPI data
type(nam_type),intent(in) :: nam       !< Namelist
type(geom_type),intent(in) :: geom     !< Geometry
type(bpar_type),intent(in) :: bpar     !< Block parameters
type(lct_type),intent(in) :: lct       !< LCT

! Local variables
integer :: ib,iv,jv,iscales,il0,ic0a
character(len=1024),parameter :: subr = 'cmat_from_lct'

! Allocation
call cmat%alloc(bpar)

! Copy attributes
do ib=1,bpar%nbe
   if (bpar%B_block(ib).and.bpar%nicas_block(ib)) cmat%blk(ib)%anisotropic = .true.
end do

! Allocation
call cmat%alloc(nam,geom,bpar)

! Initialization
call cmat%init(mpl,nam,bpar)

! Convolution parameters
do ib=1,bpar%nbe
   if (bpar%B_block(ib).and.bpar%nicas_block(ib)) then
      ! Indices
      iv = bpar%b_to_v1(ib)
      jv = bpar%b_to_v2(ib)
      if (iv/=jv) call mpl%abort(subr,'only diagonal blocks for cmat_from_lct')

      if (lct%blk(ib)%nscales>1) call mpl%warning(subr,'only the first scale is used to define cmat from LCT')
      iscales = 1

      do il0=1,geom%nl0
         do ic0a=1,geom%nc0a
            if (geom%gmask_c0a(ic0a,il0)) then
               ! Copy LCT
               cmat%blk(ib)%H11(ic0a,il0) = lct%blk(ib)%H11(ic0a,il0,iscales)
               cmat%blk(ib)%H22(ic0a,il0) = lct%blk(ib)%H22(ic0a,il0,iscales)
               cmat%blk(ib)%H33(ic0a,il0) = lct%blk(ib)%H33(ic0a,il0,iscales)
               cmat%blk(ib)%H12(ic0a,il0) = lct%blk(ib)%H12(ic0a,il0,iscales)

               ! Copy scale coefficient
               cmat%blk(ib)%Hcoef(ic0a,il0) = lct%blk(ib)%Dcoef(ic0a,il0,iscales)

               ! Compute support radii
               call lct_h2r(mpl,cmat%blk(ib)%H11(ic0a,il0),cmat%blk(ib)%H22(ic0a,il0),cmat%blk(ib)%H33(ic0a,il0), &
 & cmat%blk(ib)%H12(ic0a,il0),cmat%blk(ib)%rh(ic0a,il0),cmat%blk(ib)%rv(ic0a,il0))
            end if
         end do
      end do

      ! Set coefficients
      cmat%blk(ib)%coef_ens = 1.0
      cmat%blk(ib)%coef_sta = 0.0
      cmat%blk(ib)%wgt = 1.0
   end if
end do

end subroutine cmat_from_lct

!----------------------------------------------------------------------
! Subroutine: cmat_from_nam
!> Import radii into C matrix
!----------------------------------------------------------------------
subroutine cmat_from_nam(cmat,mpl,nam,geom,bpar)

implicit none

! Passed variables
class(cmat_type),intent(inout) :: cmat !< C matrix
type(mpl_type),intent(inout) :: mpl    !< MPI data
type(nam_type),intent(in) :: nam       !< Namelist
type(geom_type),intent(in) :: geom     !< Geometry
type(bpar_type),intent(in) :: bpar     !< Block parameters

! Local variables
integer :: ib,iv,jv
character(len=1024),parameter :: subr = 'cmat_from_nam'

write(mpl%info,'(a)') '-------------------------------------------------------------------'
call mpl%flush
write(mpl%info,'(a)') '--- Copy namelist radii into C matrix'
call mpl%flush

if (.not.cmat%allocated) then
   ! Allocation
   call cmat%alloc(bpar)

   ! Set attributes
   do ib=1,bpar%nbe
      if (bpar%B_block(ib).and.bpar%nicas_block(ib)) cmat%blk(ib)%anisotropic = .false.
   end do

   ! Allocation
   call cmat%alloc(nam,geom,bpar)

   ! Initialization
   call cmat%init(mpl,nam,bpar)
end if

! Convolution parameters
do ib=1,bpar%nbe
   if (bpar%B_block(ib).and.bpar%nicas_block(ib)) then
      ! Indices
      iv = bpar%b_to_v1(ib)
      jv = bpar%b_to_v2(ib)
      if (iv/=jv) call mpl%abort(subr,'only diagonal blocks for cmat_from_nam')

      ! Copy support radii
      cmat%blk(ib)%rh = nam%rh
      cmat%blk(ib)%rv = nam%rv

      ! Set coefficients
      cmat%blk(ib)%coef_ens = 1.0
      cmat%blk(ib)%coef_sta = 0.0
      cmat%blk(ib)%wgt = 1.0
   end if
end do

end subroutine cmat_from_nam

!----------------------------------------------------------------------
! Subroutine: cmat_from_bump
!> Import C matrix from BUMP
!----------------------------------------------------------------------
subroutine cmat_from_bump(cmat,mpl,nam,geom,bpar)

implicit none

! Passed variables
class(cmat_type),intent(inout) :: cmat !< C matrix
type(mpl_type),intent(inout) :: mpl    !< MPI data
type(nam_type),intent(in) :: nam       !< Namelist
type(geom_type),intent(in) :: geom     !< Geometry
type(bpar_type),intent(in) :: bpar     !< Block parameters

! Local variables
integer :: ib,il0,ic0a
logical :: import_standard(bpar%nbe),import_static(bpar%nbe),import_anisotropic(bpar%nbe)

if (.not.cmat%allocated) then
   ! Allocation
   call cmat%alloc(bpar)

   ! Set attributes
   do ib=1,bpar%nbe
      if (bpar%B_block(ib).and.bpar%nicas_block(ib)) cmat%blk(ib)%anisotropic = .false.
   end do

   ! Allocation
   call cmat%alloc(nam,geom,bpar)

   ! Initialization
   call cmat%init(mpl,nam,bpar)
end if

do ib=1,bpar%nbe
   ! Initialization
   import_standard(ib) = .false.
   import_static(ib) = .false.
   import_anisotropic(ib) = .false.

   if (bpar%B_block(ib).and.bpar%nicas_block(ib)) then
      ! Define import configuration
      import_standard(ib) = allocated(cmat%blk(ib)%bump_rh).and.allocated(cmat%blk(ib)%bump_rv)
      import_static(ib) = allocated(cmat%blk(ib)%bump_coef_sta)
      import_anisotropic(ib) = allocated(cmat%blk(ib)%bump_D11).and.allocated(cmat%blk(ib)%bump_D22) &
 & .and.allocated(cmat%blk(ib)%bump_D33).and.allocated(cmat%blk(ib)%bump_D12).and.allocated(cmat%blk(ib)%bump_Dcoef)

      ! Define attributes
      cmat%blk(ib)%anisotropic = cmat%blk(ib)%anisotropic.or.import_anisotropic(ib)
   end if
end do

do ib=1,bpar%nbe
   if (bpar%B_block(ib).and.bpar%nicas_block(ib)) then
      if (import_standard(ib).or.import_static(ib).or.import_anisotropic(ib)) then
         write(mpl%info,'(a7,a,a)') '','Block ',trim(bpar%blockname(ib))
         call mpl%flush

         ! Copy values
         if (import_standard(ib)) then
            write(mpl%info,'(a10,a)') '','Standard import'
            call mpl%flush
            if (allocated(cmat%blk(ib)%bump_coef_ens)) then
               cmat%blk(ib)%coef_ens = cmat%blk(ib)%bump_coef_ens
            else
               cmat%blk(ib)%coef_ens = 1.0
            end if
            call mpl%f_comm%allreduce(sum(cmat%blk(ib)%coef_ens,mask=geom%gmask_c0a),cmat%blk(ib)%wgt,fckit_mpi_sum())
            cmat%blk(ib)%wgt = cmat%blk(ib)%wgt/real(sum(geom%nc0_gmask(1:geom%nl0)),kind_real)
            cmat%blk(ib)%rh = cmat%blk(ib)%bump_rh
            cmat%blk(ib)%rv = cmat%blk(ib)%bump_rv
         end if
         if (import_static(ib)) then
            write(mpl%info,'(a10,a)') '','Static import'
            call mpl%flush
            cmat%blk(ib)%coef_sta = cmat%blk(ib)%bump_coef_sta
         end if
         if (import_anisotropic(ib)) then
            write(mpl%info,'(a10,a)') '','Anisotropic import'
            call mpl%flush
            do il0=1,geom%nl0
               do ic0a=1,geom%nc0a
                  if (geom%gmask_c0a(ic0a,il0)) then
                     ! Copy LCT
                     call lct_d2h(mpl,cmat%blk(ib)%bump_D11(ic0a,il0),cmat%blk(ib)%bump_D22(ic0a,il0), &
 & cmat%blk(ib)%bump_D33(ic0a,il0),cmat%blk(ib)%bump_D12(ic0a,il0),cmat%blk(ib)%H11(ic0a,il0),cmat%blk(ib)%H22(ic0a,il0), &
 & cmat%blk(ib)%H33(ic0a,il0),cmat%blk(ib)%H12(ic0a,il0))

                     ! Copy scale coefficient
                     cmat%blk(ib)%Hcoef(ic0a,il0) = cmat%blk(ib)%bump_Dcoef(ic0a,il0)

                     ! Copy support radii
                     call lct_h2r(mpl,cmat%blk(ib)%H11(ic0a,il0),cmat%blk(ib)%H22(ic0a,il0),cmat%blk(ib)%H33(ic0a,il0), &   
 & cmat%blk(ib)%H12(ic0a,il0),cmat%blk(ib)%rh(ic0a,il0),cmat%blk(ib)%rv(ic0a,il0))
                  end if
               end do
            end do

            ! Set coefficients
            cmat%blk(ib)%coef_ens = 1.0
            cmat%blk(ib)%coef_sta = 0.0
            cmat%blk(ib)%wgt = 1.0
         end if
      end if
   end if

   ! Release memory (partial)
   call cmat%blk(ib)%partial_bump_dealloc
end do

end subroutine cmat_from_bump

!----------------------------------------------------------------------
! Subroutine: cmat_setup_sampling
!> Setup C matrix sampling
!----------------------------------------------------------------------
subroutine cmat_setup_sampling(cmat,mpl,nam,geom,bpar)

implicit none

! Passed variables
class(cmat_type),intent(inout) :: cmat !< C matrix
type(mpl_type),intent(inout) :: mpl    !< MPI data
type(nam_type),intent(in) :: nam       !< Namelist
type(geom_type),intent(in) :: geom     !< Geometry
type(bpar_type),intent(in) :: bpar     !< Block parameters

! Local variables
integer :: ib,il0,ic0a,il0i
real(kind_real) :: rhs,rvs
real(kind_real),allocatable :: rh_c0a(:)
character(len=1024),parameter :: subr = 'cmat_setup_sampling'

! Sampling parameters
if (trim(nam%strategy)=='specific_multivariate') then
   do il0=1,geom%nl0
      do ic0a=1,geom%nc0a
         ! Get minimum
         rhs = huge_real
         rvs = huge_real
         do ib=1,bpar%nb
            if (bpar%B_block(ib).and.bpar%nicas_block(ib)) then
               rhs = min(rhs,cmat%blk(ib)%rh(ic0a,il0))
               rvs = min(rvs,cmat%blk(ib)%rv(ic0a,il0))
            end if
         end do

         ! Copy minimum
         do ib=1,bpar%nb
            if (bpar%B_block(ib).and.bpar%nicas_block(ib)) then
               cmat%blk(ib)%rhs(ic0a,il0) = rhs
               cmat%blk(ib)%rvs(ic0a,il0) = rvs
            end if
         end do
      end do
   end do
else
   ! Copy
   do ib=1,bpar%nbe
      if (bpar%B_block(ib).and.bpar%nicas_block(ib)) then
         cmat%blk(ib)%rhs = cmat%blk(ib)%rh
         cmat%blk(ib)%rvs = cmat%blk(ib)%rv
      end if
   end do
end if

select case (trim(nam%draw_type))
case ('random_coast')
   ! More points around coasts
   if (all(geom%gmask_c0a)) call mpl%abort(subr,'random_coast is not relevant if there is no coast')

   ! Allocation
   allocate(rh_c0a(geom%nc0a))

   do il0=1,geom%nl0
      ! Define modulation factor
      il0i = min(il0,geom%nl0i)
      do ic0a=1,geom%nc0a
         if (geom%gmask_c0a(ic0a,il0)) then
            rh_c0a(ic0a) = exp(-geom%mdist_c0a(ic0a,il0i)/nam%Lcoast)
         else
            rh_c0a(ic0a) = 1.0
         end if
      end do
      rh_c0a = nam%rcoast+(1.0-nam%rcoast)*(1.0-rh_c0a)

      ! Apply modulation factor
      do ic0a=1,geom%nc0a
         if (geom%gmask_c0a(ic0a,il0)) then
            do ib=1,bpar%nb
               if (bpar%B_block(ib).and.bpar%nicas_block(ib)) then
                  cmat%blk(ib)%rhs(ic0a,il0) = cmat%blk(ib)%rhs(ic0a,il0)*rh_c0a(ic0a)
               end if
            end do
         end if
      end do
   end do

   ! Release memory
   deallocate(rh_c0a)
end select

end subroutine cmat_setup_sampling

end module type_cmat
