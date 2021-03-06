!----------------------------------------------------------------------
! Module: tools_func
!> Usual functions
! Author: Benjamin Menetrier
! Licensing: this code is distributed under the CeCILL-C license
! Copyright © 2015-... UCAR, CERFACS, METEO-FRANCE and IRIT
!----------------------------------------------------------------------
module tools_func

use iso_c_binding
use atlas_module, only: atlas_geometry
use tools_asa007, only: asa007_cholesky,asa007_syminv
use tools_const, only: pi,deg2rad,rad2deg
use tools_kinds, only: kind_short,kind_real
use tools_repro, only: inf,sup,infeq,small
use type_mpl, only: mpl_type

implicit none

real(kind_real),parameter :: gc2gau = 0.28            ! GC99 support radius to Gaussian Daley length-scale (empirical)
real(kind_real),parameter :: gau2gc = 3.57            ! Gaussian Daley length-scale to GC99 support radius (empirical)
real(kind_real),parameter :: Dmin = 1.0e-12_kind_real ! Minimum tensor diagonal value
real(kind_real),parameter :: condmax = 1.0e3          ! Maximum tensor conditioning number
integer,parameter :: M = 0                            ! Number of implicit iteration for the Matern function (0: Gaussian)

interface
   function c_fletcher32(n,var) bind(c,name='fletcher32') result(hash)
   use iso_c_binding
   integer(kind=c_int32_t) :: n
   integer(kind=c_int16_t) :: var(*)
   integer(kind=c_int32_t) :: hash
   end function c_fletcher32
end interface

private
public :: gc2gau,gau2gc,Dmin,M
public :: fletcher32,lonlatmod,lonlathash,sphere_dist,reduce_arc,lonlat2xyz,xyz2lonlat,vector_product,vector_triple_product, &
 & add,divide,fit_diag,fit_func,fit_lct,lct_d2h,lct_h2r,lct_r2d,check_cond,cholesky,syminv,pseudoinv,histogram

contains

!----------------------------------------------------------------------
! Function: fletcher32
!> Fletcher-32 checksum algorithm
!----------------------------------------------------------------------
function fletcher32(var)

implicit none

! Passed variables
real(kind_real),intent(in) :: var(:) !< Variable

! Returned variable
integer :: fletcher32

! Call C function
fletcher32 = c_fletcher32(size(transfer(var,(/0_kind_short/))),transfer(var,(/0_kind_short/)))

end function fletcher32

!----------------------------------------------------------------------
! Subroutine: lonlatmod
!> Set latitude between -pi/2 and pi/2 and longitude between -pi and pi
!----------------------------------------------------------------------
subroutine lonlatmod(lon,lat)

implicit none

! Passed variables
real(kind_real),intent(inout) :: lon !< Longitude (radians)
real(kind_real),intent(inout) :: lat !< Latitude (radians)

! Check latitude bounds
if (lat>0.5*pi) then
   lat = pi-lat
   lon = lon+pi
elseif (lat<-0.5*pi) then
   lat = -pi-lat
   lon = lon+pi
end if

! Check longitude bounds
if (lon>pi) then
   lon = lon-2.0*pi
elseif (lon<-pi) then
   lon = lon+2.0*pi
end if

end subroutine lonlatmod

!----------------------------------------------------------------------
! Function: lonlathash
!> Define a unique real from a lon/lat pair
!----------------------------------------------------------------------
function lonlathash(lon,lat,il)

implicit none

! Passed variables
real(kind_real),intent(in) :: lon !< Longitude (radians)
real(kind_real),intent(in) :: lat !< Latitude (radians)
integer,intent(in),optional :: il !< Level

! Returned variable
real(kind_real) :: lonlathash

! Local variables
real(kind_real) :: lontmp,lattmp

! Set correct lon/lat
lontmp = lon
lattmp = lat
call lonlatmod(lontmp,lattmp)

! Hash value
lonlathash = aint((lontmp+pi)*1.0e6)+(lattmp+0.5*pi)*1.0e-1
if (present(il)) lonlathash = lonlathash+real(il*1e7,kind_real)

end function lonlathash

!----------------------------------------------------------------------
! Subroutine: sphere_dist
!> Compute the great-circle distance between two points
!----------------------------------------------------------------------
subroutine sphere_dist(lon_i,lat_i,lon_f,lat_f,dist)

implicit none

! Passed variable
real(kind_real),intent(in) :: lon_i !< Initial point longitude (radians)
real(kind_real),intent(in) :: lat_i !< Initial point latitude (radians)
real(kind_real),intent(in) :: lon_f !< Final point longitude (radians)
real(kind_real),intent(in) :: lat_f !< Final point longilatitudetude (radians)
real(kind_real),intent(out) :: dist !< Great-circle distance

! Local variables
type(atlas_geometry) :: ageometry

! Create ATLAS geometry
ageometry = atlas_geometry("UnitSphere")

! Compute distance
dist = ageometry%distance(lon_i*rad2deg,lat_i*rad2deg,lon_f*rad2deg,lat_f*rad2deg)

end subroutine sphere_dist

!----------------------------------------------------------------------
! Subroutine: reduce_arc
!> Reduce arc to a given distance
!----------------------------------------------------------------------
subroutine reduce_arc(lon_i,lat_i,lon_f,lat_f,maxdist,dist)

implicit none

! Passed variables
real(kind_real),intent(in) :: lon_i    !< Initial point longitude (radians)
real(kind_real),intent(in) :: lat_i    !< Initial point latitude (radians)
real(kind_real),intent(inout) :: lon_f !< Final point longitude (radians)
real(kind_real),intent(inout) :: lat_f !< Final point latitude (radians)
real(kind_real),intent(in) :: maxdist  !< Maximum distance
real(kind_real),intent(out) :: dist    !< Effective distance

! Local variable
real(kind_real) :: theta

! Compute distance
call sphere_dist(lon_i,lat_i,lon_f,lat_f,dist)

! Check with the maximum distance
if (sup(dist,maxdist)) then
   ! Compute bearing
   theta = atan2(sin(lon_f-lon_i)*cos(lat_f),cos(lat_i)*sin(lat_f)-sin(lat_i)*cos(lat_f)*cos(lon_f-lon_i))

   ! Reduce distance
   dist = maxdist

   ! Compute new point
   lat_f = asin(sin(lat_i)*cos(dist)+cos(lat_i)*sin(dist)*cos(theta))
   lon_f = lon_i+atan2(sin(theta)*sin(dist)*cos(lat_i),cos(dist)-sin(lat_i)*sin(lat_f))
end if

end subroutine reduce_arc

!----------------------------------------------------------------------
! Subroutine: lonlat2xyz
!> Convert longitude/latitude to cartesian coordinates
!----------------------------------------------------------------------
subroutine lonlat2xyz(mpl,lon,lat,x,y,z)

implicit none

! Passed variables
type(mpl_type),intent(inout) :: mpl !< MPI data
real(kind_real),intent(in) :: lon   !< Longitude (radians)
real(kind_real),intent(in) :: lat   !< Latitude (radians)
real(kind_real),intent(out) :: x    !< X coordinate
real(kind_real),intent(out) :: y    !< Y coordinate
real(kind_real),intent(out) :: z    !< Z coordinate

! Local variables
type(atlas_geometry) :: ageometry
character(len=1024),parameter :: subr = 'lonlat2xyz'

if (mpl%msv%isnot(lat).and.mpl%msv%isnot(lon)) then
   ! Check longitude/latitude
   if (inf(lon,-pi).and.sup(lon,pi)) call mpl%abort(subr,'wrong longitude')
   if (inf(lat,-0.5*pi).and.sup(lat,-0.5*pi)) call mpl%abort(subr,'wrong latitude')

   ! Create ATLAS geometry
   ageometry = atlas_geometry("UnitSphere")

   ! Convert to x/y/z
   call ageometry%lonlat2xyz(lon*rad2deg,lat*rad2deg,x,y,z)
else
   ! Missing values
   x = mpl%msv%valr
   y = mpl%msv%valr
   z = mpl%msv%valr
end if

end subroutine lonlat2xyz

!----------------------------------------------------------------------
! Subroutine: xyz2lonlat
!> Convert longitude/latitude to cartesian coordinates
!----------------------------------------------------------------------
subroutine xyz2lonlat(mpl,x,y,z,lon,lat)

implicit none

! Passed variables
type(mpl_type),intent(in) :: mpl   !< MPI data
real(kind_real),intent(in) :: x    !< X coordinate
real(kind_real),intent(in) :: y    !< Y coordinate
real(kind_real),intent(in) :: z    !< Z coordinate
real(kind_real),intent(out) :: lon !< Longitude (radians)
real(kind_real),intent(out) :: lat !< Latitude (radians)

! Local variables
type(atlas_geometry) :: ageometry

if (mpl%msv%isnot(x).and.mpl%msv%isnot(y).and.mpl%msv%isnot(z)) then
   ! Create ATLAS geometry
   ageometry = atlas_geometry("UnitSphere")

   ! Convert to lon/lat
   call ageometry%xyz2lonlat(x,y,z,lon,lat)

   ! Copy coordinates
   lon = lon*deg2rad
   lat = lat*deg2rad
else
   ! Missing values
   lon = mpl%msv%valr
   lat = mpl%msv%valr
end if

end subroutine xyz2lonlat

!----------------------------------------------------------------------
! Subroutine: vector_product
!> Compute normalized vector product
!----------------------------------------------------------------------
subroutine vector_product(v1,v2,vp)

implicit none

! Passed variables
real(kind_real),intent(in) :: v1(3)  !< First vector
real(kind_real),intent(in) :: v2(3)  !< Second vector
real(kind_real),intent(out) :: vp(3) !< Vector product

! Local variable
real(kind_real) :: r

! Vector product
vp(1) = v1(2)*v2(3)-v1(3)*v2(2)
vp(2) = v1(3)*v2(1)-v1(1)*v2(3)
vp(3) = v1(1)*v2(2)-v1(2)*v2(1)

! Normalization
r = sqrt(sum(vp**2))
if (r>0.0) vp = vp/r

end subroutine vector_product

!----------------------------------------------------------------------
! Subroutine: vector_triple_product
!> Compute vector triple product
!----------------------------------------------------------------------
subroutine vector_triple_product(v1,v2,v3,p,cflag)

implicit none

! Passed variables
real(kind_real),intent(in) :: v1(3) !< First vector
real(kind_real),intent(in) :: v2(3) !< Second vector
real(kind_real),intent(in) :: v3(3) !< Third vector
real(kind_real),intent(out) :: p    !< Triple product
logical,intent(out) :: cflag        !< Confidence flag

! Local variable
integer :: i
real(kind_real) :: terms(6)

! Terms
terms(1) = v1(2)*v2(3)*v3(1)
terms(2) = -v1(3)*v2(2)*v3(1)
terms(3) = v1(3)*v2(1)*v3(2)
terms(4) = -v1(1)*v2(3)*v3(2)
terms(5) = v1(1)*v2(2)*v3(3)
terms(6) = -v1(2)*v2(1)*v3(3)

! Sum
p = sum(terms)

! Confidence flag
cflag = .true.
do i=1,6
   if ((abs(terms(i))>0.0).and.small(p,terms(i))) cflag = .false.
end do

end subroutine vector_triple_product

!----------------------------------------------------------------------
! Subroutine: add
!> Check if value missing and add if not missing
!----------------------------------------------------------------------
subroutine add(mpl,val,cumul,num,wgt)

implicit none

! Passed variables
type(mpl_type),intent(in) :: mpl           !< MPI data
real(kind_real),intent(in) :: val          !< Value to add
real(kind_real),intent(inout) :: cumul     !< Cumul
real(kind_real),intent(inout) :: num       !< Number of values
real(kind_real),intent(in),optional :: wgt !< Weight

! Local variables
real(kind_real) :: lwgt

! Initialize weight
lwgt = 1.0
if (present(wgt)) lwgt = wgt

! Add value to cumul
if (mpl%msv%isnot(val)) then
   cumul = cumul+lwgt*val
   num = num+lwgt
end if

end subroutine add

!----------------------------------------------------------------------
! Subroutine: divide
!> Check if value missing and divide if not missing
!----------------------------------------------------------------------
subroutine divide(mpl,val,num)

implicit none

! Passed variables
type(mpl_type),intent(in) :: mpl     !< MPI data
real(kind_real),intent(inout) :: val !< Value to divide
real(kind_real),intent(in) :: num    !< Divider

! Divide cumul by num
if (abs(num)>0.0) then
   val = val/num
else
   val = mpl%msv%valr
end if

end subroutine divide

!----------------------------------------------------------------------
! Subroutine: fit_diag
!> Compute diagnostic fit function
!----------------------------------------------------------------------
subroutine fit_diag(mpl,nc3,nl0r,nl0,l0rl0_to_l0,disth,distv,coef,rh,rv,fit)

implicit none

! Passed variables
type(mpl_type),intent(inout) :: mpl              !< MPI data
integer,intent(in) :: nc3                        !< Number of classes
integer,intent(in) :: nl0r                       !< Effective number of levels
integer,intent(in) :: nl0                        !< Number of levels
integer,intent(in) :: l0rl0_to_l0(nl0r,nl0)      !< Effective level to level
real(kind_real),intent(in) :: disth(nc3)         !< Horizontal distance
real(kind_real),intent(in) :: distv(nl0,nl0)     !< Vertical distance
real(kind_real),intent(in) :: coef(nl0)          !< Diagonal coefficient
real(kind_real),intent(in) :: rh(nl0)            !< Horizontal support radius
real(kind_real),intent(in) :: rv(nl0)            !< Vertical support radius
real(kind_real),intent(out) :: fit(nc3,nl0r,nl0) !< Fit

! Local variables
integer :: jl0r,jl0,djl0,il0,kl0r,kl0,ic3,jc3,djc3,kc3,ip,jp,np,np_new,nc3max
integer :: plist(nc3*nl0r,2),plist_new(nc3*nl0r,2)
real(kind_real) :: rhsq,rvsq,distvsq,disttest,rhmax
real(kind_real) :: predistnorm(-1:1,nc3,-1:1,nl0),distnorm(nc3,nl0r)
logical :: add_to_front

! Initialization
fit = 0.0

! Find maximum class
rhmax = maxval(rh)
nc3max = 0
do ic3=1,nc3
   if (disth(ic3)>rhmax) exit
   nc3max = ic3
end do

! Precompute normalized local distances
predistnorm = 1.0
do il0=1,nl0
   do djl0=-1,1
      ! Level index
      jl0 = max(1,min(il0+djl0,nl0))

      ! Averaged fit parameters
      if (mpl%msv%isnot(rh(il0)).and.mpl%msv%isnot(rh(jl0))) then
         rhsq = 0.5*(rh(il0)**2+rh(jl0)**2)
      else
         rhsq = 0.0
      end if
      if (mpl%msv%isnot(rv(il0)).and.mpl%msv%isnot(rv(jl0))) then
         rvsq = 0.5*(rv(il0)**2+rv(jl0)**2)
      else
         rvsq = 0.0
      end if

      ! Squared vertical distance
      if (rvsq>0.0) then
         distvsq = distv(jl0,il0)**2/rvsq
      elseif (il0/=jl0) then
         distvsq = 1.0
      else
         distvsq = 0.0
      end if

      do ic3=1,nc3max
         do djc3=-1,1
            ! Horizontal index
            jc3 = max(1,min(ic3+djc3,nc3max))

            ! Squared normalized distance initialization (squared vertical component)
            predistnorm(djc3,ic3,djl0,il0) = distvsq

            ! Add squared horizontal component to squared normalized distance
            if (rhsq>0.0) then
               predistnorm(djc3,ic3,djl0,il0) = predistnorm(djc3,ic3,djl0,il0)+(disth(jc3)-disth(ic3))**2/rhsq
            elseif (ic3/=jc3) then
               predistnorm(djc3,ic3,djl0,il0) = predistnorm(djc3,ic3,djl0,il0)+1.0
            end if

            ! Get normalized distance
            predistnorm(djc3,ic3,djl0,il0) = sqrt(predistnorm(djc3,ic3,djl0,il0))
         end do
      end do
   end do
end do

! Compte fit
do il0=1,nl0
   ! Initialize the front
   np = 1
   plist = mpl%msv%vali
   plist(1,1) = 1
   do jl0r=1,nl0r
      if (l0rl0_to_l0(jl0r,il0)==il0) plist(1,2) = jl0r
   end do
   distnorm = 1.0
   distnorm(plist(1,1),plist(1,2)) = 0.0

   do while (np>0)
      ! Propagate the front
      np_new = 0

      do ip=1,np
         ! Indices of the central point
         jc3 = plist(ip,1)
         jl0r = plist(ip,2)
         jl0 = l0rl0_to_l0(jl0r,il0)

         ! Loop over neighbors              
         do kc3=max(jc3-1,1),min(jc3+1,nc3max)
            do kl0r=max(jl0r-1,1),min(jl0r+1,nl0r)
               kl0 = l0rl0_to_l0(kl0r,il0)
               disttest = distnorm(jc3,jl0r)+predistnorm(kc3-jc3,jc3,kl0-jl0,jl0)

               if (disttest<1.0) then
                  ! Point is inside the support
                  if (disttest<distnorm(kc3,kl0r)) then
                     ! Update distance
                     distnorm(kc3,kl0r) = disttest

                     ! Check if the point should be added to the front (avoid duplicates)
                     add_to_front = .true.
                     do jp=1,np_new
                        if ((plist_new(jp,1)==kc3).and.(plist_new(jp,2)==kl0r)) then
                           add_to_front = .false.
                           exit
                        end if
                     end do

                     if (add_to_front) then
                        ! Add point to the front
                        np_new = np_new+1
                        plist_new(np_new,1) = kc3
                        plist_new(np_new,2) = kl0r
                     end if
                  end if
               end if
            end do
         end do
      end do

      ! Copy new front
      np = np_new
      plist(1:np,:) = plist_new(1:np,:)
   end do

   do jl0r=1,nl0r
      ! Level index
      jl0 = l0rl0_to_l0(jl0r,il0)

      do jc3=1,nc3
         ! Fit function
         fit(jc3,jl0r,il0) = fit_func(mpl,distnorm(jc3,jl0r))
      end do
   end do
end do

! Diagonal coefficient
do il0=1,nl0
   do jl0r=1,nl0r
      jl0 = l0rl0_to_l0(jl0r,il0)
      fit(:,jl0r,il0) = fit(:,jl0r,il0)*sqrt(coef(il0)*coef(jl0))
   end do
end do

! Set to missing values if no value available
do il0=1,nl0
   if (mpl%msv%is(coef(il0)).or.mpl%msv%is(rh(il0)).or.mpl%msv%is(rv(il0))) fit(:,:,il0) = mpl%msv%valr
   do jl0r=1,nl0r
      jl0 = l0rl0_to_l0(jl0r,il0)
      if (mpl%msv%is(coef(jl0)).or.mpl%msv%is(rh(jl0)).or.mpl%msv%is(rv(jl0))) fit(:,jl0r,il0) = mpl%msv%valr
   end do
end do

end subroutine fit_diag

!----------------------------------------------------------------------
! Function: gc99
!> Gaspari and Cohn (1999) function, with the support radius as a parameter
!----------------------------------------------------------------------
function gc99(distnorm)

! Passed variables
real(kind_real),intent(in) :: distnorm !< Normalized distance

! Returned variable
real(kind_real) :: gc99

! Gaspari and Cohn (1999) function
if (distnorm<0.5) then
   gc99 = -8.0*distnorm**5+8.0*distnorm**4+5.0*distnorm**3-20.0/3.0*distnorm**2+1.0
else if (distnorm<1.0) then
   gc99 = 8.0/3.0*distnorm**5-8.0*distnorm**4+5.0*distnorm**3+20.0/3.0*distnorm**2-10.0*distnorm+4.0-1.0/(3.0*distnorm)
else
   gc99 = 0.0
end if

end function gc99

!----------------------------------------------------------------------
! Function: fit_func
!> Fit_function
!----------------------------------------------------------------------
function fit_func(mpl,distnorm)

! Passed variables
type(mpl_type),intent(inout) :: mpl    !< MPI data
real(kind_real),intent(in) :: distnorm !< Normalized distance

! Returned variable
real(kind_real) :: fit_func

! Local variables
character(len=1024),parameter :: subr = 'fit_func'

! Distance check bound
if (distnorm<0.0) call mpl%abort(subr,'negative normalized distance')

! Gaspari and Cohn (1999) function
fit_func = gc99(distnorm)

! Enforce positivity
fit_func = max(fit_func,0.0_kind_real)

end function fit_func

!----------------------------------------------------------------------
! Subroutine: fit_lct
!> LCT fit
!----------------------------------------------------------------------
subroutine fit_lct(mpl,nc,nl0,dxsq,dysq,dxdy,dzsq,dmask,nscales,D,coef,fit)

implicit none

! Passed variables
type(mpl_type),intent(inout) :: mpl         !< MPI data
integer,intent(in) :: nc                    !< Number of classes
integer,intent(in) :: nl0                   !< Number of levels
real(kind_real),intent(in) :: dxsq(nc,nl0)  !< Zonal separation squared
real(kind_real),intent(in) :: dysq(nc,nl0)  !< Meridian separation squared
real(kind_real),intent(in) :: dxdy(nc,nl0)  !< Zonal x meridian separations product
real(kind_real),intent(in) :: dzsq(nc,nl0)  !< Vertical separation squared
logical,intent(in) :: dmask(nc,nl0)         !< Mask
integer,intent(in) :: nscales               !< Number of LCT scales
real(kind_real),intent(in) :: D(4,nscales)  !< LCT components
real(kind_real),intent(in) :: coef(nscales) !< LCT coefficients
real(kind_real),intent(out) :: fit(nc,nl0)  !< Fit

! Local variables
integer :: jl0,jc3,iscales
real(kind_real) :: Dcoef(nscales),D11,D22,D33,D12,H11,H22,H33,H12,rsq

! Initialization
fit = mpl%msv%valr

! Coefficients
Dcoef = max(Dmin,min(coef,1.0_kind_real))
Dcoef = Dcoef/sum(Dcoef)

do iscales=1,nscales
   ! Ensure positive-definiteness of D
   D11 = max(Dmin,D(1,iscales))
   D22 = max(Dmin,D(2,iscales))
   if (nl0>1) then
      D33 = max(Dmin,D(3,iscales))
   else
      D33 = 0.0
   end if
   D12 = sqrt(D11*D22)*max(-1.0_kind_real+Dmin,min(D(4,iscales),1.0_kind_real-Dmin))

   ! Inverse D to get H
   call lct_d2h(mpl,D11,D22,D33,D12,H11,H22,H33,H12)

   ! Homogeneous anisotropic approximation
   do jl0=1,nl0
      do jc3=1,nc
         if (dmask(jc3,jl0)) then
            ! Initialization
            if (iscales==1) fit(jc3,jl0) = 0.0

            ! Squared distance
            rsq = H11*dxsq(jc3,jl0)+H22*dysq(jc3,jl0)+H33*dzsq(jc3,jl0)+2.0*H12*dxdy(jc3,jl0)

            if (M==0) then
               ! Gaussian function
               if (rsq<40.0) fit(jc3,jl0) = fit(jc3,jl0)+Dcoef(iscales)*exp(-0.5*rsq)
            else
               ! Matern function
               fit(jc3,jl0) = fit(jc3,jl0)+Dcoef(iscales)*matern(mpl,M,sqrt(rsq))
            end if
         end if
      end do
   end do
end do

end subroutine fit_lct

!----------------------------------------------------------------------
! Subroutine: lct_d2h
!> From D (Daley tensor) to H (local correlation tensor)
!----------------------------------------------------------------------
subroutine lct_d2h(mpl,D11,D22,D33,D12,H11,H22,H33,H12)

implicit none

! Passed variables
type(mpl_type),intent(inout) :: mpl!< MPI data
real(kind_real),intent(in) :: D11  !< Daley tensor component 11
real(kind_real),intent(in) :: D22  !< Daley tensor component 22
real(kind_real),intent(in) :: D33  !< Daley tensor component 33
real(kind_real),intent(in) :: D12  !< Daley tensor component 12
real(kind_real),intent(out) :: H11 !< Local correlation tensor component 11
real(kind_real),intent(out) :: H22 !< Local correlation tensor component 22
real(kind_real),intent(out) :: H33 !< Local correlation tensor component 33
real(kind_real),intent(out) :: H12 !< Local correlation tensor component 12

! Local variables
real(kind_real) :: det
character(len=1024),parameter :: subr = 'lct_d2h'

! Compute horizontal determinant
det = D11*D22-D12**2

! Inverse D to get H
if (det>0.0) then
   H11 = D22/det
   H22 = D11/det
   H12 = -D12/det
else
   call mpl%abort(subr,'non-invertible tensor')
end if
if (D33>0.0) then
   H33 = 1.0/D33
else
   H33 = 0.0
end if

end subroutine lct_d2h

!----------------------------------------------------------------------
! Subroutine: lct_h2r
!> From H (local correlation tensor) to support radii
!----------------------------------------------------------------------
subroutine lct_h2r(mpl,H11,H22,H33,H12,rh,rv)

implicit none

! Passed variables
type(mpl_type),intent(inout) :: mpl !< MPI data
real(kind_real),intent(in) :: H11   !< Local correlation tensor component 11
real(kind_real),intent(in) :: H22   !< Local correlation tensor component 22
real(kind_real),intent(in) :: H33   !< Local correlation tensor component 33
real(kind_real),intent(in) :: H12   !< Local correlation tensor component 12
real(kind_real),intent(out) :: rh   !< Horizontal support radius
real(kind_real),intent(out) :: rv   !< Vertical support radius

! Local variables
real(kind_real) :: tr,det,diff
character(len=1024),parameter :: subr = 'lct_h2r'

! Check diagonal positivity
if ((H11<0.0).or.(H22<0.0)) call mpl%abort(subr,'negative diagonal LCT coefficients')

! Compute horizontal trace
tr = H11+H22

! Compute horizontal determinant
det = H11*H22-H12**2

! Compute horizontal support radius
diff = 0.25*(H11-H22)**2+H12**2
if ((det>0.0).and..not.(diff<0.0)) then
   if (0.5*tr>sqrt(diff)) then
      rh = gau2gc/sqrt(0.5*tr-sqrt(diff))
   else
      call mpl%abort(subr,'non positive-definite LCT (eigenvalue)')
   end if
else
   call mpl%abort(subr,'non positive-definite LCT (determinant)')
end if

! Compute vertical support radius
if (H33>0.0) then
   rv = gau2gc/sqrt(H33)
else
   rv = 0.0
end if

end subroutine lct_h2r

!----------------------------------------------------------------------
! Subroutine: lct_r2d
!> From support radius to Daley tensor diagonal element
!----------------------------------------------------------------------
subroutine lct_r2d(r,D)

implicit none

! Passed variables
real(kind_real),intent(in) :: r  !< Support radius
real(kind_real),intent(out) :: D !< Daley tensor diagonal element

! Convert from support radius to Daley length-scale and square
D = (gc2gau*r)**2

end subroutine lct_r2d

!----------------------------------------------------------------------
! Subroutine: check_cond
!> Check tensor conditioning
!----------------------------------------------------------------------
subroutine check_cond(d1,d2,nod,valid)

implicit none

! Passed variables
real(kind_real),intent(in) :: d1  !< First diagonal coefficient
real(kind_real),intent(in) :: d2  !< Second diagonal coefficient
real(kind_real),intent(in) :: nod !< Normalized off-diagonal coefficient
logical,intent(out) :: valid      !< Conditioning validity

! Local variables
real(kind_real) :: det,tr,diff,ev1,ev2

! Compute trace and determinant
tr = d1+d2
det = d1*d2*(1.0-nod**2)
diff = 0.25*(d1-d2)**2+d1*d2*nod**2

if ((det>0.0).and..not.(diff<0.0)) then
   ! Compute eigenvalues
   ev1 = 0.5*tr+sqrt(diff)
   ev2 = 0.5*tr-sqrt(diff)

   if (ev2>0.0) then
      ! Check conditioning
      valid = inf(ev1,condmax*ev2)
   else
      ! Lowest negative eigenvalue is negative
      valid = .false.
   end if
else
   ! Non-positive definite tensor
   valid = .false.
end if

end subroutine check_cond

!----------------------------------------------------------------------
! Function: matern
!> Compute the normalized diffusion function from eq. (55) of Mirouze and Weaver (2013), for the 3d case (d = 3)
!----------------------------------------------------------------------
real(kind_real) function matern(mpl,M,x)

implicit none

! Passed variables
type(mpl_type),intent(inout) :: mpl !< MPI data
integer,intent(in) :: M             !< Matern function order
real(kind_real),intent(in) :: x     !< Argument

! Local variables
integer :: j
real(kind_real) :: xtmp,beta
character(len=1024),parameter :: subr = 'matern'

! Check
if (M<2) call mpl%abort(subr,'M should be larger than 2')
if (mod(M,2)>0) call mpl%abort(subr,'M should be even')

! Initialization
matern = 0.0
beta = 1.0
xtmp = x*sqrt(real(2*M-5,kind_real))

do j=0,M-3
   ! Update sum
   matern = matern+beta*(xtmp)**(M-2-j)

   ! Update beta
   beta = beta*real((j+1+M-2)*(-j+M-2),kind_real)/real(2*(j+1),kind_real)
end do

! Last term and normalization
matern = matern/beta+1.0

! Exponential factor
matern = matern*exp(-xtmp)

end function matern

!----------------------------------------------------------------------
! Subroutine: cholesky
!> Compute cholesky decomposition
! Author: Original FORTRAN77 version by Michael Healy, modifications by AJ Miller, FORTRAN90 version by John Burkardt.
!----------------------------------------------------------------------
subroutine cholesky(mpl,n,a,u,ierr)

implicit none

! Passed variables
type(mpl_type),intent(inout) :: mpl   !< MPI data
integer,intent(in) :: n               !< Matrix rank
real(kind_real),intent(in) :: a(n,n)  !< Matrix
real(kind_real),intent(out) :: u(n,n) !< Matrix square-root
integer,intent(out) :: ierr           !< Error status

! Local variables
integer :: nn,i,j,ij
real(kind_real),allocatable :: apack(:),upack(:)

! Allocation
nn = (n*(n+1))/2
allocate(apack(nn))
allocate(upack(nn))

! Pack matrix
ij = 0
do i=1,n
   do j=1,i
      ij = ij+1
      apack(ij) = a(i,j)
   end do
end do

! Cholesky decomposition
call asa007_cholesky(mpl,n,nn,apack,upack,ierr)

! Unpack matrix
ij = 0
u = 0.0
do i=1,n
   do j=1,i
      ij = ij+1
      u(i,j) = upack(ij)
   end do
end do

! Release memory
deallocate(apack)
deallocate(upack)

end subroutine cholesky

!----------------------------------------------------------------------
! Subroutine: syminv
!> Compute inverse of a symmetric matrix
! Author: Original FORTRAN77 version by Michael Healy, modifications by AJ Miller, FORTRAN90 version by John Burkardt.
!----------------------------------------------------------------------
subroutine syminv(mpl,n,a,c,ierr)

implicit none

! Passed variables
type(mpl_type),intent(inout) :: mpl   !< MPI data
integer,intent(in) :: n               !< Matrix rank
real(kind_real),intent(in) :: a(n,n)  !< Matrix
real(kind_real),intent(out) :: c(n,n) !< Matrix inverse
integer,intent(out) :: ierr           !< Error status

! Local variables
integer :: nn,i,j,ij
real(kind_real),allocatable :: apack(:),cpack(:)

! Allocation
nn = (n*(n+1))/2
allocate(apack(nn))
allocate(cpack(nn))

! Pack matrix
ij = 0
do i=1,n
   do j=1,i
      ij = ij+1
      apack(ij) = a(i,j)
   end do
end do

! Matrix inversion
call asa007_syminv(mpl,n,nn,apack,cpack,ierr)

! Unpack matrix
ij = 0
do i=1,n
   do j=1,i
      ij = ij+1
      c(i,j) = cpack(ij)
      c(j,i) = c(i,j)
   end do
end do

! Release memory
deallocate(apack)
deallocate(cpack)

end subroutine syminv

!----------------------------------------------------------------------
! Subroutine: pseudoinv
! Purpose: Compute pseudo inverse of a symmetric matrix.
! Author: This routine is from WRFDA.
!----------------------------------------------------------------------
subroutine pseudoinv(mpl,n,a,c,ierr,mmax,var_th)

implicit none

! Passed variables
type(mpl_type),intent(inout) :: mpl           ! MPI data
integer,intent(in) :: n                       ! Matrix rank
real(kind_real),intent(in) :: a(n,n)          ! Matrix
real(kind_real),intent(out) :: c(n,n)         ! Matrix inverse
integer,intent(out) :: ierr                   ! Error status
integer,intent(in),optional :: mmax           ! Dominant mode
real(kind_real),intent(in),optional :: var_th ! Variance threshold

! Local variables
integer :: k,k2,m,lmmax
real(kind_real),allocatable :: work(:,:),evec(:,:),eval(:),laminvet(:,:)
real(kind_real),allocatable :: summ,total_variance,cumul_variance
character(len=1024),parameter :: subr = 'pseudoinv'

! Allocation
allocate(work(n,n))
allocate(evec(n,n))
allocate(eval(n))
allocate(laminvet(n,n))

! Initialization
work = a
laminvet = 0.0

! EOF decomposition
call da_eof_decomposition(n,work,evec,eval,ierr)

! Select dominant mode
if (present(mmax)) then
   ! Input argument
   lmmax = mmax
else
   if (present(var_th)) then
      ! Based on variance threshold 
      summ = 0.0
      do m=1,n
         summ = summ+eval(m)
      end do
      total_variance = summ
      cumul_variance = 0.0
      lmmax = n
      do m=1,n
         cumul_variance = cumul_variance+eval(m)/total_variance
         if (cumul_variance>1.0-var_th ) then
            lmmax = m-1
            exit
         end if
      end do
   else
      call mpl%abort(subr,'either dominant mode or variance threshold should be specified')
   end if
end if
if (lmmax>n) call mpl%abort(subr,'dominant mode should smaller than the matrix rank')

! Lam{-1} . E^T:
do k=1,n
   do m=1,lmmax
      laminvet(m,k) = evec(k,m)/eval(m)
   end do
end do

! <a,a>^{-1} = E . Lam{-1} . E^T:
do k=1,n
   do k2=1,k
      summ = 0.0
      do m=1,n
         summ = summ+evec(k,m)*laminvet(m,k2)
      end do
      c(k,k2) = summ
   end do
end do

! Symmetry
do k=1,n
   do k2=k+1,n
      c(k,k2) = c(k2,k)
   end do
end do

! Release memory
deallocate(work)
deallocate(evec)
deallocate(eval)
deallocate(laminvet)

end subroutine pseudoinv

!----------------------------------------------------------------------
! Subroutine: da_eof_decomposition
! Purpose: Compute eigenvectors E and eigenvalues L of covariance matrix.
!!         B_{x} defined by equation:  E^{T} B_{x} E = L, given input kz x kz matrix.
! Author: This routine is from WRFDA.
!----------------------------------------------------------------------
subroutine da_eof_decomposition(kz,bx,e,l,ierr)

implicit none

! Passed variables
integer, intent(in)  :: kz                  ! Dimension of error matrix
real(kind_real),intent(in) :: bx(1:kz,1:kz) ! Vert. background error
real(kind_real),intent(out) :: e(1:kz,1:kz) ! Eigenvectors of Bx
real(kind_real),intent(out) :: l(1:kz)      ! Eigenvalues of Bx
integer,intent(out) :: ierr                 ! Error status

! Local variables
integer :: work,m,info
real(kind_real) :: work_array(1:3*kz-1),ecopy(1:kz,1:kz),lcopy(1:kz)

! Initialization
work = 3*kz-1
ecopy = bx
lcopy = 0.0

! Perform global eigenvalue decomposition using LAPACK software
call dsyev('V','U',kz,ecopy,kz,lcopy,work_array,work,ierr)

! Swap order of eigenvalues, vectors so 1st is one with most variance
do m=1,kz
   l(m) = lcopy(kz+1-m)
   e(1:kz,m) = ecopy(1:kz,kz+1-m)
end do

end subroutine da_eof_decomposition

!----------------------------------------------------------------------
! Subroutine: histogram
!> Compute bins and histogram from a list of values
!----------------------------------------------------------------------
subroutine histogram(mpl,nlist,list,nbins,histmin,histmax,bins,hist)

implicit none

! Passed variables
type(mpl_type),intent(inout) :: mpl          !< MPI data
integer,intent(in) :: nlist                  !< List size
real(kind_real),intent(in) :: list(nlist)    !< List
integer,intent(in) :: nbins                  !< Number of bins
real(kind_real),intent(in) :: histmin        !< Histogram minimum
real(kind_real),intent(in) :: histmax        !< Histogram maximum
real(kind_real),intent(out) :: bins(nbins+1) !< Bins
real(kind_real),intent(out) :: hist(nbins)   !< Histogram

! Local variables
integer :: ibins,ilist
real(kind_real) :: delta
logical :: found
character(len=1024) :: subr = 'histogram'

! Check data
if (nbins<=0) call mpl%abort(subr,'the number of bins should be positive')
if (histmax>histmin) then
   if (minval(list,mask=mpl%msv%isnot(list))<histmin) call mpl%abort(subr,'values below histogram minimum')
   if (maxval(list,mask=mpl%msv%isnot(list))>histmax) call mpl%abort(subr,'values over histogram maximum')

   ! Compute bins
   delta = (histmax-histmin)/real(nbins,kind_real)
   bins(1) = histmin
   do ibins=2,nbins
      bins(ibins) = histmin+real(ibins-1,kind_real)*delta
   end do
   bins(nbins+1) = histmax

   ! Extend first and last bins
   bins(1) = bins(1)-1.0e-6*delta
   bins(nbins+1) = bins(nbins+1)+1.0e-6*delta

   ! Compute histogram
   hist = 0.0
   do ilist=1,nlist
      if (mpl%msv%isnot(list(ilist))) then
         ibins = 0
         found = .false.
         do while (.not.found)
            ibins = ibins+1
            if (ibins>nbins) call mpl%abort(subr,'bin not found')
            if (infeq(bins(ibins),list(ilist)).and.inf(list(ilist),bins(ibins+1))) then
               hist(ibins) = hist(ibins)+1.0
               found = .true.
            end if
         end do
      end if
   end do
   if (abs(sum(hist)-real(count(mpl%msv%isnot(list)),kind_real))>0.5) &
 & call mpl%abort(subr,'histogram sum is not equal to the number of valid elements')
else
   bins = mpl%msv%valr
   hist = 0.0
end if

end subroutine histogram

end module tools_func
