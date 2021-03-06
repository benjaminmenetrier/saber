!----------------------------------------------------------------------
! Subroutine: model_qg_coord
!> Get QG coordinates
!----------------------------------------------------------------------
subroutine model_qg_coord(model,mpl,nam)

implicit none

! Passed variables
class(model_type),intent(inout) :: model !< Model
type(mpl_type),intent(inout) :: mpl      !< MPI data
type(nam_type),intent(in) :: nam         !< Namelist

! Local variables
integer :: img,ilon,ilat,il0
integer :: ncid,nx_id,ny_id,nz_id,lon_id,lat_id,z_id,area_id,lmask_id
real(kind_real),allocatable :: lon(:,:),lat(:,:),z(:),area(:,:),lmask(:,:,:)
character(len=1024),parameter :: subr = 'model_qg_coord'

! Open file and get dimensions
model%ntile = 1
call mpl%ncerr(subr,nf90_open(trim(nam%datadir)//'/grid.nc',nf90_share,ncid))
call mpl%ncerr(subr,nf90_inq_dimid(ncid,'nx',nx_id))
call mpl%ncerr(subr,nf90_inq_dimid(ncid,'ny',ny_id))
call mpl%ncerr(subr,nf90_inquire_dimension(ncid,nx_id,len=model%nlon))
model%nlon = model%nlon+1
call mpl%ncerr(subr,nf90_inquire_dimension(ncid,ny_id,len=model%nlat))
model%nmg = model%nlon*model%nlat
call mpl%ncerr(subr,nf90_inq_dimid(ncid,'nz',nz_id))
call mpl%ncerr(subr,nf90_inquire_dimension(ncid,nz_id,len=model%nlev))

! Allocation
call model%alloc
allocate(lon(model%nlon,model%nlat))
allocate(lat(model%nlon,model%nlat))
allocate(z(model%nlev))
allocate(area(model%nlon,model%nlat))
allocate(lmask(model%nlon,model%nlat,model%nlev))

! Read data and close file
call mpl%ncerr(subr,nf90_inq_varid(ncid,'lon',lon_id))
call mpl%ncerr(subr,nf90_inq_varid(ncid,'lat',lat_id))
call mpl%ncerr(subr,nf90_inq_varid(ncid,'z',z_id))
call mpl%ncerr(subr,nf90_inq_varid(ncid,'area',area_id))
call mpl%ncerr(subr,nf90_inq_varid(ncid,'lmask',lmask_id))
call mpl%ncerr(subr,nf90_get_var(ncid,lon_id,lon(1:model%nlon-1,:)))
call mpl%ncerr(subr,nf90_get_var(ncid,lat_id,lat(1:model%nlon-1,:)))
call mpl%ncerr(subr,nf90_get_var(ncid,z_id,z))
call mpl%ncerr(subr,nf90_get_var(ncid,area_id,area(1:model%nlon-1,:)))
call mpl%ncerr(subr,nf90_get_var(ncid,lmask_id,lmask(1:model%nlon-1,:,:)))
call mpl%ncerr(subr,nf90_close(ncid))

! Add redundant longitude for tests
lon(model%nlon,:) = lon(1,:)
lat(model%nlon,:) = lat(1,:)
area(model%nlon,:) = area(1,:)
lmask(model%nlon,:,:) = lmask(1,:,:)

! Convert to radian
lon = lon*deg2rad
lat = lat*deg2rad

! Model grid
img = 0
do ilon=1,model%nlon
   do ilat=1,model%nlat
      img = img+1
      model%mg_to_tile(img) = 1
      model%mg_to_lon(img) = ilon
      model%mg_to_lat(img) = ilat
      model%lon(img) = lon(ilon,ilat)
      model%lat(img) = lat(ilon,ilat)
      model%area(img) = area(ilon,ilat)/req**2
      do il0=1,model%nl0
        model%mask(img,il0) = (lmask(ilon,ilat,nam%levs(il0))>0.5)
      end do
   end do
end do

! Vertical unit
do il0=1,model%nl0
   model%vunit(1:model%nmg,il0) = z(nam%levs(il0))
end do

! Release memory
deallocate(lon)
deallocate(lat)
deallocate(z)
deallocate(area)
deallocate(lmask)

end subroutine model_qg_coord

!----------------------------------------------------------------------
! Subroutine: model_qg_read
!> Read QG field
!----------------------------------------------------------------------
subroutine model_qg_read(model,mpl,nam,filename,fld)

implicit none

! Passed variables
class(model_type),intent(inout) :: model                        !< Model
type(mpl_type),intent(inout) :: mpl                             !< MPI data
type(nam_type),intent(in) :: nam                                !< Namelist
character(len=*),intent(in) :: filename                         !< File name
real(kind_real),intent(out) :: fld(model%nmga,model%nl0,nam%nv) !< Field

! Local variables
integer :: iv,il0,img,ilon,ilat,ndims
integer :: ncid,fld_id
real(kind_real) :: fld_mg(model%nmg,model%nl0)
real(kind_real),allocatable :: fld_tmp(:,:,:)
character(len=1024),parameter :: subr = 'model_qg_read'

if (mpl%main) then
   ! Allocation
   allocate(fld_tmp(model%nlon,model%nlat,model%nl0))

   ! Open file
   call mpl%ncerr(subr,nf90_open(trim(nam%datadir)//'/'//trim(filename)//'.nc',nf90_nowrite,ncid))
end if

do iv=1,nam%nv
   if (mpl%main) then
      ! Get variable id
      call mpl%ncerr(subr,nf90_inq_varid(ncid,nam%variables(iv),fld_id))

      ! Check field size
      call mpl%ncerr(subr,nf90_inquire_variable(ncid,fld_id,ndims=ndims))

      ! Read data
      select case (ndims)
      case (2)
         ! 2D data
         fld_tmp = mpl%msv%valr
         if (trim(nam%lev2d)=='first') then
            call mpl%ncerr(subr,nf90_get_var(ncid,fld_id,fld_tmp(1:model%nlon-1,:,1),(/1,1/), &
 & (/model%nlon-1,model%nlat/)))
            fld_tmp(model%nlon,:,1) = fld_tmp(1,:,1)
         elseif (trim(nam%lev2d)=='last') then
            call mpl%ncerr(subr,nf90_get_var(ncid,fld_id,fld_tmp(1:model%nlon-1,:,model%nl0),(/1,1/), &
 & (/model%nlon-1,model%nlat/)))
            fld_tmp(model%nlon,:,model%nl0) = fld_tmp(1,:,model%nl0)
         end if
      case (3) 
         ! 3D data
         do il0=1,nam%nl
            call mpl%ncerr(subr,nf90_get_var(ncid,fld_id,fld_tmp(1:model%nlon-1,:,il0),(/1,1,nam%levs(il0)/), &
 & (/model%nlon-1,model%nlat,1/)))
            fld_tmp(model%nlon,:,il0) = fld_tmp(1,:,il0)
         end do
      case default
         call mpl%abort(subr,'wrong number of dimensions for variable '//trim(nam%variables(iv)))
      end select

      ! Pack
      do il0=1,model%nl0
         do img=1,model%nmg
            ilon = model%mg_to_lon(img)
            ilat = model%mg_to_lat(img)
            fld_mg(img,il0) = fld_tmp(ilon,ilat,il0)
         end do
      end do
   end if
   call mpl%glb_to_loc(model%nl0,model%nmga,model%nmg,model%mga_to_mg,fld_mg,fld(:,:,iv))
end do

if (mpl%main) then
   ! Close file
   call mpl%ncerr(subr,nf90_close(ncid))

   ! Release memory
   deallocate(fld_tmp)
end if

end subroutine model_qg_read
