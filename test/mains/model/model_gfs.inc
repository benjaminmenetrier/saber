!----------------------------------------------------------------------
! Subroutine: model_gfs_coord
!> Get GFS coordinates
!----------------------------------------------------------------------
subroutine model_gfs_coord(model,mpl,nam)

implicit none

! Passed variables
class(model_type),intent(inout) :: model !< Model
type(mpl_type),intent(inout) :: mpl      !< MPI data
type(nam_type),intent(in) :: nam         !< Namelist

! Local variables
integer :: img,ilon,ilat
integer :: ncid,nlon_id,nlat_id,nlev_id,lon_id,lat_id,a_id,b_id
real(kind_real),allocatable :: lon(:),lat(:),a(:),b(:)
character(len=1024),parameter :: subr = 'model_gfs_coord'

! Open file and get dimensions
model%ntile = 1
call mpl%ncerr(subr,nf90_open(trim(nam%datadir)//'/grid.nc',nf90_share,ncid))
call mpl%ncerr(subr,nf90_inq_dimid(ncid,'longitude',nlon_id))
call mpl%ncerr(subr,nf90_inq_dimid(ncid,'latitude',nlat_id))
call mpl%ncerr(subr,nf90_inquire_dimension(ncid,nlon_id,len=model%nlon))
call mpl%ncerr(subr,nf90_inquire_dimension(ncid,nlat_id,len=model%nlat))
model%nmg = model%nlon*model%nlat
call mpl%ncerr(subr,nf90_inq_dimid(ncid,'level',nlev_id))
call mpl%ncerr(subr,nf90_inquire_dimension(ncid,nlev_id,len=model%nlev))

! Allocation
call model%alloc
allocate(lon(model%nlon))
allocate(lat(model%nlat))
allocate(a(model%nlev+1))
allocate(b(model%nlev+1))

! Read data and close file
call mpl%ncerr(subr,nf90_inq_varid(ncid,'longitude',lon_id))
call mpl%ncerr(subr,nf90_inq_varid(ncid,'latitude',lat_id))
call mpl%ncerr(subr,nf90_inq_varid(ncid,'ak',a_id))
call mpl%ncerr(subr,nf90_inq_varid(ncid,'bk',b_id))
call mpl%ncerr(subr,nf90_get_var(ncid,lon_id,lon))
call mpl%ncerr(subr,nf90_get_var(ncid,lat_id,lat))
call mpl%ncerr(subr,nf90_get_var(ncid,a_id,a))
call mpl%ncerr(subr,nf90_get_var(ncid,b_id,b))
call mpl%ncerr(subr,nf90_close(ncid))

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
      model%lon(img) = lon(ilon)
      model%lat(img) = lat(ilat)
   end do
end do
model%area = 4.0*pi/real(model%nmg,kind_real)
model%mask = .true.

! Vertical unit
do img=1,model%nmg
   if (nam%logpres) then
      model%vunit(img,1:nam%nl) = log(0.5*(a(nam%levs(1:nam%nl))+a(nam%levs(1:nam%nl)+1)) &
 & +0.5*(b(nam%levs(1:nam%nl))+b(nam%levs(1:nam%nl)+1))*ps)
      if (model%nl0>nam%nl) model%vunit(img,model%nl0) = log(ps)
   else
      model%vunit(img,:) = real(nam%levs(1:model%nl0),kind_real)
   end if
end do

! Release memory
deallocate(lon)
deallocate(lat)
deallocate(a)
deallocate(b)

end subroutine model_gfs_coord

!----------------------------------------------------------------------
! Subroutine: model_gfs_read
!> Read GFS field
!----------------------------------------------------------------------
subroutine model_gfs_read(model,mpl,nam,filename,fld)

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
character(len=1024),parameter :: subr = 'model_gfs_read'

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


      ! Read data
      select case (ndims)
      case (2)
         ! 2D data
         fld_tmp = mpl%msv%valr
         if (trim(nam%lev2d)=='first') then
            call mpl%ncerr(subr,nf90_get_var(ncid,fld_id,fld_tmp(:,:,1),(/1,1/),(/model%nlon,model%nlat/)))
         elseif (trim(nam%lev2d)=='last') then
            call mpl%ncerr(subr,nf90_get_var(ncid,fld_id,fld_tmp(:,:,model%nl0),(/1,1/),(/model%nlon,model%nlat/)))
         end if
      case (3) 
         ! 3D data
         do il0=1,nam%nl
            call mpl%ncerr(subr,nf90_get_var(ncid,fld_id,fld_tmp(:,:,il0),(/1,1,nam%levs(il0)/),(/model%nlon,model%nlat,1/)))
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

end subroutine model_gfs_read
