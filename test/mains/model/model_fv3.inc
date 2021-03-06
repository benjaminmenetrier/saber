!----------------------------------------------------------------------
! Subroutine: model_fv3_coord
!> Get FV3 coordinates
!----------------------------------------------------------------------
subroutine model_fv3_coord(model,mpl,nam)

implicit none

! Passed variables
class(model_type),intent(inout) :: model !< Model
type(mpl_type),intent(inout) :: mpl      !< MPI data
type(nam_type),intent(in) :: nam         !< Namelist

! Local variables
integer :: img,ilon,ilat,itile,il0
integer :: ncid,nlon_id,nlat_id,ntile_id,nlev_id,lon_id,lat_id,a_id,b_id
real(kind_real) :: sigmaup,sigmadn,sigma
real(kind_real),allocatable :: lon(:,:,:),lat(:,:,:),a(:),b(:)
character(len=1024),parameter :: subr = 'model_fv3_coord'

! Open file and get dimensions
call mpl%ncerr(subr,nf90_open(trim(nam%datadir)//'/grid.nc',nf90_share,ncid))
call mpl%ncerr(subr,nf90_inq_dimid(ncid,'fxdim',nlon_id))
call mpl%ncerr(subr,nf90_inq_dimid(ncid,'fydim',nlat_id))
call mpl%ncerr(subr,nf90_inq_dimid(ncid,'ntile',ntile_id))
call mpl%ncerr(subr,nf90_inquire_dimension(ncid,nlon_id,len=model%nlon))
call mpl%ncerr(subr,nf90_inquire_dimension(ncid,nlat_id,len=model%nlat))
call mpl%ncerr(subr,nf90_inquire_dimension(ncid,ntile_id,len=model%ntile))
model%nmg = model%nlon*model%nlat*model%ntile
call mpl%ncerr(subr,nf90_inq_dimid(ncid,'xaxis_1',nlev_id))
call mpl%ncerr(subr,nf90_inquire_dimension(ncid,nlev_id,len=model%nlev))
model%nlev = model%nlev-1

! Allocation
call model%alloc
allocate(lon(model%nlon,model%nlat,model%ntile))
allocate(lat(model%nlon,model%nlat,model%ntile))
allocate(a(model%nlev+1))
allocate(b(model%nlev+1))

! Read data and close file
call mpl%ncerr(subr,nf90_inq_varid(ncid,'flons',lon_id))
call mpl%ncerr(subr,nf90_inq_varid(ncid,'flats',lat_id))
call mpl%ncerr(subr,nf90_inq_varid(ncid,'ak',a_id))
call mpl%ncerr(subr,nf90_inq_varid(ncid,'bk',b_id))
call mpl%ncerr(subr,nf90_get_var(ncid,lon_id,lon))
call mpl%ncerr(subr,nf90_get_var(ncid,lat_id,lat))
call mpl%ncerr(subr,nf90_get_var(ncid,a_id,a,(/1,1/),(/model%nlev+1,1/)))
call mpl%ncerr(subr,nf90_get_var(ncid,b_id,b,(/1,1/),(/model%nlev+1,1/)))
call mpl%ncerr(subr,nf90_close(ncid))

! Model grid
img = 0
do itile=1,model%ntile
   do ilon=1,model%nlon
      do ilat=1,model%nlat
         img = img+1
         model%mg_to_tile(img) = itile
         model%mg_to_lon(img) = ilon
         model%mg_to_lat(img) = ilat
         model%lon(img) = lon(ilon,ilat,itile)
         model%lat(img) = lat(ilon,ilat,itile)
         call lonlatmod(model%lon(img),model%lat(img))
      end do
   end do
end do
model%area = 4.0*pi/real(model%nmg,kind_real)
model%mask = .true.

! Vertical unit
do il0=1,model%nl0
   if (il0<nam%nl) then
      sigmaup = a(nam%levs(il0)+1)/ps+b(nam%levs(il0)+1)
      sigmadn = a(nam%levs(il0))/ps+b(nam%levs(il0))
      sigma = 0.5*(sigmaup+sigmadn)
   else
      sigma = 1.0
   end if
   if (nam%logpres) then
      model%vunit(:,il0) = log(sigma*ps)
      if (model%nl0>nam%nl) model%vunit(:,model%nl0) = log(ps)
   else
      model%vunit(:,il0) = sigma
   end if
end do

! Release memory
deallocate(lon)
deallocate(lat)
deallocate(a)
deallocate(b)

end subroutine model_fv3_coord

!----------------------------------------------------------------------
! Subroutine: model_fv3_read
!> Read FV3 field
!----------------------------------------------------------------------
subroutine model_fv3_read(model,mpl,nam,filename,fld)

implicit none

! Passed variables
class(model_type),intent(inout) :: model                        !< Model
type(mpl_type),intent(inout) :: mpl                             !< MPI data
type(nam_type),intent(in) :: nam                                !< Namelist
character(len=*),intent(in) :: filename                         !< File name
real(kind_real),intent(out) :: fld(model%nmga,model%nl0,nam%nv) !< Field

! Local variables
integer :: iv,il0,imgt,img,ilon,ilat
integer :: ncid,fld_id,dimids(4),zlen
real(kind_real) :: fld_mgt(model%nmgt,model%nl0)
real(kind_real),allocatable :: fld_tmp(:,:,:)
character(len=1) :: ctile
character(len=1024),parameter :: subr = 'model_fv3_read'

if (model%ioproc(model%mytile)==mpl%myproc) then
   ! Allocation
   allocate(fld_tmp(model%nlon,model%nlat,model%nl0))

   ! Open file
   write(ctile,'(i1.1)') model%mytile
   call mpl%ncerr(subr,nf90_open(trim(nam%datadir)//'/'//trim(filename)//'_tile'//ctile//'.nc',nf90_nowrite,ncid))
end if

do iv=1,nam%nv
   if (model%ioproc(model%mytile)==mpl%myproc) then
      ! Get variable id
      call mpl%ncerr(subr,nf90_inq_varid(ncid,nam%variables(iv),fld_id))

      ! Check field size
      call mpl%ncerr(subr,nf90_inquire_variable(ncid,fld_id,dimids=dimids))
      call mpl%ncerr(subr,nf90_inquire_dimension(ncid,dimids(3),len=zlen))

      ! Read data
      if (zlen==1) then
         ! 2D data
         fld_tmp = mpl%msv%valr
         if (trim(nam%lev2d)=='first') then
            call mpl%ncerr(subr,nf90_get_var(ncid,fld_id,fld_tmp(:,:,1),(/1,1,1,1/),(/model%nlon,model%nlat,1,1/)))
         elseif (trim(nam%lev2d)=='last') then
            call mpl%ncerr(subr,nf90_get_var(ncid,fld_id,fld_tmp(:,:,model%nl0),(/1,1,1,1/),(/model%nlon,model%nlat,1,1/)))
         end if
      else
         ! 3D data
         do il0=1,nam%nl
            call mpl%ncerr(subr,nf90_get_var(ncid,fld_id,fld_tmp(:,:,il0),(/1,1,nam%levs(il0),1/),(/model%nlon,model%nlat,1,1/)))
         end do
      end if

      ! Pack
      do imgt=1,model%nmgt
         img = model%mgt_to_mg(imgt)
         ilon = model%mg_to_lon(img)
         ilat = model%mg_to_lat(img)
         fld_mgt(imgt,:) = fld_tmp(ilon,ilat,:)
      end do
   end if
   call mpl%glb_to_loc(model%nl0,model%nmga,model%nmgt,model%mga_to_mgt,fld_mgt,fld(:,:,iv), &
 & model%ioproc(model%mytile),model%tilepool(:,model%mytile))
end do

if (model%ioproc(model%mytile)==mpl%myproc) then
   ! Close file
   call mpl%ncerr(subr,nf90_close(ncid))

   ! Release memory
   deallocate(fld_tmp)
end if

end subroutine model_fv3_read
