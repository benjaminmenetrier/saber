!----------------------------------------------------------------------
! Subroutine: model_aro_coord
!> Load AROME coordinates
!----------------------------------------------------------------------
subroutine model_aro_coord(model,mpl,nam)

implicit none

! Passed variables
class(model_type),intent(inout) :: model !< Model
type(mpl_type),intent(inout) :: mpl      !< MPI data
type(nam_type),intent(in) :: nam         !< Namelist

! Local variables
integer :: img,ilon,ilat,il0,il0_bot,il0_top,ie,its
integer :: ncid,nlon_id,nlat_id,nlev_id,pp_id,lon_id,lat_id,cmask_id,a_id,b_id,fld_id,mask_id
integer,allocatable :: mask_counter(:)
real(kind_real) :: dx,dy
real(kind_real),allocatable :: lon(:,:),lat(:,:),cmask(:,:),a(:),b(:),fld_loc(:,:),fld(:,:)
character(len=3) :: ilchar
character(len=1024) :: filename
character(len=1024),parameter :: subr = 'model_aro_coord'

! Open file and get dimensions
model%ntile = 1
call mpl%ncerr(subr,nf90_open(trim(nam%datadir)//'/grid.nc',nf90_share,ncid))
call mpl%ncerr(subr,nf90_inq_dimid(ncid,'X',nlon_id))
call mpl%ncerr(subr,nf90_inq_dimid(ncid,'Y',nlat_id))
call mpl%ncerr(subr,nf90_inquire_dimension(ncid,nlon_id,len=model%nlon))
call mpl%ncerr(subr,nf90_inquire_dimension(ncid,nlat_id,len=model%nlat))
model%nmg = model%nlon*model%nlat
call mpl%ncerr(subr,nf90_inq_dimid(ncid,'Z',nlev_id))
call mpl%ncerr(subr,nf90_inquire_dimension(ncid,nlev_id,len=model%nlev))

! Allocation
call model%alloc
allocate(lon(model%nlon,model%nlat))
allocate(lat(model%nlon,model%nlat))
allocate(cmask(model%nlon,model%nlat))
allocate(a(model%nlev+1))
allocate(b(model%nlev+1))

! Read data and close file
call mpl%ncerr(subr,nf90_inq_varid(ncid,'longitude',lon_id))
call mpl%ncerr(subr,nf90_inq_varid(ncid,'latitude',lat_id))
call mpl%ncerr(subr,nf90_inq_varid(ncid,'cmask',cmask_id))
call mpl%ncerr(subr,nf90_inq_varid(ncid,'hybrid_coef_A',a_id))
call mpl%ncerr(subr,nf90_inq_varid(ncid,'hybrid_coef_B',b_id))
call mpl%ncerr(subr,nf90_inq_varid(ncid,'Projection_parameters',pp_id))
call mpl%ncerr(subr,nf90_get_var(ncid,lon_id,lon))
call mpl%ncerr(subr,nf90_get_var(ncid,lat_id,lat))
call mpl%ncerr(subr,nf90_get_var(ncid,cmask_id,cmask))
call mpl%ncerr(subr,nf90_get_var(ncid,a_id,a))
call mpl%ncerr(subr,nf90_get_var(ncid,b_id,b))
call mpl%ncerr(subr,nf90_get_att(ncid,pp_id,'x_resolution',dx))
call mpl%ncerr(subr,nf90_get_att(ncid,pp_id,'y_resolution',dy))
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
      model%lon(img) = lon(ilon,ilat)
      model%lat(img) = lat(ilon,ilat)
      select case (trim(zone))
      case ('C')
         model%mask(img,:) = (cmask(ilon,ilat)>0.75)
      case ('C+I')
         model%mask(img,:) = (cmask(ilon,ilat)>0.25)
      case ('C+I+E')
         model%mask(img,:) = .true.
      case default
         call mpl%abort(subr,'wrong AROME zone')
      end select
   end do
end do
model%area = dx*dy/req**2

! Specific mask
select case (trim(nam%mask_type))
case ("hyd")
   ! Based on an external file

   ! Allocation
   allocate(fld_loc(model%nlon,model%nlat))

   ! Open file
   call mpl%ncerr(subr,nf90_open(trim(nam%datadir)//'/'//trim(nam%prefix)//'_hyd.nc',nf90_nowrite,ncid))

   ! Read mask
   do il0=1,nam%nl
      ! Get id
      write(ilchar,'(i3.3)') nam%levs(il0)
      call mpl%ncerr(subr,nf90_inq_varid(ncid,'S'//ilchar//'MASK',mask_id))

      ! Read data
      call mpl%ncerr(subr,nf90_get_var(ncid,mask_id,fld_loc))

      ! Pack data
      img = 0
      do ilon=1,model%nlon
         do ilat=1,model%nlat
            img = img+1
            model%mask(img,il0) = (fld_loc(ilon,ilat)>nam%mask_th(1))
         end do
      end do
   end do

   ! Close file
   call mpl%ncerr(subr,nf90_close(ncid))

   ! Release memory
   deallocate(fld_loc)
case ("lwc_fog","lwc_clear","lwc_stratus")
   ! Based on the ensemble-mean LWC

   ! Allocation
   allocate(fld_loc(model%nlon,model%nlat))
   allocate(fld(model%nmg,model%nl0))
   allocate(mask_counter(model%nmga))

   ! Initialization
   mask_counter = 0
   select case (trim(nam%mask_type))
   case("lwc_fog")
      il0_bot = nam%nl
      il0_top = nam%nl-2
   case("lwc_clear")
      il0_bot = nam%nl
      il0_top = 1
   case("lwc_stratus")
      il0_bot = nam%nl-30
      il0_top = nam%nl-66
   end select

   ! Setup mask
   do ie=1,nam%ens1_ne
      ! Set file name
      write(filename,'(a,i6.6)') 'ens1_',ie

      ! Open file
      call mpl%ncerr(subr,nf90_open(trim(nam%datadir)//'/'//trim(filename)//'.nc',nf90_nowrite,ncid))

      ! Read LWC
      do il0=1,nam%nl
         ! Get id
         write(ilchar,'(i3.3)') nam%levs(il0)
         call mpl%ncerr(subr,nf90_inq_varid(ncid,'S'//ilchar//'CLOUD_WATER',fld_id))

         ! Read data
         call mpl%ncerr(subr,nf90_get_var(ncid,fld_id,fld_loc))

         ! Pack data
         img = 0
         do ilon=1,model%nlon
            do ilat=1,model%nlat
               img = img+1
               fld(img,il0) = real(fld_loc(ilon,ilat),kind_real)
            end do
         end do
      end do

      ! Close file
      call mpl%ncerr(subr,nf90_close(ncid))

      ! Update mask
      select case (trim(nam%mask_type))
      case("lwc_fog")
         do img=1,model%nmg
            if (all(fld(img,il0_top:il0_bot)>nam%mask_th)) mask_counter(img) = mask_counter(img)+1
         end do
      case("lwc_clear")
         do img=1,model%nmg
            if (all(fld(img,il0_top:il0_bot)<nam%mask_th)) mask_counter(img) = mask_counter(img)+1
         end do
      case("lwc_stratus")
         do img=1,model%nmg
            if (count(fld(img,il0_top:il0_bot)>nam%mask_th)>3) mask_counter(img) = mask_counter(img)+1
         end do
      end select
   end do

   ! Update mask
   do img=1,model%nmg
      if (real(mask_counter(img),kind_real)<0.9*real(nam%ens1_ne,kind_real)) model%mask(img,:) = .false.
   end do
end select

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
deallocate(cmask)
deallocate(a)
deallocate(b)

end subroutine model_aro_coord

!----------------------------------------------------------------------
! Subroutine: model_aro_read
!> Read AROME field
!----------------------------------------------------------------------
subroutine model_aro_read(model,mpl,nam,filename,fld)

implicit none

! Passed variables
class(model_type),intent(inout) :: model                        !< Model
type(mpl_type),intent(inout) :: mpl                             !< MPI data
type(nam_type),intent(in) :: nam                                !< Namelist
character(len=*),intent(in) :: filename                         !< File name
real(kind_real),intent(out) :: fld(model%nmga,model%nl0,nam%nv) !< Field

! Local variables
integer :: iv,il0,img,ilon,ilat,info,iv_q,iv_lwc
integer :: ncid,fld_id
real(kind_real) :: fld_mg(model%nmg,model%nl0)
real(kind_real),allocatable :: fld_tmp(:,:,:)
character(len=3) :: ilchar
character(len=1024),parameter :: subr = 'model_aro_read'

if (mpl%main) then
   ! Allocation
   allocate(fld_tmp(model%nlon,model%nlat,model%nl0))

   ! Open file
   call mpl%ncerr(subr,nf90_open(trim(nam%datadir)//'/'//trim(filename)//'.nc',nf90_nowrite,ncid))
end if

do iv=1,nam%nv
   if (mpl%main) then
      do il0=1,nam%nl
         ! Get id
         write(ilchar,'(i3.3)') nam%levs(il0)
         info = nf90_inq_varid(ncid,'S'//ilchar//trim(nam%variables(iv)),fld_id)

         ! Read data
         if (info==nf90_noerr) then
            ! 3D data
            call mpl%ncerr(subr,nf90_get_var(ncid,fld_id,fld_tmp(:,:,il0)))
         else
            ! Check if the variable exists as 2D data
            call mpl%ncerr(subr,nf90_inq_varid(ncid,nam%variables(iv),fld_id))

            ! 2D data
            if (((trim(nam%lev2d)=='first').and.(il0==1)).or.((trim(nam%lev2d)=='last').and.(il0==model%nl0))) then
               fld_tmp = mpl%msv%valr
               call mpl%ncerr(subr,nf90_get_var(ncid,fld_id,fld_tmp(:,:,il0)))
            end if
         end if
      end do

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

! Post-processing
iv_q = mpl%msv%vali
iv_lwc = mpl%msv%vali
do iv=1,nam%nv
   if (trim(nam%variables(iv))=='HUMI_SPECIFI') iv_q = iv
   if (trim(nam%variables(iv))=='CLOUD_WATER') iv_lwc = iv
end do
select case (trim(nam%variable_change))
case ('logq')
   if (mpl%msv%isnot(iv_q)) then
       fld(:,:,iv_q) = log(max(fld(:,:,iv_q),qmin))
   else
      call mpl%abort(subr,'specific humidity not found')
   end if
case ('logqtot')
   if ((mpl%msv%isnot(iv_q)).and.(mpl%msv%isnot(iv_lwc))) then
      fld(:,:,iv_q) = log(max(fld(:,:,iv_q)+fld(:,:,iv_lwc),qmin))
      fld(:,:,iv_lwc) = mpl%msv%valr
   else
      call mpl%abort(subr,'specific humidity, or LWC not found')
   end if
end select

end subroutine model_aro_read
