!----------------------------------------------------------------------
! Module: type_mesh
!> Mesh derived type
! Author: Benjamin Menetrier
! Licensing: this code is distributed under the CeCILL-C license
! Copyright © 2015-... UCAR, CERFACS, METEO-FRANCE and IRIT
!----------------------------------------------------------------------
module type_mesh

!$ use omp_lib
use tools_const, only: pi,req
use tools_func, only: lonlathash,sphere_dist,lonlat2xyz,xyz2lonlat,vector_product
use tools_kinds, only: kind_real
use tools_qsort, only: qsort
use tools_stripack, only: addnod,bnodes,inside,trfind,trlist,trmesh
use type_mpl, only: mpl_type
use type_rng, only: rng_type

implicit none

logical,parameter :: shuffle = .true. ! Shuffle mesh order (more efficient to compute the Delaunay triangulation)

! Mesh derived type
type mesh_type
   ! Mesh structure
   integer :: n                                 !< Number of points
   integer,allocatable :: order(:)              !< Order of shuffled points
   integer,allocatable :: order_inv(:)          !< Inverse order of shuffled points
   real(kind_real),allocatable :: lon(:)        !< Points longitudes
   real(kind_real),allocatable :: lat(:)        !< Points latitudes
   real(kind_real),allocatable :: x(:)          !< x-coordinate
   real(kind_real),allocatable :: y(:)          !< y-coordinate
   real(kind_real),allocatable :: z(:)          !< z-coordinate
   integer,allocatable :: list(:)               !< Stripack list
   integer,allocatable :: lptr(:)               !< Stripack list pointer
   integer,allocatable :: lend(:)               !< Stripack list end
   integer :: lnew                              !< Stripack pointer to the first empty location in list
   integer :: nb                                !< Number of boundary nodes
   integer,allocatable :: bnd(:)                !< Boundary nodes
   integer,allocatable :: barc(:,:)             !< Boundary arcs
   real(kind_real),allocatable :: barc_lon(:,:) !< Boundary arcs longitudes
   real(kind_real),allocatable :: barc_lat(:,:) !< Boundary arcs latitudes
   real(kind_real),allocatable :: barc_dist(:)  !< Boundary arcs distance
   real(kind_real),allocatable :: barc_vp(:,:)  !< Boundary arcs normal vector

   ! Mesh attributes (used publicly)
   integer,allocatable :: nnb(:)                !< Number of neighbors
   integer,allocatable :: inb(:,:)              !< Neighbors indices
   real(kind_real),allocatable :: bdist(:)      !< Distance to the closest boundary arc

   ! Triangles data
   integer :: nt                                !< Number of triangles
   integer :: na                                !< Number of arcs
   integer,allocatable :: ltri(:,:)             !< Triangles indices
   integer,allocatable :: larc(:,:)             !< Arcs indices
   logical,allocatable :: valid(:)              !< Valid mesh nodes
contains
   procedure :: alloc => mesh_alloc
   procedure :: init => mesh_init
   procedure :: dealloc => mesh_dealloc
   procedure :: copy => mesh_copy
   procedure :: store => mesh_store
   procedure :: trlist => mesh_trlist
   procedure :: bnodes => mesh_bnodes
   procedure :: find_bdist => mesh_find_bdist
   procedure :: check => mesh_check
   procedure :: inside => mesh_inside
   procedure :: barycentric => mesh_barycentric
   procedure :: count_bnda => mesh_count_bnda
   procedure :: get_bnda => mesh_get_bnda
end type mesh_type

private
public :: mesh_type

contains

!----------------------------------------------------------------------
! Subroutine: mesh_alloc
!> Allocation
!----------------------------------------------------------------------
subroutine mesh_alloc(mesh,n)

implicit none

! Passed variables
class(mesh_type),intent(inout) :: mesh !< Mesh
integer,intent(in) :: n                !< Mesh size

! Allocation
mesh%n = n

! Allocation
allocate(mesh%order(mesh%n))
allocate(mesh%order_inv(mesh%n))
allocate(mesh%lon(mesh%n))
allocate(mesh%lat(mesh%n))
allocate(mesh%x(mesh%n))
allocate(mesh%y(mesh%n))
allocate(mesh%z(mesh%n))
allocate(mesh%list(6*(mesh%n-2)))
allocate(mesh%lptr(6*(mesh%n-2)))
allocate(mesh%lend(mesh%n))
allocate(mesh%nnb(mesh%n))

end subroutine mesh_alloc

!----------------------------------------------------------------------
! Subroutine: mesh_init
!> Intialization
!----------------------------------------------------------------------
subroutine mesh_init(mesh,mpl,rng,lon,lat)

implicit none

! Passed variables
class(mesh_type),intent(inout) :: mesh            !< Mesh
type(mpl_type),intent(inout) :: mpl               !< MPI data
type(rng_type),intent(inout) :: rng               !< Random number generator
real(kind_real),intent(in) :: lon(mesh%n)         !< Longitudes
real(kind_real),intent(in) :: lat(mesh%n)         !< Latitudes

! Local variables
integer :: i,ii,j,k,info,nnbmax
integer :: near(mesh%n),next(mesh%n)
integer,allocatable :: jtab(:)
real(kind_real) :: dist(mesh%n)
real(kind_real),allocatable :: list(:)
logical :: init
character(len=1024),parameter :: subr = 'mesh_init'

! Points order
do i=1,mesh%n
   mesh%order(i) = i
end do

! Allocation
allocate(list(mesh%n))

! Reorder points
do i=1,mesh%n
   list(i) = lonlathash(lon(i),lat(i))
end do
call qsort(mesh%n,list,mesh%order)

if (shuffle) then
   ! Allocation
   allocate(jtab(mesh%n))

   ! Shuffle order (more efficient to compute the Delaunay triangulation)
   call rng%resync(mpl)
   call rng%rand_integer(1,mesh%n,jtab)
   call rng%desync(mpl)
   do i=mesh%n,2,-1
      k = mesh%order(jtab(i))
      mesh%order(jtab(i)) = mesh%order(i)
      mesh%order(i) = k
   end do

   ! Release memory
   deallocate(jtab)
end if

! Restrictive inverse order
mesh%order_inv = mpl%msv%vali
do i=1,mesh%n
   mesh%order_inv(mesh%order(i)) = i
end do

! Store coordinates
call mesh%store(mpl,lon,lat)

! Create mesh
mesh%list = 0
mesh%lend = 0
mesh%lnew = 0
if (mesh%n>2) then
   call trmesh(mpl,mesh%n,mesh%x,mesh%y,mesh%z,mesh%list,mesh%lptr,mesh%lend,mesh%lnew,near,next,dist,info)
   if (info/=0) call mpl%abort(subr,'trmesh failed')
end if

! Boundaries not computed yet
mesh%nb = mpl%msv%vali

! Count neighbors
do i=1,mesh%n
   ii = mesh%order(i)
   mesh%nnb(ii) = 0
   j = mesh%lend(i)
   init = .true.
   do while ((j/=mesh%lend(i)).or.init)
      mesh%nnb(ii) = mesh%nnb(ii)+1
      j = mesh%lptr(j)
      init = .false.
   end do
end do

! Find neighbors indices
nnbmax = maxval(mesh%nnb)
allocate(mesh%inb(mesh%n,nnbmax))
do i=1,mesh%n
   ii = mesh%order(i)
   mesh%nnb(ii) = 0
   j = mesh%lend(i)
   init = .true.
   do while ((j/=mesh%lend(i)).or.init)
      mesh%nnb(ii) = mesh%nnb(ii)+1
      mesh%inb(ii,mesh%nnb(ii)) = mesh%order(abs(mesh%list(j)))
      j = mesh%lptr(j)
      init = .false.
   end do
end do

! Release memory
deallocate(list)

end subroutine mesh_init

!----------------------------------------------------------------------
! Subroutine: mesh_dealloc
!> Release memory
!----------------------------------------------------------------------
subroutine mesh_dealloc(mesh)

implicit none

! Passed variables
class(mesh_type),intent(inout) :: mesh !< Mesh

! Release memory
if (allocated(mesh%order)) deallocate(mesh%order)
if (allocated(mesh%order_inv)) deallocate(mesh%order_inv)
if (allocated(mesh%lon)) deallocate(mesh%lon)
if (allocated(mesh%lat)) deallocate(mesh%lat)
if (allocated(mesh%x)) deallocate(mesh%x)
if (allocated(mesh%y)) deallocate(mesh%y)
if (allocated(mesh%z)) deallocate(mesh%z)
if (allocated(mesh%list)) deallocate(mesh%list)
if (allocated(mesh%lptr)) deallocate(mesh%lptr)
if (allocated(mesh%lend)) deallocate(mesh%lend)
if (allocated(mesh%bnd)) deallocate(mesh%bnd)
if (allocated(mesh%barc)) deallocate(mesh%barc)
if (allocated(mesh%barc_lon)) deallocate(mesh%barc_lon)
if (allocated(mesh%barc_lat)) deallocate(mesh%barc_lat)
if (allocated(mesh%barc_dist)) deallocate(mesh%barc_dist)
if (allocated(mesh%barc_vp)) deallocate(mesh%barc_vp)
if (allocated(mesh%nnb)) deallocate(mesh%nnb)
if (allocated(mesh%inb)) deallocate(mesh%inb)
if (allocated(mesh%bdist)) deallocate(mesh%bdist)
if (allocated(mesh%ltri)) deallocate(mesh%ltri)
if (allocated(mesh%larc)) deallocate(mesh%larc)
if (allocated(mesh%valid)) deallocate(mesh%valid)

end subroutine mesh_dealloc

!----------------------------------------------------------------------
! Subroutine: mesh_copy
!> Copy
!----------------------------------------------------------------------
subroutine mesh_copy(mesh_out,mesh_in)

implicit none

! Passed variables
class(mesh_type),intent(inout) :: mesh_out !< Output mesh
type(mesh_type),intent(in) :: mesh_in      !< Input mesh

! Release memory
call mesh_out%dealloc

! Allocation
call mesh_out%alloc(mesh_in%n)
if (allocated(mesh_in%bnd)) allocate(mesh_out%bnd(mesh_in%nb))
if (allocated(mesh_in%barc)) allocate(mesh_out%barc(2,mesh_in%nb))
if (allocated(mesh_in%barc_lon)) allocate(mesh_out%barc_lon(2,mesh_in%nb))
if (allocated(mesh_in%barc_lat)) allocate(mesh_out%barc_lat(2,mesh_in%nb))
if (allocated(mesh_in%barc_dist)) allocate(mesh_out%barc_dist(mesh_in%nb))
if (allocated(mesh_in%barc_vp)) allocate(mesh_out%barc_vp(3,mesh_in%nb))
if (allocated(mesh_in%inb)) allocate(mesh_out%inb(size(mesh_in%inb,1),size(mesh_in%inb,2)))
if (allocated(mesh_in%bdist)) allocate(mesh_out%bdist(mesh_in%n))
if (allocated(mesh_in%ltri)) allocate(mesh_out%ltri(3,mesh_in%nt))
if (allocated(mesh_in%larc)) allocate(mesh_out%larc(2,mesh_in%na))
if (allocated(mesh_in%valid)) allocate(mesh_out%valid(mesh_in%n))

! Copy data
mesh_out%order = mesh_in%order
mesh_out%order_inv = mesh_in%order_inv
mesh_out%lon = mesh_in%lon
mesh_out%lat = mesh_in%lat
mesh_out%x = mesh_in%x
mesh_out%y = mesh_in%y
mesh_out%z = mesh_in%z
mesh_out%list = mesh_in%list
mesh_out%lptr = mesh_in%lptr
mesh_out%lend = mesh_in%lend
mesh_out%lnew = mesh_in%lnew
mesh_out%nb = mesh_in%nb
if (allocated(mesh_in%bnd)) mesh_out%bnd = mesh_in%bnd
if (allocated(mesh_in%barc)) mesh_out%barc = mesh_in%barc
if (allocated(mesh_in%barc_lon)) mesh_out%barc_lon = mesh_in%barc_lon
if (allocated(mesh_in%barc_lat)) mesh_out%barc_lat = mesh_in%barc_lat
if (allocated(mesh_in%barc_dist)) mesh_out%barc_dist = mesh_in%barc_dist
if (allocated(mesh_in%barc_vp)) mesh_out%barc_vp = mesh_in%barc_vp
mesh_out%nnb = mesh_in%nnb
mesh_out%inb = mesh_in%inb
if (allocated(mesh_in%bdist)) mesh_out%bdist = mesh_in%bdist
mesh_out%nt = mesh_in%nt
mesh_out%na = mesh_in%na
if (allocated(mesh_in%ltri)) mesh_out%ltri = mesh_in%ltri
if (allocated(mesh_in%larc)) mesh_out%larc = mesh_in%larc
if (allocated(mesh_in%valid)) mesh_out%valid = mesh_in%valid

end subroutine mesh_copy

!----------------------------------------------------------------------
! Subroutine: mesh_store
!> Store mesh cartesian coordinates
!----------------------------------------------------------------------
subroutine mesh_store(mesh,mpl,lon,lat)

implicit none

! Passed variables
class(mesh_type),intent(inout) :: mesh    !< Mesh
type(mpl_type),intent(inout) :: mpl       !< MPI data
real(kind_real),intent(in) :: lon(mesh%n) !< Longitude
real(kind_real),intent(in) :: lat(mesh%n) !< Latitude

! Local variables
integer :: i

! Initialize
mesh%lon = mpl%msv%valr
mesh%lat = mpl%msv%valr

! Copy lon/lat
mesh%lon = lon(mesh%order)
mesh%lat = lat(mesh%order)

! Transform to cartesian coordinates
do i=1,mesh%n
   call lonlat2xyz(mpl,mesh%lon(i),mesh%lat(i),mesh%x(i),mesh%y(i),mesh%z(i))
end do

end subroutine mesh_store

!----------------------------------------------------------------------
! Subroutine: mesh_trlist
!> Compute triangle list, arc list
!----------------------------------------------------------------------
subroutine mesh_trlist(mesh,mpl)

implicit none

! Passed variables
class(mesh_type),intent(inout) :: mesh !< Mesh
type(mpl_type),intent(inout) :: mpl    !< MPI data

! Local variables
integer :: info,ia,it,i,i1,i2
integer :: ltri(9,2*(mesh%n-2))
character(len=6) :: notvalidchar
character(len=1024),parameter :: subr = 'mesh_trlist'

if (mesh%n>2) then
   ! Create triangles list
   call trlist(mesh%n,mesh%list,mesh%lptr,mesh%lend,9,mesh%nt,ltri,info)
   if (info/=0) call mpl%abort(subr,'trlist failed')

   ! Allocation
   mesh%na = maxval(ltri(7:9,1:mesh%nt))
   allocate(mesh%ltri(3,mesh%nt))
   allocate(mesh%larc(2,mesh%na))
   allocate(mesh%valid(mesh%n))

   ! Copy triangle list
   mesh%ltri = ltri(1:3,1:mesh%nt)
else
   ! No mesh
   mesh%nt = 0
   mesh%na = 0
end if

! Copy arcs list
do ia=1,mesh%na
   it = 1
   do while (it<=mesh%nt)
      if (any(ltri(7:9,it)==ia)) exit
      it = it+1
   end do
   i = 1
   do while (i<=3)
      if (ltri(6+i,it)==ia) exit
      i = i+1
   end do
   i1 = mod(i+1,3)
   if (i1==0) i1 = 3
   i2 = mod(i+2,3)
   if (i2==0) i2 = 3
   mesh%larc(1,ia) = mesh%order(ltri(i1,it))
   mesh%larc(2,ia) = mesh%order(ltri(i2,it))
end do

! Check mesh
call mesh%check(mpl,mesh%valid)
if (.not.all(mesh%valid)) then
   write(notvalidchar,'(i6)') count(.not.mesh%valid)
   call mpl%warning(subr,'unvalid mesh at creation ('//notvalidchar//' points)')
end if

end subroutine mesh_trlist

!----------------------------------------------------------------------
! Subroutine: mesh_bnodes
!> Find boundary nodes
!----------------------------------------------------------------------
subroutine mesh_bnodes(mesh,mpl,bdist)

implicit none

! Passed variables
class(mesh_type),intent(inout) :: mesh !< Mesh
type(mpl_type),intent(inout) :: mpl    !< MPI data
logical,intent(in),optional :: bdist   !< Find minimum distance a boundary arc

! Local variables
integer :: i,ii,bnd(mesh%n)
real(kind_real) :: v1(3),v2(3)
logical :: lbdist

! Initialization
lbdist = .false.
if (present(bdist)) lbdist = bdist

if (mesh%n>2) then
   ! Find boundary nodes
   bnd = mpl%msv%vali
   call bnodes(mesh%n,mesh%list,mesh%lptr,mesh%lend,bnd,mesh%nb,mesh%na,mesh%nt)

   ! Allocation
   allocate(mesh%bnd(mesh%nb))

   ! Copy
   mesh%bnd = bnd(1:mesh%nb)
else
   ! No mesh
   mesh%nb = 0
   mesh%na = 0
   mesh%nt = 0
end if


! Allocation
if (mesh%nb>0) then
   allocate(mesh%barc(2,mesh%nb))
   allocate(mesh%barc_lon(2,mesh%nb))
   allocate(mesh%barc_lat(2,mesh%nb))
   allocate(mesh%barc_dist(mesh%nb))
   allocate(mesh%barc_vp(3,mesh%nb))
end if
allocate(mesh%bdist(mesh%n))

! Define boundary arcs
if (mesh%nb>0) then
   do i=1,mesh%nb-1
      mesh%barc(1,i) = mesh%bnd(i)
      mesh%barc(2,i) = mesh%bnd(i+1)
   end do
   mesh%barc(1,mesh%nb) = mesh%bnd(mesh%nb)
   mesh%barc(2,mesh%nb) = mesh%bnd(1)
end if

! Compute boundary arcs properties
do i=1,mesh%nb
   mesh%barc_lon(:,i) = (/mesh%lon(mesh%barc(1,i)),mesh%lon(mesh%barc(2,i))/)
   mesh%barc_lat(:,i) = (/mesh%lat(mesh%barc(1,i)),mesh%lat(mesh%barc(2,i))/)
   call sphere_dist(mesh%barc_lon(1,i),mesh%barc_lat(1,i),mesh%barc_lon(2,i),mesh%barc_lat(2,i),mesh%barc_dist(i))
   v1 = (/mesh%x(mesh%barc(1,i)),mesh%y(mesh%barc(1,i)),mesh%z(mesh%barc(1,i))/)
   v2 = (/mesh%x(mesh%barc(2,i)),mesh%y(mesh%barc(2,i)),mesh%z(mesh%barc(2,i))/)
   call vector_product(v1,v2,mesh%barc_vp(:,i))
end do

if (lbdist) then
   ! Find minimal distance to a boundary arc
   do i=1,mesh%n
      ii = mesh%order(i)
      call mesh%find_bdist(mpl,mesh%lon(i),mesh%lat(i),mesh%bdist(ii))
   end do
else
   ! Missing
   mesh%bdist = mpl%msv%valr
end if

end subroutine mesh_bnodes

!----------------------------------------------------------------------
! Subroutine: mesh_find_bdist
!> Find shortest distance to boundary arcs
!----------------------------------------------------------------------
subroutine mesh_find_bdist(mesh,mpl,lon,lat,bdist)

implicit none

! Passed variables
class(mesh_type),intent(in) :: mesh  !< Mesh
type(mpl_type),intent(inout) :: mpl  !< MPI data
real(kind_real),intent(in) :: lon    !< Longitude
real(kind_real),intent(in) :: lat    !< Latitude
real(kind_real),intent(out) :: bdist !< Distance to boundary

! Local variables
integer :: i
real(kind_real) :: v(3),vf(3),vt(3),tlat,tlon,dist_t1,dist_t2
character(len=1024),parameter :: subr = 'mesh_find_bdist'

! Check
if (mpl%msv%is(mesh%nb)) call mpl%abort(subr,'boundary arcs have not been computed')

! Initialization
bdist = pi

if (mesh%nb>0) then
   ! Transform to cartesian coordinates
   call lonlat2xyz(mpl,lon,lat,v(1),v(2),v(3))
end if

! Compute the shortest distance from each boundary arc great-circle
do i=1,mesh%nb
   ! Vector products
   call vector_product(v,mesh%barc_vp(:,i),vf)
   call vector_product(mesh%barc_vp(:,i),vf,vt)

   ! Back to spherical coordinates
   call xyz2lonlat(mpl,vt(1),vt(2),vt(3),tlon,tlat)

   ! Check whether T is on the arc
   call sphere_dist(tlon,tlat,mesh%barc_lon(1,i),mesh%barc_lat(1,i),dist_t1)
   call sphere_dist(tlon,tlat,mesh%barc_lon(2,i),mesh%barc_lat(2,i),dist_t2)
   if ((dist_t1<mesh%barc_dist(i)).and.(dist_t2<mesh%barc_dist(i))) then
      ! T is on the arc
      call sphere_dist(lon,lat,tlon,tlat,dist_t1)
      bdist = min(bdist,dist_t1)
   else
      ! T is not on the arc
      call sphere_dist(lon,lat,mesh%barc_lon(1,i),mesh%barc_lat(1,i),dist_t1)
      call sphere_dist(lon,lat,mesh%barc_lon(2,i),mesh%barc_lat(2,i),dist_t2)
      bdist = min(bdist,min(dist_t1,dist_t2))
   end if
end do

end subroutine mesh_find_bdist

!----------------------------------------------------------------------
! Subroutine: mesh_check
!> Check whether the mesh is made of counter-clockwise triangles
!----------------------------------------------------------------------
subroutine mesh_check(mesh,mpl,valid)

implicit none

! Passed variables
class(mesh_type),intent(inout) :: mesh !< Mesh
type(mpl_type),intent(inout) :: mpl    !< MPI data
logical,intent(out) :: valid(mesh%n)   !< Validity flag

! Local variables
integer :: it
real(kind_real),allocatable :: a(:),b(:),c(:),cd(:),cp(:),v1(:),v2(:)
logical :: validt(mesh%nt)

!$omp parallel do schedule(static) private(it) firstprivate(a,b,c,cd,cp,v1,v2)
do it=1,mesh%nt
   ! Allocation
   allocate(a(3))
   allocate(b(3))
   allocate(c(3))
   allocate(cd(3))
   allocate(cp(3))
   allocate(v1(3))
   allocate(v2(3))

   ! Check vertices status
   if (mpl%msv%isallnot(mesh%x(mesh%ltri(:,it))).and.mpl%msv%isallnot(mesh%y(mesh%ltri(:,it))) &
 & .and.mpl%msv%isallnot(mesh%z(mesh%ltri(:,it)))) then
      ! Vertices
      a = (/mesh%x(mesh%ltri(1,it)),mesh%y(mesh%ltri(1,it)),mesh%z(mesh%ltri(1,it))/)
      b = (/mesh%x(mesh%ltri(2,it)),mesh%y(mesh%ltri(2,it)),mesh%z(mesh%ltri(2,it))/)
      c = (/mesh%x(mesh%ltri(3,it)),mesh%y(mesh%ltri(3,it)),mesh%z(mesh%ltri(3,it))/)

      ! Cross-product (c-b)x(a-b)
      v1 = c-b
      v2 = a-b
      call vector_product(v1,v2,cp)

      ! Centroid
      cd = (a+b+c)/3.0

      ! Compare the directions
      validt(it) = sum(cp*cd)>0.0
   else
      ! At least one vertex is missing
      validt(it) = .false.
   end if

   ! Release memory
   deallocate(a)
   deallocate(b)
   deallocate(c)
   deallocate(cd)
   deallocate(cp)
   deallocate(v1)
   deallocate(v2)
end do
!$omp end parallel do

! Check vertices
valid = .true.
do it=1,mesh%nt
   if (.not.validt(it)) valid(mesh%ltri(:,it)) = .false.
end do

end subroutine mesh_check

!----------------------------------------------------------------------
! Subroutine: mesh_inside
!> Find whether a point is inside the mesh
!----------------------------------------------------------------------
subroutine mesh_inside(mesh,mpl,lon,lat,inside_mesh)

implicit none

! Passed variables
class(mesh_type),intent(in) :: mesh !< Mesh
type(mpl_type),intent(inout) :: mpl !< MPI data
real(kind_real),intent(in) :: lon   !< Longitude
real(kind_real),intent(in) :: lat   !< Latitude
logical,intent(out) :: inside_mesh  !< True if the point is inside the mesh

! Local variables
integer :: info
real(kind_real) :: p(3)

if (mesh%nb>0) then
   ! Transform to cartesian coordinates
   call lonlat2xyz(mpl,lon,lat,p(1),p(2),p(3))

   ! Find whether the point is inside the convex hull
   inside_mesh = inside(p,mesh%n,mesh%x,mesh%y,mesh%z,mesh%nb,mesh%bnd,info)
else
   ! No boundary
   inside_mesh = .true.
end if

end subroutine mesh_inside

!----------------------------------------------------------------------
! Subroutine: mesh_barycentric
!> Compute barycentric coordinates
!----------------------------------------------------------------------
subroutine mesh_barycentric(mesh,mpl,lon,lat,istart,b,ib)

implicit none

! Passed variables
class(mesh_type),intent(in) :: mesh !< Mesh
type(mpl_type),intent(inout) :: mpl !< MPI data
real(kind_real),intent(in) :: lon   !< Longitude
real(kind_real),intent(in) :: lat   !< Latitude
integer,intent(in) :: istart        !< Starting index
real(kind_real),intent(out) :: b(3) !< Barycentric weights
integer,intent(out) :: ib(3)        !< Barycentric indices

! Local variables
integer :: i
real(kind_real) :: p(3)

! Transform to cartesian coordinates
call lonlat2xyz(mpl,lon,lat,p(1),p(2),p(3))

! Compute barycentric coordinates
b = 0.0
ib = 0
if (mesh%n>2) call trfind(istart,p,mesh%n,mesh%x,mesh%y,mesh%z,mesh%list,mesh%lptr,mesh%lend,b(1),b(2),b(3),ib(1),ib(2),ib(3))

! Transform indices
do i=1,3
   if (ib(i)>0) ib(i) = mesh%order(ib(i))
end do

end subroutine mesh_barycentric

!----------------------------------------------------------------------
! Subroutine: mesh_count_bnda
!> Count boundary arcs
!----------------------------------------------------------------------
subroutine mesh_count_bnda(mesh,gmask,nbnda)

implicit none

! Passed variables
class(mesh_type),intent(in) :: mesh !< Mesh
logical,intent(in) :: gmask(mesh%n) !< Mask
integer,intent(out) :: nbnda        !< Number of boundary nodes

! Local variables
integer :: i,j,k,ii,jj,kk,iend
logical :: init

! Initialiation
nbnda = 0

! Loop over points
do i=1,mesh%n
   ii = mesh%order(i)
   if (.not.gmask(ii)) then
      ! Initialization
      iend = mesh%lend(i)
      init = .true.

      ! Loop over neigbors
      do while ((iend/=mesh%lend(i)).or.init)
         j = abs(mesh%list(iend))
         k = abs(mesh%list(mesh%lptr(iend)))
         jj = mesh%order(j)
         kk = mesh%order(k)
         if (.not.gmask(jj).and.gmask(kk)) nbnda = nbnda+1
         iend = mesh%lptr(iend)
         init = .false.
      end do
   end if
end do

end subroutine mesh_count_bnda

!----------------------------------------------------------------------
! Subroutine: mesh_get_bnda
!> Get boundary arcs
!----------------------------------------------------------------------
subroutine mesh_get_bnda(mesh,gmask,nbnda,bnda_index)

implicit none

! Passed variables
class(mesh_type),intent(in) :: mesh        !< Mesh
logical,intent(in) :: gmask(mesh%n)        !< Mask
integer,intent(in) :: nbnda                !< Number of boundary nodes
integer,intent(out) :: bnda_index(2,nbnda) !< Boundary node index

! Local variables
integer :: ibnda,i,j,k,ii,jj,kk,iend
logical :: init

! Initialiation
ibnda = 0
bnda_index = 0

! Loop over points
do i=1,mesh%n
   ii = mesh%order(i)
   if (.not.gmask(ii)) then
      ! Initialization
      iend = mesh%lend(i)
      init = .true.

      ! Loop over neigbors
      do while ((iend/=mesh%lend(i)).or.init)
         j = abs(mesh%list(iend))
         k = abs(mesh%list(mesh%lptr(iend)))
         jj = mesh%order(j)
         kk = mesh%order(k)
         if (.not.gmask(jj).and.gmask(kk)) then
            ibnda = ibnda+1
            bnda_index(1,ibnda) = ii
            bnda_index(2,ibnda) = jj
         end if
         iend = mesh%lptr(iend)
         init = .false.
      end do
   end if
end do

end subroutine mesh_get_bnda

end module type_mesh
