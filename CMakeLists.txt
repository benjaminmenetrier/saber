# (C) Copyright 2017-2020 UCAR.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

################################################################################
# SABER
################################################################################

cmake_minimum_required( VERSION 3.12 )

project( saber VERSION 1.0.0 LANGUAGES CXX Fortran )

## Ecbuild integration
find_package( ecbuild QUIET )
include( ecbuild_system NO_POLICY_SCOPE )
ecbuild_declare_project()
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )
set( CMAKE_DIRECTORY_LABELS ${PROJECT_NAME} )

## Configuration options
option( ENABLE_SABER_DOC "Build SABER documentation" OFF )
option( ENABLE_MKL "Use MKL for LAPACK implementation (if available)" ON )
include( ${PROJECT_NAME}_compiler_flags )

## Dependencies
# Required
find_package( OpenMP REQUIRED COMPONENTS Fortran )
find_package( MPI REQUIRED COMPONENTS Fortran )
find_package( NetCDF REQUIRED COMPONENTS Fortran )
find_package( eckit 1.11.6 REQUIRED COMPONENTS MPI )
find_package( fckit 0.7.0  REQUIRED )
find_package( atlas 0.20.2 REQUIRED COMPONENTS OMP_Fortran )
if( ENABLE_MKL )
    find_package( MKL )
endif()
if( MKL_FOUND )
    set( LAPACK_LIBRARIES ${MKL_LIBRARIES} )
else()
    find_package( LAPACK REQUIRED )
endif()
# Optionals
find_package( oops QUIET OPTIONAL_COMPONENTS qg )

## Sources
add_subdirectory( src/saber )
add_subdirectory( tools ) # tools is needed for test
add_subdirectory( test )
if( ENABLE_SABER_DOC )
    add_subdirectory( docs )
endif()

## Global tests
ecbuild_add_test( TARGET saber_coding_norms_src
                  TYPE SCRIPT
                  COMMAND ${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME}_cpplint.py
                  ARGS --quiet --recursive ${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/src )
ecbuild_add_test( TARGET saber_coding_norms_test
                  TYPE SCRIPT
                  COMMAND ${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME}_cpplint.py
                  ARGS --quiet --recursive ${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/test )

## Package Config
ecbuild_install_project( NAME ${PROJECT_NAME} )

ecbuild_print_summary()
