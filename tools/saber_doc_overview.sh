#!/bin/bash
#----------------------------------------------------------------------
# Bash shell script: saber_doc_overview
# Author: Benjamin Menetrier
# Licensing: this code is distributed under the CeCILL-C license
# Copyright © 2015-... UCAR, CERFACS, METEO-FRANCE and IRIT
#----------------------------------------------------------------------

# Directories
rootdir=$1/..
docdir=${rootdir}/docs

# Languages
languages="*.cc *.f *.F90 *.h"

cat<<EOF > ${docdir}/mainpage.h
#ifndef DOCS_MAINPAGE_H_
#define DOCS_MAINPAGE_H_

// This file defines what appears on the Main Page of the documentation
// generated by doxygen. The file contains no code, and does not appear
// in any cpp include statement.
//
/*!
 * \mainpage System Agnostic Background Error Representation
 *
 * \section IntroSection Introduction
 * Welcome to the documentation for the System Agnostic Background Error 
 * Representation (SABER)
 *
 * \section DirStructure Directory structure
EOF

# Directory structure
echo "## Directory structure"
echo -e " * The SABER repository is organized as follows:" >> ${docdir}/mainpage.h
dir=()
name=()
cd ${rootdir}
lev1=`ls -d */ 2> /dev/null`
for dir1 in ${lev1}; do
   cd ${dir1}
   desc=`cat .description 2> /dev/null`
   echo -e " * - **"${dir1%?}"**: "${desc} >> ${docdir}/mainpage.h
   list=`ls ${languages} 2> /dev/null`
   if test -n "${list}"; then
      name+=("${dir1%?}")
   fi
   lev2=`ls -d */ 2> /dev/null`
   for dir2 in ${lev2}; do
      cd ${dir2}
      desc=`cat .description 2> /dev/null`
      echo -e " *   - **"${dir2%?}"**: "${desc} >> ${docdir}/mainpage.h
      list=`ls ${languages} 2> /dev/null`
      if test -n "${list}"; then
         dir+=("${dir1%?}/${dir2%?}")
         name+=("${dir1%?}_${dir2%?}")
      fi
      lev3=`ls -d */ 2> /dev/null`
      for dir3 in ${lev3}; do
         cd ${dir3}
         desc=`cat .description 2> /dev/null`
         echo -e " *     - **"${dir3%?}"**: "${desc} >> ${docdir}/mainpage.h
         list=`ls ${languages} 2> /dev/null`
         if test -n "${list}"; then
            dir+=("${dir1%?}/${dir2%?}/${dir3%?}")
            name+=("${dir1%?}_${dir2%?}_${dir3%?}")
         fi
         lev4=`ls -d */ 2> /dev/null`
         for dir4 in ${lev4}; do
            cd ${dir4}
            desc=`cat .description 2> /dev/null`
            echo -e " *       - **"${dir4%?}"**: "${desc} >> ${docdir}/mainpage.h
            list=`ls ${languages} 2> /dev/null`
            if test -n "${list}"; then
               dir+=("${dir1%?}/${dir2%?}/${dir3%?}/${dir4%?}")
               name+=("${dir1%?}_${dir2%?}_${dir3%?}_${dir4%?}")
            fi
            lev5=`ls -d */ 2> /dev/null`
            for dir5 in ${lev5}; do
               cd ${dir4}
               desc=`cat .description 2> /dev/null`
               echo -e " *         - **"${dir5%?}"**: "${desc} >> ${docdir}/mainpage.h
               list=`ls ${languages} 2> /dev/null`
               if test -n "${list}"; then
                  dir+=("${dir1%?}/${dir2%?}/${dir3%?}/${dir4%?}/${dir5%?}")
                  name+=("${dir1%?}_${dir2%?}_${dir3%?}_${dir4%?}_${dir5%?}")
               fi
            done
            cd ..
         done
         cd ..  
      done
      cd ..
   done
   cd ..
done

if type "cloc" > /dev/null ; then
   echo -e " *" >> ${docdir}/mainpage.h

   # Cloc report
   for index in ${!dir[*]}; do
      cloc --quiet --csv --exclude-lang=CMake --out=cloc_${name[$index]}.csv ${rootdir}/${dir[$index]}
   done

   # Code size and characteristics
   echo "## Code size and characteristics"
   echo -e " * \section CLOC Code size and characteristics" >> ${docdir}/mainpage.h
   echo -e " * Code report obtained with [CLOC](https://github.com/AlDanial/cloc).\n" >> ${docdir}/mainpage.h
   OLDIFS=$IFS
   IFS=,
   for index in ${!dir[*]}; do
      echo -e " * \subsection dir_${index} ${dir[$index]}" >> ${docdir}/mainpage.h
      i=0
      while read files language blank comment code dum ; do
         if test $i == 0 ; then
            ratio="${comment}/${code} ratio"
         else
            let ratio=100*comment/code
            ratio="${ratio} %"
         fi
         echo -e " * | ${language} | ${files} | ${blank} | ${comment} | ${code} | ${ratio} |" >> ${docdir}/mainpage.h
         if test $i == 0 ; then
            echo -e " * |:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|" >> ${docdir}/mainpage.h
         fi
         let i=i+1
      done < cloc_${name[$index]}.csv
      echo -e " *" >> ${docdir}/mainpage.h
   done
   IFS=$OLDIFS
   for index in ${!dir[*]}; do
      rm -f cloc_${name[$index]}.csv
   done
else
   echo "cloc not found: no cloc report"
fi

echo -e "*/" >> ${docdir}/mainpage.h
echo -e "" >> ${docdir}/mainpage.h
echo -e "#endif  // DOCS_MAINPAGE_H_" >> ${docdir}/mainpage.h
