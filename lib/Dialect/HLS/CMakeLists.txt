##===- CMakeLists.txt - HLS dialect code --------------------*- cmake -*-===//
##
## Implementation files for the HLS dialect.
##
##===----------------------------------------------------------------------===//

set(srcs
  HLSDialect.cpp
  )

set(HLS_LinkLibs
  CIRCTSupport
  CIRCTSV
  CIRCTHW
  MLIRArithDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRTransforms
  )

set(HLS_Deps
  ${HLS_LinkLibs}
  )

add_circt_dialect_library(CIRCTHLS
  ${srcs}

  DEPENDS
  ${HLS_Deps}

  LINK_COMPONENTS
  Core
  Support

  LINK_LIBS PUBLIC
  ${HLS_LinkLibs}
  )

add_subdirectory(Transforms)
