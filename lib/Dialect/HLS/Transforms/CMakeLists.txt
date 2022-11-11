add_circt_dialect_library(CIRCTHLSTransforms
  UnrollMarkedLoops.cpp

  DEPENDS
  CIRCTHLSTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHLS
  CIRCTHW
  CIRCTPipelineOps
  CIRCTSupport
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRAffineDialect
  MLIRTransformUtils
  )
