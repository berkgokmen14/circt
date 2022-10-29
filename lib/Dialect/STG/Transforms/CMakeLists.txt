add_circt_dialect_library(CIRCTSTGTransforms

  DEPENDS
  CIRCTSTGTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTPipelineOps
  CIRCTSTGOps
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
