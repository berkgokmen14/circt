add_circt_library(CIRCTSCFToSTG
  SCFToSTG.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineToStandard
  MLIRPass
  CIRCTScheduling
  CIRCTSchedulingAnalysis
  CIRCTSTGOps
  )
