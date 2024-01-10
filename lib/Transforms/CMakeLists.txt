add_circt_library(CIRCTTransforms
  FlattenMemRefs.cpp
  StripDebugInfoWithPred.cpp
  UnrollForLoopSchedule.cpp
  MapArithToComb.cpp
  HoistInterIterationArg.cpp
  PipelineForLoopSchedule.cpp
  MaximizeSSA.cpp
  InsertMergeBlocks.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Transforms

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSupport
  MLIRTransforms

  DEPENDS
  CIRCTTransformsPassIncGen
  MLIRArithDialect
)
