add_circt_dialect(HLS hls)
add_circt_dialect_doc(HLS hls)

set(LLVM_TARGET_DEFINITIONS HLS.td)

set(LLVM_TARGET_DEFINITIONS HLSPasses.td)
mlir_tablegen(HLSPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTHLSTransformsIncGen)
add_circt_doc(HLSPasses HLSPasses -gen-pass-doc)
