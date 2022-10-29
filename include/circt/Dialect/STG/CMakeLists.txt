add_circt_dialect(STG stg)
add_circt_dialect_doc(STG stg)

set(LLVM_TARGET_DEFINITIONS STG.td)

set(LLVM_TARGET_DEFINITIONS STGPasses.td)
mlir_tablegen(STGPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTSTGTransformsIncGen)
add_circt_doc(STGPasses STGPasses -gen-pass-doc)
