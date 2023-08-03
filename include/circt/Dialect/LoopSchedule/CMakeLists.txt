add_circt_dialect(LoopSchedule loopschedule)
add_circt_dialect_doc(LoopSchedule loopschedule)

set(LLVM_TARGET_DEFINITIONS LoopScheduleInterfaces.td)
mlir_tablegen(LoopScheduleInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(LoopScheduleInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRLoopScheduleInterfacesIncGen)
add_dependencies(circt-headers MLIRLoopScheduleInterfacesIncGen)
