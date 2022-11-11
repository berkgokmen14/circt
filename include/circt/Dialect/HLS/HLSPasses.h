//===- CalyxPasses.h - Calyx pass entry points ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HLS_HLSPASSES_H
#define CIRCT_DIALECT_HLS_HLSPASSES_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <memory>

namespace circt {
namespace hls {

std::unique_ptr<mlir::Pass> createUnrollMarkedLoopsPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/HLS/HLSPasses.h.inc"

} // namespace hls
} // namespace circt

#endif // CIRCT_DIALECT_HLS_HLSPASSES_H
