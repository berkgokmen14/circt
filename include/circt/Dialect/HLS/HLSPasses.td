//===-- CalyxPasses.td - Calyx pass definition file --------*- tablegen -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the Calyx dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HLS_HLSPASSES_TD
#define CIRCT_DIALECT_HLS_HLSPASSES_TD

include "mlir/Pass/PassBase.td"

def UnrollMarkedLoops : Pass<"hls-unroll-loops", "mlir::func::FuncOp"> {
  let summary = "Unrolls loops with an `hls.unroll` attribute by a given factor.";
  let description = [{
  }];
  let dependentDialects = ["mlir::scf::SCFDialect,mlir::AffineDialect"];
  let constructor = "circt::hls::createUnrollMarkedLoopsPass()";
}

#endif // CIRCT_DIALECT_HLS_HLSPASSES_TD
