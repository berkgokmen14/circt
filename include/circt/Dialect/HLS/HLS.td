//===- Interop.td - Interop dialect definition -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Interop dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HLS_HLS
#define CIRCT_DIALECT_HLS_HLS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

//===----------------------------------------------------------------------===//
// HLS Dialect
//===----------------------------------------------------------------------===//

def HLSDialect : Dialect {
  let name = "hls";

  let summary = "HLS pragma-like passes";
  let description = [{
  }];
  let cppNamespace = "::circt::hls";
}

#endif // CIRCT_DIALECT_HLS_HLS
