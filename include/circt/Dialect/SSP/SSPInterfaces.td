//===- SSPInterfaces.td - SSP Interfaces -------------*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the interfaces in the SSP dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"

// Op interface for allocations.
def AllocOpInterface : OpInterface<"AllocInterface"> {
  let cppNamespace = "::circt::ssp";

  let description = [{
    This is an op interface for alloc ops to be lowered.
  }];
}

// Op interface for loads.
def LoadOpInterface : OpInterface<"LoadInterface"> {
  let cppNamespace = "::circt::ssp";

  let description = [{
    This is an op interface for schedulable loads.
  }];

  let methods = [
    InterfaceMethod<
      "This returns the latency of the load.",
      "std::optional<unsigned int>",
      "getLatency",
      (ins)>,
    InterfaceMethod<
      "This returns the limit of the load.",
      "std::optional<unsigned int>",
      "getLimit",
      (ins)>,
    InterfaceMethod<
      "This returns the unique identifier of the associated memory",
      "std::string",
      "getUniqueId",
      (ins)>
    /* InterfaceMethod< */
    /*   "This returns the value of the associated memory", */
    /*   "mlir::Value", */
    /*   "getMemoryValue", */
    /*   (ins)>, */
    /* InterfaceMethod< */
    /*   "This returns the value of the associated memory", */
    /*   "mlir::Operation::operand_range", */
    /*   "getIndices", */
    /*   (ins)>, */
    /* InterfaceMethod< */
    /*   "This returns the value of the associated memory", */
    /*   "mlir::Value", */
    /*   "getResult", */
    /*   (ins)>, */
    /* InterfaceMethod< */
    /*   "This returns the value of the associated memory", */
    /*   "mlir::Value", */
    /*   "connectToMemInterface", */
    /*   (ins "mlir::OpBuilder":$rewriter, "calyx::ComponentLoweringStateInterface":$state)> */
  ];
  /* let verify = [{ return verifyGroupInterface(op); }]; */
}

// Op interface for stores.
def StoreOpInterface : OpInterface<"StoreInterface"> {
  let cppNamespace = "::circt::ssp";

  let description = [{
    This is an op interface for schedulable stores.
  }];

  let methods = [
    InterfaceMethod<
      "This returns the latency of the store.",
      "std::optional<unsigned int>",
      "getLatency",
      (ins)>,
    InterfaceMethod<
      "This returns the limit of the store.",
      "std::optional<unsigned int>",
      "getLimit",
      (ins)>,
    InterfaceMethod<
      "This returns the unique identifier of the associated memory",
      "std::string",
      "getUniqueId",
      (ins)>
    /* InterfaceMethod< */
    /*   "This returns the value of the associated memory", */
    /*   "mlir::Value", */
    /*   "getMemoryValue", */
    /*   (ins)>, */
    /* InterfaceMethod< */
    /*   "This returns the value of the associated memory", */
    /*   "mlir::Operation::operand_range", */
    /*   "getIndices", */
    /*   (ins)>, */
    /* InterfaceMethod< */
    /*   "This returns the value of the associated memory", */
    /*   "mlir::Value", */
    /*   "connectToMemInterface", */
    /*   (ins "mlir::OpBuilder":$rewriter, "calyx::ComponentLoweringStateInterface":$state)> */
  ];
  /* let verify = [{ return verifyGroupInterface(op); }]; */
}

// Op interface for AffineLoads.
def SchedulableAffineOpInterface : OpInterface<"SchedulableAffineInterface"> {
  let cppNamespace = "::circt::ssp";

  let description = [{
    This is an op interface for schedulable loads.
  }];

  let methods = [
    InterfaceMethod<
      "Creates a new non-affine op equivalent to the original Op",
      "Operation*",
      "createNonAffineOp",
      (ins "mlir::OpBuilder":$builder, "mlir::ValueRange":$indices)>
  ];
}
