//===- LoopScheduleInterfaces.td - LoopSchedule Interfaces -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the interfaces in the Calyx dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

// Op interface for scheduled loops.
def LoopOpInterface : OpInterface<"LoopInterface"> {
  let cppNamespace = "::circt::loopschedule";

  let description = [{
    This is an op interface for LoopSchedule scheduled loops.
  }];

  let methods = [
    InterfaceMethod<
      "Get the arguments passed to the body by the loop.",
      "Block::BlockArgListType",
      "getBodyArgs">,
    InterfaceMethod<
      "Get the initial values for iter args.",
      "Operation::operand_range",
      "getInits">,
    InterfaceMethod<
      "Get a pointer to the body block.",
      "Block*",
      "getBodyBlock">,
    InterfaceMethod<
      "Get a pointer to the condition block.",
      "Block*",
      "getConditionBlock">,
    InterfaceMethod<
      "Get the SSA value corresponding to the loop condition.",
      "Value",
      "getConditionValue">,
    InterfaceMethod<
      "Get the number of times a loop will run, if known statically.",
      "std::optional<uint64_t>",
      "getBound">,
    InterfaceMethod<
      "Returns if the loop is pipelined and false otherwise",
      "bool",
      "isPipelined">
  ];

  let verify = [{ return verifyLoop(op); }];
}

// Op interface for ops that describes a phase of the schedule.
def PhaseOpInterface : OpInterface<"PhaseInterface"> {
  let cppNamespace = "::circt::loopschedule";

  let description = [{
    This is an op interface for LoopSchedule ops that describes
    a phase of the schedule.
  }];

  let methods = [
    InterfaceMethod<
      "",
      "std::string",
      "getRegisterNamePrefix">,
    InterfaceMethod<
      "",
      "Block &",
      "getBodyBlock">,
    InterfaceMethod<
      "",
      "bool",
      "isStatic">,
    InterfaceMethod<
      "",
      "std::optional<uint64_t>",
      "getStartTime">
  ];
}
