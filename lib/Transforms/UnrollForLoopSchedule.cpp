//===- UnrollForLoopSchedule.cpp - Unroll nested loops for parallelism ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>

using namespace mlir;
using namespace mlir::affine;
using namespace circt::analysis;

namespace {
struct UnrollForLoopSchedule
    : public circt::UnrollForLoopScheduleBase<UnrollForLoopSchedule> {
  using UnrollForLoopScheduleBase<
      UnrollForLoopSchedule>::UnrollForLoopScheduleBase;
  void runOnOperation() override;

private:
  DenseSet<AffineMapAccessInterface> updateMemoryOps(AffineForOp affineFor);
  uint64_t getMinDepDistance(AffineForOp affineFor);
  std::optional<uint64_t> consumePragma(AffineForOp affineFor);
  LogicalResult unrollForDataParallel(AffineForOp affineFor);
  std::pair<AffineForOp, unsigned>
  getDeepestNestedForOp(std::pair<AffineForOp, unsigned> pair);
  LogicalResult unrollForPipelineParallel(AffineForOp affineFor);
  DenseMap<AffineForOp, DenseSet<AffineMapAccessInterface>> loopToMemOps;
  MemoryDependenceAnalysis *memDepAnalysis;
};
} // namespace

// We need to keep track of memory ops in within the body to understand the
// amount of pipeline parallelism that exists within
DenseSet<AffineMapAccessInterface>
UnrollForLoopSchedule::updateMemoryOps(AffineForOp affineFor) {
  DenseSet<AffineMapAccessInterface> memOps;
  auto &bodyRegion = affineFor.getLoopBody();
  // Only affine memory dependencies are supported
  auto affineLdOps = bodyRegion.getOps<AffineLoadOp>();
  auto affineStOps = bodyRegion.getOps<AffineStoreOp>();
  memOps.insert(affineLdOps.begin(), affineLdOps.end());
  memOps.insert(affineStOps.begin(), affineStOps.end());
  loopToMemOps[affineFor] = memOps;
  return memOps;
}

// Find the limiting carried dependence and get its distance
uint64_t UnrollForLoopSchedule::getMinDepDistance(AffineForOp affineFor) {
  auto memOps = loopToMemOps[affineFor];
  uint64_t minDistance = std::numeric_limits<uint64_t>::max();
  for (auto memOp : memOps) {

    ArrayRef<MemoryDependence> dependences =
        memDepAnalysis->getDependences(memOp);
    if (dependences.empty())
      continue;

    for (MemoryDependence memoryDep : dependences) {
      if (!hasDependence(memoryDep.dependenceType))
        continue;

      auto lb = memoryDep.dependenceComponents.back().lb;
      if (!lb.has_value())
        continue;
      int64_t distanceComp = *memoryDep.dependenceComponents.back().lb;
      minDistance =
          std::min(minDistance, distanceComp < 0 ? (uint64_t)distanceComp * -1
                                                 : (uint64_t)distanceComp);
    }
  }

  return minDistance;
}

// Get unroll factor from IR if it is there
std::optional<uint64_t>
UnrollForLoopSchedule::consumePragma(AffineForOp affineFor) {
  auto unrollAttr = affineFor->getAttr("hls.unroll");
  std::optional<uint64_t> maxUnrollFactor;
  if (unrollAttr != nullptr && unrollAttr.isa<StringAttr>()) {
    auto val = unrollAttr.cast<StringAttr>().getValue();
    if (val == "full")
      maxUnrollFactor = std::numeric_limits<uint64_t>::max();
  } else if (unrollAttr != nullptr && unrollAttr.isa<IntegerAttr>()) {
    maxUnrollFactor = unrollAttr.cast<IntegerAttr>().getUInt();
  }
  affineFor->removeAttr("hls.unroll");
  return maxUnrollFactor;
}

// Look for data-level parallelism towards the top level
LogicalResult
UnrollForLoopSchedule::unrollForDataParallel(AffineForOp affineFor) {

  std::optional<uint64_t> maxUnrollFactor = consumePragma(affineFor);

  // We are not checking for loop-carried dependencies
  // Looking for perfect-nesting so just use max value
  uint64_t unrollFactor =
      maxUnrollFactor.value_or(std::numeric_limits<uint64_t>::max());

  // If this loop nest is not perfectly nested return
  SmallVector<AffineForOp> innerLoopCount(affineFor.getOps<AffineForOp>());
  bool perfectlyNested =
      innerLoopCount.size() == 1 && isPerfectlyNested(SmallVector<AffineForOp>(
                                        {affineFor, innerLoopCount.back()}));
  if (!perfectlyNested)
    return success();

  // We need to know if we are fully-unrolling, because then the body will get
  // promoted to the containing region
  auto *containingRegion = affineFor->getParentRegion();
  bool unrollFull =
      unrollFactor >= getConstantTripCount(affineFor).value_or(
                          std::numeric_limits<uint64_t>::max() - 1);

  if (loopUnrollUpToFactor(affineFor, unrollFactor).failed())
    return failure();

  SmallVector<AffineForOp> innerLoops;
  // TODO(matth2k): If the loop got promoted, then we only want to grab the
  // loops that are related together. This should be trivial, but I have no way
  // to track which loops are clones. So this code will throw errors sometimes
  if (unrollFull)
    innerLoops.append(containingRegion->getOps<AffineForOp>().begin(),
                      containingRegion->getOps<AffineForOp>().end());
  else
    innerLoops.append(affineFor.getOps<AffineForOp>().begin(),
                      affineFor.getOps<AffineForOp>().end());

  auto destLoop = innerLoops.pop_back_val();
  OpBuilder builder(containingRegion);
  IRRewriter rewriter(builder);

  for (auto innerLoop : innerLoops) {
    destLoop.getRegion().getBlocks().back().getTerminator()->erase();
    rewriter.mergeBlocks(
        &innerLoop.getRegion().getBlocks().back(),
        &destLoop.getRegion().getBlocks().back(),
        destLoop.getRegion().getBlocks().back().getArguments());
  }

  for (auto innerLoop : innerLoops)
    innerLoop->erase();

  return success();
}

// This looks for pipeline parallelism via unrolling to reduce II
// TODO: this is not really the algorithm yet
LogicalResult
UnrollForLoopSchedule::unrollForPipelineParallel(AffineForOp affineFor) {
  updateMemoryOps(affineFor);
  uint64_t maxUnrollByDep = getMinDepDistance(affineFor) - 1;

  std::optional<uint64_t> maxUnrollFactor = consumePragma(affineFor);

  uint64_t unrollFactor =
      std::min(maxUnrollByDep,
               maxUnrollFactor.value_or(std::numeric_limits<uint64_t>::max()));

  if (loopUnrollUpToFactor(affineFor, unrollFactor).failed())
    return failure();

  return success();
}

std::pair<AffineForOp, unsigned> UnrollForLoopSchedule::getDeepestNestedForOp(
    std::pair<AffineForOp, unsigned> pair) {
  auto root = pair.first;
  auto rootDepth = pair.second;
  for (auto nested : root.getOps<AffineForOp>()) {
    auto recur = getDeepestNestedForOp(std::pair(nested, rootDepth + 1));
    if (recur.second > pair.second) {
      pair.second = recur.second;
      pair.first = recur.first;
    }
  }
  return pair;
}

void UnrollForLoopSchedule::runOnOperation() {

  // Gert scheduling analyses
  memDepAnalysis = &getAnalysis<MemoryDependenceAnalysis>();

  SmallVector<AffineForOp> rootForOps(getOperation().getOps<AffineForOp>());
  for (auto loop : rootForOps)
    if (unrollForDataParallel(loop).failed())
      return signalPassFailure();

  SmallVector<AffineForOp> opsToPipeline;
  for (auto loop : getOperation().getOps<AffineForOp>())
    opsToPipeline.push_back(getDeepestNestedForOp(std::pair(loop, 0)).first);

  for (auto loop : opsToPipeline)
    if (unrollForPipelineParallel(loop).failed())
      return signalPassFailure();
}

namespace circt {
std::unique_ptr<mlir::Pass> createUnrollForLoopSchedulePass() {
  return std::make_unique<UnrollForLoopSchedule>();
}
} // namespace circt
